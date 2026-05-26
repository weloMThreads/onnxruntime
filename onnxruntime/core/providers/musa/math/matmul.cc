// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/common.h"
#include "core/providers/musa/math/matmul.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/musa_execution_provider.h"
#include <algorithm>
#include <musa.h>
#include <musa_runtime.h>
#include <mudnn.h>
#include <mublas.h>
#include <string>
#include <vector>

using onnxruntime::common::Status;
namespace onnxruntime {
namespace musa {

// ============================================================================
// muBLAS Handle Pooling - Thread-local handle with stream reset on each use
// Key insight: torch_musa and CUDA EP both reset stream on every handle use
// ============================================================================
namespace {
thread_local mublasHandle_t g_mublas_handle = nullptr;
thread_local bool g_mublas_initialized = false;

#if MUSA_VERSION >= 50100
constexpr mublasMath kMublasTensorOpMathMode = MUBLAS_TF32_TENSOR_OP_MATH;
#else
constexpr mublasMath kMublasTensorOpMathMode = MUBLAS_TP32_TENSOR_OP_MATH;
#endif

// Get pooled muBLAS handle with stream binding
// CRITICAL: mublasSetStream must be called on every use to ensure correct stream context
mublasHandle_t GetMublasHandle(musaStream_t stream) {
  if (!g_mublas_initialized) {
    mublasCreate(&g_mublas_handle);
    g_mublas_initialized = true;
  }
  // Reset stream on every use - this is the key fix for handle pooling
  mublasSetStream(g_mublas_handle, stream);
  // Enable TF32 for FP32 acceleration on Matrix Unit
  // Following CUDA EP pattern: use_tf32{true} with the SDK-specific tensor-op math mode.
  mublasSetMathMode(g_mublas_handle, kMublasTensorOpMathMode);
  return g_mublas_handle;
}

// Release stream binding after muBLAS operation to avoid conflicts with muDNN
void ReleaseMublasStream() {
  if (g_mublas_initialized && g_mublas_handle != nullptr) {
    // Unbind stream to prevent resource conflicts with other libraries (muDNN)
    mublasSetStream(g_mublas_handle, nullptr);
  }
}
}  // anonymous namespace

template <typename T>
Status MatMul<T>::ComputeInternal(OpKernelContext* ctx) const {
  const auto* A = ctx->Input<Tensor>(0);
  const auto* B = ctx->Input<Tensor>(1);

  if (!A || !B) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input tensors");
  }

  // Handle 1D vectors - disable transpose for 1D inputs (same as CudaEP)
  bool trans_a = trans_A_;
  bool trans_b = trans_B_;
  if (A->Shape().NumDimensions() == 1) trans_a = false;
  if (B->Shape().NumDimensions() == 1) trans_b = false;

  // Use standard MatMulComputeHelper for dimension calculations
  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(A->Shape(), B->Shape(), trans_a, trans_b,
                                     false, false, false));

  // Create output tensor
  Tensor* Y = ctx->Output(0, helper.OutputShape());
  if (!Y) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Failed to create output tensor");
  }

  // Handle empty output case
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  // Handle K=0 case (empty input dimension)
  if (helper.K() == 0) {
    // For K=0 case, the mathematical result should be zero matrix
    // We need to explicitly zero the output tensor
    auto* stream = Stream(ctx);
    musaError_t musa_status;

    if (stream) {
      musa_status = musaMemsetAsync(Y->MutableDataRaw(), 0, Y->SizeInBytes(), stream);
    } else {
      musa_status = musaMemset(Y->MutableDataRaw(), 0, Y->SizeInBytes());
    }

    if (musa_status != musaSuccess) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to zero output tensor for K=0 case: " + std::string(musaGetErrorString(musa_status)));
    }

    return Status::OK();
  }

  // Prepare MUSA operation - use EP mode for thread-safe Handle management
  const auto* ep = static_cast<const MusaExecutionProvider*>(
      Info().GetExecutionProvider());
  MusaPreparation prepare(ep);
  ORT_RETURN_IF_ERROR(PrepareMatMul(ctx, prepare, helper));

  // Select and execute compute strategy
  auto strategy = SelectStrategy(A->Shape(), B->Shape(), A->DataType(),
                                 helper.OutputOffsets().size());

  auto* ort_stream = ctx->GetComputeStream();

  Status result;
  switch (strategy) {
    case ComputeStrategy::MU_DNN_BATCH:
      result = ExecuteWithMuDNN(prepare, helper, true, ort_stream);
      break;
    case ComputeStrategy::MU_DNN_MATMUL:
      result = ExecuteWithMuDNN(prepare, helper, false, ort_stream);
      break;
    case ComputeStrategy::MU_BLAS_GEMM:
      result = ExecuteWithMuBLAS(prepare, helper, Stream(ctx));
      break;
    case ComputeStrategy::MU_DNN_LOOP:
      result = ExecuteWithMuDNNLoop(ctx, helper, ort_stream);
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unknown compute strategy");
  }

  return result;
}

template <typename T>
Status MatMul<T>::PrepareMatMul(OpKernelContext* ctx, MusaPreparation& prepare,
                                const MatMulComputeHelper& helper) const {
  const Tensor* A = ctx->Input<Tensor>(0);
  const Tensor* B = ctx->Input<Tensor>(1);
  Tensor* Y = ctx->Output(0, helper.OutputShape());

  // Basic validation
  if (!A || !B || !Y) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid tensors");
  }

  // Store tensor pointers
  prepare.input_a_ptr = A->DataRaw();
  prepare.input_b_ptr = B->DataRaw();
  prepare.output_ptr = Y->MutableDataRaw();
  prepare.output_size = Y->Shape().Size();

  // Get MUSA data type
  const auto musaType = GetMusaDataType<T>();

  // Set stream
  auto* stream = Stream(ctx);
  if (prepare.handle && stream) {
    auto status = prepare.handle->SetStream(stream);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set MUSA stream");
    }
  }

  // Prepare tensors
  size_t batch_count = helper.OutputOffsets().size();
  prepare.inputTensors.resize(2);
  prepare.outputTensors.resize(1);

  // Check for 4D->3D reshape scenario (Attention pattern)
  bool use_4d_to_3d_reshape = CanReshape4DTo3D(A->Shape(), B->Shape());

  if (use_4d_to_3d_reshape) {
    // 4D -> 3D reshape for MuDNN BatchMatMul
    // {A,B,M,K} @ {A,B,K,N} -> {A*B,M,K} @ {A*B,K,N}
    ORT_RETURN_IF_ERROR(SetupMusaTensorWithReshape(
        prepare.inputTensors[0], A, musaType, &prepare, 3));
    ORT_RETURN_IF_ERROR(SetupMusaTensorWithReshape(
        prepare.inputTensors[1], B, musaType, &prepare, 3));
    ORT_RETURN_IF_ERROR(SetupMusaTensorWithReshape(
        prepare.outputTensors[0], Y, musaType, &prepare, 3));
  } else if (batch_count == 1) {
    // Single matrix multiplication
    bool a_can_reshape = CanReshapeTo2D(A->Shape(), batch_count);
    bool b_can_reshape = CanReshapeTo2D(B->Shape(), batch_count);

    if (a_can_reshape && b_can_reshape) {
      // Use reshape-aware tensor setup for MuDNN
      // 3D -> 2D: target_dims=2
      int target_a = (A->Shape().NumDimensions() == 3) ? 2 : 0;
      int target_b = (B->Shape().NumDimensions() == 3) ? 2 : 0;
      int target_y = (Y->Shape().NumDimensions() == 3) ? 2 : 0;

      ORT_RETURN_IF_ERROR(SetupMusaTensorWithReshape(
          prepare.inputTensors[0], A, musaType, &prepare, target_a));
      ORT_RETURN_IF_ERROR(SetupMusaTensorWithReshape(
          prepare.inputTensors[1], B, musaType, &prepare, target_b));
      ORT_RETURN_IF_ERROR(SetupMusaTensorWithReshape(
          prepare.outputTensors[0], Y, musaType, &prepare, target_y));
    } else {
      // Standard tensor setup for 2D matrices
      ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.inputTensors[0], A, musaType, &prepare));
      ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.inputTensors[1], B, musaType, &prepare));
      ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.outputTensors[0], Y, musaType, &prepare));
    }
  } else {
    // Batch matrix multiplication - use standard tensor setup
    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.inputTensors[0], A, musaType, &prepare));
    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.inputTensors[1], B, musaType, &prepare));
    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.outputTensors[0], Y, musaType, &prepare));
  }

  return Status::OK();
}

template <typename T>
bool MatMul<T>::CanReshapeTo2D(const TensorShape& shape, size_t batch_count) const {
  // Only consider cases with batch_count=1 (simple matrix multiplication)
  if (batch_count != 1) {
    return false;
  }

  // Accept both 2D and 3D tensors
  if (shape.NumDimensions() != 2 && shape.NumDimensions() != 3) {
    return false;
  }

  // Check if all dimensions are valid (no zero or negative dimensions)
  for (int64_t dim : shape.GetDims()) {
    if (dim <= 0) {
      return false;
    }
  }

  // 2D tensors are already compatible
  // 3D tensors like [A,B,C] can be safely reshaped to [A*B,C] or [A,B*C] for MuDNN
  // This is safe because batch_count=1 means no complex broadcasting
  return true;
}

template <typename T>
bool MatMul<T>::CanReshape4DTo3D(const TensorShape& a_shape,
                                 const TensorShape& b_shape) const {
  // Condition 1: Both must be 4D tensors
  if (a_shape.NumDimensions() != 4 || b_shape.NumDimensions() != 4) {
    return false;
  }

  // Condition 2: Batch dimensions must match exactly (no broadcasting)
  // {2,8,50,64} @ {2,8,64,50} ✓
  // {2,8,50,64} @ {2,1,64,50} ✗
  if (a_shape[0] != b_shape[0] || a_shape[1] != b_shape[1]) {
    return false;
  }

  // Condition 3: All dimensions must be valid (positive)
  for (size_t i = 0; i < 4; ++i) {
    if (a_shape[i] <= 0 || b_shape[i] <= 0) {
      return false;
    }
  }

  return true;
}

template <typename T>
Status MatMul<T>::SetupMusaTensorWithReshape(::musa::dnn::Tensor& musa_tensor,
                                             const Tensor* ort_tensor,
                                             ::musa::dnn::Tensor::Type data_type,
                                             MusaPreparation* preparation,
                                             int target_dims) const {
  const auto& original_shape = ort_tensor->Shape();
  const auto& dims = original_shape.GetDims();
  int num_dims = static_cast<int>(original_shape.NumDimensions());

  // No reshape needed if already at target dims or target_dims is invalid
  if (num_dims == target_dims || target_dims <= 0) {
    return SetupMusaTensor(musa_tensor, ort_tensor, data_type, preparation);
  }

  std::vector<int64_t> new_dims;

  if (num_dims == 3 && target_dims == 2) {
    // 3D -> 2D: {A,B,C} -> {A*B, C}
    new_dims = {dims[0] * dims[1], dims[2]};
  } else if (num_dims == 4 && target_dims == 3) {
    // 4D -> 3D: {A,B,M,K} -> {A*B, M, K}
    new_dims = {dims[0] * dims[1], dims[2], dims[3]};
  } else {
    // Unsupported reshape, use standard setup
    return SetupMusaTensor(musa_tensor, ort_tensor, data_type, preparation);
  }

  // Set tensor type
  auto status = musa_tensor.SetType(data_type);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to set MUSA tensor type for reshaped tensor, status: " +
                               std::to_string(static_cast<int>(status)));
  }

  // Set data address (same as original tensor - zero-copy reshape)
  const void* data_ptr = ort_tensor->DataRaw();
  if (data_ptr) {
    status = musa_tensor.SetAddr(data_ptr);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set MUSA tensor address for reshaped tensor, status: " +
                                 std::to_string(static_cast<int>(status)));
    }
  }

  // Set format - use NCHW for compatibility
  status = musa_tensor.SetFormat(::musa::dnn::Tensor::Format::NCHW);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to set MUSA tensor format for reshaped tensor, status: " +
                               std::to_string(static_cast<int>(status)));
  }

  // Set the new shape
  status = musa_tensor.SetNdInfo(static_cast<int>(new_dims.size()), new_dims.data());
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to set MUSA tensor shape info for reshaped tensor, status: " +
                               std::to_string(static_cast<int>(status)));
  }

  return Status::OK();
}

template <typename T>
typename MatMul<T>::ComputeStrategy MatMul<T>::SelectStrategy(
    const TensorShape& a_shape, const TensorShape& b_shape,
    MLDataType dtype, size_t batch_count) const {
  if (CanReshape4DTo3D(a_shape, b_shape)) {
    return ComputeStrategy::MU_DNN_BATCH;
  }

  // For complex broadcasting (batch_count == 0):
  // Use muDNN Loop to handle per-batch MatMul
  if (batch_count == 0) {
    return ComputeStrategy::MU_DNN_LOOP;
  }

  if (batch_count == 1) {
    bool a_can_reshape = CanReshapeTo2D(a_shape, batch_count);
    bool b_can_reshape = CanReshapeTo2D(b_shape, batch_count);

    if (a_can_reshape && b_can_reshape) {
      return ComputeStrategy::MU_DNN_MATMUL;
    }
  }

  // For high-dimensional inputs (>2D) with batch_count > 0 (no complex broadcasting):
  // Use muDNN BatchMatMul with TENSOR mode for both FP32 and FP16
  if (a_shape.NumDimensions() > 2 || b_shape.NumDimensions() > 2) {
    return ComputeStrategy::MU_DNN_BATCH;
  }

  if (batch_count > 1) {
    return ComputeStrategy::MU_DNN_BATCH;
  }

  if (trans_A_ || trans_B_) {
    return ComputeStrategy::MU_DNN_MATMUL;
  }

  return ComputeStrategy::MU_DNN_MATMUL;
}

template <typename T>
Status MatMul<T>::ExecuteWithMuDNN(const MusaPreparation& prepare,
                                   const MatMulComputeHelper& helper, bool use_batch,
                                   onnxruntime::Stream* ort_stream) const {
  if (use_batch) {
    // Use BatchMatMul
    ::musa::dnn::BatchMatMul batch_op;

    // Set parameters - using default compute mode
    auto status = ::musa::dnn::Status::SUCCESS;

    status = batch_op.SetAlpha(static_cast<double>(alpha_));
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set BatchMatMul alpha");
    }

    status = batch_op.SetBeta(0.0);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set BatchMatMul beta");
    }

    status = batch_op.SetTranspose(trans_A_, trans_B_);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set BatchMatMul transpose");
    }

    // Enable Tensor Core / Matrix Unit for better performance (torch_musa pattern)
    status = batch_op.SetComputeMode(::musa::dnn::BatchMatMul::ComputeMode::TENSOR);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set BatchMatMul compute mode");
    }

    std::vector<IAllocatorUniquePtr<void>> workspace_buffers_holder;

    auto memory_allocator = [this, ort_stream, &workspace_buffers_holder](size_t size) -> ::musa::dnn::MemoryHandler {
      if (size == 0) {
        return ::musa::dnn::MemoryHandler(nullptr, [](void*) {});
      }
      auto scratch = this->GetScratchBuffer<void>(size, ort_stream);
      void* ptr = scratch.get();
      workspace_buffers_holder.push_back(std::move(scratch));
      return ::musa::dnn::MemoryHandler(ptr, [](void*) {});
    };

    // Execute computation with MemoryHandler (torch_musa pattern)
    status = batch_op.Run(prepare.GetHandle(),
                          const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]),
                          prepare.inputTensors[0], prepare.inputTensors[1],
                          memory_allocator);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "BatchMatMul computation failed");
    }
  } else {
    // Use standard MatMul
    ::musa::dnn::MatMul matmul_op;

    // Set parameters
    auto status = matmul_op.SetAlpha(static_cast<double>(alpha_));
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set MatMul alpha");
    }

    status = matmul_op.SetBeta(0.0);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set MatMul beta");
    }

    status = matmul_op.SetTranspose(trans_A_, trans_B_);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set MatMul transpose");
    }

    // Enable Tensor Core / Matrix Unit for better performance (torch_musa pattern)
    status = matmul_op.SetComputeMode(::musa::dnn::MatMul::ComputeMode::TENSOR);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set MatMul compute mode");
    }

    std::vector<IAllocatorUniquePtr<void>> workspace_buffers_holder;

    auto memory_allocator = [this, ort_stream, &workspace_buffers_holder](size_t size) -> ::musa::dnn::MemoryHandler {
      if (size == 0) {
        return ::musa::dnn::MemoryHandler(nullptr, [](void*) {});
      }
      auto scratch = this->GetScratchBuffer<void>(size, ort_stream);
      void* ptr = scratch.get();
      workspace_buffers_holder.push_back(std::move(scratch));
      return ::musa::dnn::MemoryHandler(ptr, [](void*) {});
    };

    // Execute computation with MemoryHandler (torch_musa pattern)
    status = matmul_op.Run(prepare.GetHandle(),
                           const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]),
                           prepare.inputTensors[0], prepare.inputTensors[1],
                           memory_allocator);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MatMul computation failed");
    }
  }

  return Status::OK();
}

template <typename T>
Status MatMul<T>::ExecuteWithMuBLAS(const MusaPreparation& prepare,
                                    const MatMulComputeHelper& helper,
                                    musaStream_t stream) const {
  // Get pooled muBLAS handle with stream binding (key optimization)
  mublasHandle_t mublas_handle = GetMublasHandle(stream);
  mublasStatus_t status;

  // Get GEMM parameters from helper
  int64_t M = helper.M();
  int64_t N = helper.N();
  int64_t K = helper.K();

  // Check if this is a complex broadcasting scenario that needs offset processing
  const size_t initial_batch_count = helper.OutputOffsets().size();

  if (initial_batch_count == 0) {
    // Complex broadcasting case: need to fill offsets
    const_cast<MatMulComputeHelper&>(helper).FillOffsets();
  }

  // Get batch information and offsets
  const auto& output_offsets = helper.OutputOffsets();
  const auto& left_offsets = helper.LeftOffsets();
  const auto& right_offsets = helper.RightOffsets();

  // Set up GEMM operation parameters
  mublasOperation_t transA = trans_A_ ? MUBLAS_OP_T : MUBLAS_OP_N;
  mublasOperation_t transB = trans_B_ ? MUBLAS_OP_T : MUBLAS_OP_N;

  // Get data pointers
  const T* A_data = static_cast<const T*>(prepare.input_a_ptr);
  const T* B_data = static_cast<const T*>(prepare.input_b_ptr);
  T* C_data = static_cast<T*>(prepare.output_ptr);

  // Set alpha and beta values
  const float alpha_val = static_cast<float>(alpha_);
  const float beta_val = 0.0f;

  const size_t batch_count = output_offsets.size();

  if (batch_count == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "No batches to process");
  }

  if (batch_count == 1) {
    // Simple case: single GEMM operation, no offset processing needed
    if constexpr (std::is_same_v<T, float>) {
      // Calculate leading dimensions based on transpose flags
      int lda = trans_A_ ? static_cast<int>(M) : static_cast<int>(K);
      int ldb = trans_B_ ? static_cast<int>(K) : static_cast<int>(N);
      int ldc = static_cast<int>(N);

      status = mublasSgemm(mublas_handle,
                           transA, transB,
                           static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
                           &alpha_val,
                           B_data, ldb,
                           A_data, lda,
                           &beta_val,
                           C_data, ldc);

      if (status != MUBLAS_STATUS_SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "muBLAS GEMM computation failed");
      }

    } else if constexpr (std::is_same_v<T, MLFloat16>) {
      int lda = trans_A_ ? static_cast<int>(M) : static_cast<int>(K);
      int ldb = trans_B_ ? static_cast<int>(K) : static_cast<int>(N);
      int ldc = static_cast<int>(N);

      status = mublasGemmEx(mublas_handle,
                            transA, transB,
                            static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
                            &alpha_val,
                            B_data, MUSA_R_16F, ldb,
                            A_data, MUSA_R_16F, lda,
                            &beta_val,
                            C_data, MUSA_R_16F, ldc,
                            MUBLAS_COMPUTE_32F, MUBLAS_GEMM_DEFAULT_TENSOR_OP);

      if (status != MUBLAS_STATUS_SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "muBLAS GemmEx computation failed for MLFloat16");
      }

    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported data type for muBLAS GEMM");
    }
  } else {
    // Complex case: multiple batches, process with offsets
    if (left_offsets.size() != batch_count || right_offsets.size() != batch_count) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Mismatched offset sizes");
    }

    for (size_t batch = 0; batch < batch_count; ++batch) {
      // Calculate data pointers for current batch
      const T* A_batch = A_data + left_offsets[batch];
      const T* B_batch = B_data + right_offsets[batch];
      T* C_batch = C_data + output_offsets[batch];

      if constexpr (std::is_same_v<T, float>) {
        // Calculate leading dimensions based on transpose flags
        int lda = trans_A_ ? static_cast<int>(M) : static_cast<int>(K);
        int ldb = trans_B_ ? static_cast<int>(K) : static_cast<int>(N);
        int ldc = static_cast<int>(N);

        status = mublasSgemm(mublas_handle,
                             transA, transB,
                             static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
                             &alpha_val,
                             B_batch, ldb,
                             A_batch, lda,
                             &beta_val,
                             C_batch, ldc);

        if (status != MUBLAS_STATUS_SUCCESS) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "muBLAS GEMM computation failed for batch " + std::to_string(batch));
        }

      } else if constexpr (std::is_same_v<T, MLFloat16>) {
        int lda = trans_A_ ? static_cast<int>(M) : static_cast<int>(K);
        int ldb = trans_B_ ? static_cast<int>(K) : static_cast<int>(N);
        int ldc = static_cast<int>(N);

        status = mublasGemmEx(mublas_handle,
                              transA, transB,
                              static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
                              &alpha_val,
                              B_batch, MUSA_R_16F, ldb,
                              A_batch, MUSA_R_16F, lda,
                              &beta_val,
                              C_batch, MUSA_R_16F, ldc,
                              MUBLAS_COMPUTE_32F, MUBLAS_GEMM_DEFAULT_TENSOR_OP);

        if (status != MUBLAS_STATUS_SUCCESS) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "muBLAS GemmEx computation failed for MLFloat16 batch " + std::to_string(batch));
        }

      } else {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported data type for muBLAS GEMM");
      }
    }
  }

  // Release stream binding to avoid conflicts with muDNN operations
  ReleaseMublasStream();

  // Handle is pooled and reused, no cleanup needed (system will reclaim on process exit)
  return Status::OK();
}

template <typename T>
Status MatMul<T>::ExecuteWithMuDNNLoop(OpKernelContext* ctx,
                                       const MatMulComputeHelper& helper,
                                       onnxruntime::Stream* ort_stream) const {
  const Tensor* A = ctx->Input<Tensor>(0);
  const Tensor* B = ctx->Input<Tensor>(1);
  Tensor* Y = ctx->Output(0, helper.OutputShape());

  if (!A || !B || !Y) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid tensors");
  }

  MatMulComputeHelper& mutable_helper = const_cast<MatMulComputeHelper&>(helper);
  if (helper.OutputOffsets().empty()) {
    mutable_helper.FillOffsets();
  }

  const auto& left_offsets = helper.LeftOffsets();
  const auto& right_offsets = helper.RightOffsets();
  const auto& output_offsets = helper.OutputOffsets();

  if (output_offsets.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "No batches to process");
  }

  const T* A_data = A->Data<T>();
  const T* B_data = B->Data<T>();
  T* Y_data = Y->MutableData<T>();

  int64_t M = helper.M();
  int64_t N = helper.N();
  int64_t K = helper.K();

  const auto* ep = static_cast<const MusaExecutionProvider*>(Info().GetExecutionProvider());
  auto* musa_stream = Stream(ctx);

  // Move preparation and MatMul op outside loop for better performance
  // (avoid repeated object creation and configuration per batch)
  MusaPreparation prepare(ep);
  if (prepare.handle && musa_stream) {
    prepare.handle->SetStream(musa_stream);
  }

  ::musa::dnn::MatMul matmul_op;
  matmul_op.SetAlpha(static_cast<double>(alpha_));
  matmul_op.SetBeta(0.0);
  matmul_op.SetTranspose(trans_A_, trans_B_);
  matmul_op.SetComputeMode(::musa::dnn::MatMul::ComputeMode::TENSOR);

  // Pre-configure tensor dimensions (only addresses change per batch)
  const auto musa_type = GetMusaDataType<T>();
  ::musa::dnn::Tensor a_tensor, b_tensor, y_tensor;
  a_tensor.SetType(musa_type);
  a_tensor.SetNdInfo({static_cast<int>(M), static_cast<int>(K)}, {static_cast<int>(K), 1});
  b_tensor.SetType(musa_type);
  b_tensor.SetNdInfo({static_cast<int>(K), static_cast<int>(N)}, {static_cast<int>(N), 1});
  y_tensor.SetType(musa_type);
  y_tensor.SetNdInfo({static_cast<int>(M), static_cast<int>(N)}, {static_cast<int>(N), 1});

  // Workspace holder for all batches
  std::vector<IAllocatorUniquePtr<void>> workspace_holder;
  auto mem_alloc = [this, ort_stream, &workspace_holder](size_t size) -> ::musa::dnn::MemoryHandler {
    if (size == 0) return ::musa::dnn::MemoryHandler(nullptr, [](void*) {});
    auto buf = this->GetScratchBuffer<void>(size, ort_stream);
    void* ptr = buf.get();
    workspace_holder.push_back(std::move(buf));
    return ::musa::dnn::MemoryHandler(ptr, [](void*) {});
  };

  for (size_t batch = 0; batch < output_offsets.size(); ++batch) {
    const T* A_batch = A_data + left_offsets[batch];
    const T* B_batch = B_data + right_offsets[batch];
    T* Y_batch = Y_data + output_offsets[batch];

    // Only update addresses per batch (tensor metadata already set)
    a_tensor.SetAddr(const_cast<T*>(A_batch));
    b_tensor.SetAddr(const_cast<T*>(B_batch));
    y_tensor.SetAddr(Y_batch);

    auto status = matmul_op.Run(prepare.GetHandle(), y_tensor, a_tensor, b_tensor, mem_alloc);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "muDNN MatMul failed in loop at batch " + std::to_string(batch));
    }
  }

  return Status::OK();
}

// Macro for registering typed kernel
#define REGISTER_MUSA_MATMUL_TYPED_KERNEL(ver, T)                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      MatMul, kOnnxDomain, ver, T, kMusaExecutionProvider,        \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      MatMul<T>);

// Versioned macro for registering typed kernel
#define REGISTER_MUSA_MATMUL_VERSIONED_TYPED_KERNEL(startver, endver, T) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                               \
      MatMul, kOnnxDomain, startver, endver, T, kMusaExecutionProvider,  \
      (*KernelDefBuilder::Create())                                      \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),        \
      MatMul<T>);

// Combined macro for both kernel types (HF = Half, Float)
#define REGISTER_MUSA_MATMUL_TYPED(ver, T) REGISTER_MUSA_MATMUL_TYPED_KERNEL(ver, T)
#define REGISTER_MUSA_MATMUL_VERSIONED_TYPED(startver, endver, T) REGISTER_MUSA_MATMUL_VERSIONED_TYPED_KERNEL(startver, endver, T)

#define REGISTER_MUSA_MATMUL_HF(ver)         \
  REGISTER_MUSA_MATMUL_TYPED(ver, MLFloat16) \
  REGISTER_MUSA_MATMUL_TYPED(ver, float)

#define REGISTER_MUSA_MATMUL_VERSIONED_HF(startver, endver)         \
  REGISTER_MUSA_MATMUL_VERSIONED_TYPED(startver, endver, MLFloat16) \
  REGISTER_MUSA_MATMUL_VERSIONED_TYPED(startver, endver, float)

// Integer types are not supported by MUSA mudnn MatMul operations
// Only float and MLFloat16 types are supported

// Register operations for different versions
REGISTER_MUSA_MATMUL_VERSIONED_HF(1, 8)  // ONNX v1-8 (float, MLFloat16)

REGISTER_MUSA_MATMUL_VERSIONED_HF(9, 12)  // ONNX v9-12 (float, MLFloat16)

REGISTER_MUSA_MATMUL_HF(13)  // ONNX v13+ (float, MLFloat16)

// Explicit template instantiation - only supported types
template class MatMul<float>;
template class MatMul<MLFloat16>;

}  // namespace musa
}  // namespace onnxruntime
