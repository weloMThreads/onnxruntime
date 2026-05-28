// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/common.h"
#include "core/providers/musa/math/gemm.h"
#include "core/providers/musa/musa_call.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/musa_execution_provider.h"
#include "core/providers/musa/musa_utils.h"
#include <algorithm>
#include <musa_runtime.h>
#include <mudnn.h>
#include <string>
#include <vector>

using onnxruntime::common::Status;
namespace onnxruntime {
namespace musa {

template <typename T>
Status GemmZeroK(musaStream_t stream, const MusaExecutionProvider* ep, const Tensor* C, Tensor* Y, float beta) {
  const size_t output_bytes = static_cast<size_t>(Y->Shape().Size()) * sizeof(T);
  MUSA_RETURN_IF_ERROR(musaMemsetAsync(Y->MutableDataRaw(), 0, output_bytes, stream));

  if (C == nullptr || beta == 0.0f) {
    return Status::OK();
  }

  MusaPreparation prepare(ep);
  if (prepare.handle && stream) {
    auto status = prepare.handle->SetStream(stream);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set MUSA stream for Gemm K=0 path, status: ",
                             static_cast<int>(status));
    }
  }

  ::musa::dnn::Tensor output_tensor;
  ::musa::dnn::Tensor bias_tensor;
  const auto musa_type = GetMusaDataType<T>();
  ORT_RETURN_IF_ERROR(SetupMusaTensor(output_tensor, Y, musa_type, &prepare));
  ORT_RETURN_IF_ERROR(SetupMusaTensor(bias_tensor, C, musa_type, &prepare));

  ::musa::dnn::Binary binary_op;
  auto status = binary_op.SetMode(::musa::dnn::Binary::Mode::ADD_ALPHA);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set Binary mode for Gemm K=0 path");
  }
  status = binary_op.SetAlpha(static_cast<double>(beta));
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set Binary alpha for Gemm K=0 path");
  }
  status = binary_op.Run(prepare.GetHandle(), output_tensor, output_tensor, bias_tensor);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "mudnn Binary ADD_ALPHA failed in Gemm K=0 path, status: ",
                           static_cast<int>(status));
  }

  return Status::OK();
}

template <typename T>
Status Gemm<T>::ApplyActivationInPlace(MusaPreparation& prepare) const {
  if (activation_.empty()) {
    return Status::OK();
  }

  if (prepare.outputTensors.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "FusedGemm activation missing output tensor");
  }

  ::musa::dnn::Unary::Mode mode;
  if (activation_ == "Relu") {
    mode = ::musa::dnn::Unary::Mode::RELU;
  } else if (activation_ == "Tanh") {
    mode = ::musa::dnn::Unary::Mode::TANH;
  } else if (activation_ == "Sigmoid") {
    mode = ::musa::dnn::Unary::Mode::SIGMOID;
  } else if (activation_ == "LeakyRelu") {
    mode = ::musa::dnn::Unary::Mode::LEAKY_RELU;
  } else if (activation_ == "Softplus") {
    mode = ::musa::dnn::Unary::Mode::SOFTPLUS;
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Unsupported MUSA FusedGemm activation: ", activation_);
  }

  ::musa::dnn::Unary unary_op;
  auto status = unary_op.SetMode(mode);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to set FusedGemm activation mode, status: ",
                           static_cast<int>(status));
  }

  if (activation_ == "LeakyRelu") {
    status = unary_op.SetAlpha(static_cast<double>(activation_alpha_));
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set FusedGemm LeakyRelu alpha, status: ",
                             static_cast<int>(status));
    }
  } else if (activation_ == "Softplus") {
    status = unary_op.SetAlpha(activation_alpha_ == 0.0f ? 1.0 : static_cast<double>(activation_alpha_));
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set FusedGemm Softplus alpha, status: ",
                             static_cast<int>(status));
    }
    status = unary_op.SetBeta(activation_beta_ == 0.0f ? 20.0 : static_cast<double>(activation_beta_));
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set FusedGemm Softplus beta, status: ",
                             static_cast<int>(status));
    }
  }

  status = unary_op.Run(prepare.GetHandle(),
                        const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]),
                        prepare.outputTensors[0]);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "mudnn FusedGemm activation failed, status: ",
                           static_cast<int>(status));
  }

  return Status::OK();
}

template <typename T>
Status Gemm<T>::Prepare(OpKernelContext* ctx, MusaPreparation& prepare, int M, int N, int K) const {
  // 1. Get input tensors A, B and optional C (bias)
  const Tensor* A = ctx->Input<Tensor>(0);
  const Tensor* B = ctx->Input<Tensor>(1);
  const Tensor* C = ctx->Input<Tensor>(2);  // bias, can be nullptr

  if (!A || !B) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid input tensors");
  }

  // 2. Create output tensor
  Tensor* Y = ctx->Output(0, {M, N});
  if (!Y) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor");
  }

  // 3. Store tensor pointers and shapes in preparation for use in ComputeInternal
  prepare.input_a_ptr = A->DataRaw();
  prepare.input_b_ptr = B->DataRaw();
  prepare.output_ptr = Y->MutableDataRaw();
  prepare.output_size = Y->Shape().Size();
  prepare.input_a_shape = A->Shape();
  prepare.input_b_shape = B->Shape();
  prepare.output_shape = Y->Shape();

  if (!prepare.input_a_ptr || !prepare.input_b_ptr || !prepare.output_ptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid tensor data pointers");
  }

  // 4. Get MUSA data type
  const auto musaType = GetMusaDataType<T>();

  // 5. Prepare MUSA tensors - using ORT_TRY/ORT_CATCH like CANN
  ORT_TRY {
    // Set up MUSA stream for asynchronous execution
    auto* stream = Stream(ctx);
    if (prepare.handle) {
      if (stream) {
        auto status = prepare.handle->SetStream(stream);
        if (status != ::musa::dnn::Status::SUCCESS) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                 "Failed to set MUSA stream, status: " +
                                     std::to_string(static_cast<int>(status)));
        }
      } else {
        // Use default stream for backward compatibility
        LOGS_DEFAULT(WARNING) << "No stream provided, using default MUSA stream";
      }
    }

    // Initialize tensors vectors - account for optional bias
    size_t num_inputs = C ? 3 : 2;
    prepare.inputTensors.resize(num_inputs);
    prepare.outputTensors.resize(1);

    // Setup input tensor A
    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.inputTensors[0], A, musaType, &prepare));

    // Setup input tensor B
    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.inputTensors[1], B, musaType, &prepare));

    // Setup optional bias tensor C
    if (C) {
      ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.inputTensors[2], C, musaType, &prepare));
    }

    // Setup output tensor Y
    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.outputTensors[0], Y, musaType, &prepare));
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  return Status::OK();
}

template <typename T>
Status Gemm<T>::ComputeInternal(OpKernelContext* ctx) const {
  const auto* X = ctx->Input<Tensor>(0);
  const auto* W = ctx->Input<Tensor>(1);
  const auto* B = ctx->Input<Tensor>(2);

  /* Bias could be missing. Treat as scalar 0 if that is the case. */
  GemmHelper helper(X->Shape(), trans_A_, W->Shape(), trans_B_, B != nullptr ? B->Shape() : TensorShape({}));
  if (!helper.State().IsOK()) return helper.State();

  int M = gsl::narrow_cast<int>(helper.M());
  int N = gsl::narrow_cast<int>(helper.N());
  int K = gsl::narrow_cast<int>(helper.K());

  auto* Y = ctx->Output(0, {M, N});
  /* Bail out early if the output is going to be empty */
  if (Y->Shape().Size() == 0) return Status::OK();
  if (K == 0) {
    const auto* ep = static_cast<const MusaExecutionProvider*>(
        Info().GetExecutionProvider());
    ORT_RETURN_IF_ERROR(GemmZeroK<T>(Stream(ctx), ep, B, Y, B != nullptr ? beta_ : 0.0f));
    if (activation_.empty()) {
      return Status::OK();
    }

    MusaPreparation activation_prepare(ep);
    if (activation_prepare.handle && Stream(ctx)) {
      auto status = activation_prepare.handle->SetStream(Stream(ctx));
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Failed to set MUSA stream for FusedGemm K=0 activation, status: ",
                               static_cast<int>(status));
      }
    }
    activation_prepare.outputTensors.resize(1);
    ORT_RETURN_IF_ERROR(SetupMusaTensor(activation_prepare.outputTensors[0],
                                        Y, GetMusaDataType<T>(), &activation_prepare));
    return ApplyActivationInPlace(activation_prepare);
  }

  /* Prepare MUSA operation */
  const auto* ep = static_cast<const MusaExecutionProvider*>(
      Info().GetExecutionProvider());
  MusaPreparation prepare(ep);
  ORT_RETURN_IF_ERROR(Prepare(ctx, prepare, M, N, K));

  /* Create workspace allocator (required for FP16 support, following torch_musa pattern) */
  auto* ort_stream = ctx->GetComputeStream();
  std::vector<IAllocatorUniquePtr<void>> workspace_buffers;
  auto memory_allocator = [this, ort_stream, &workspace_buffers](size_t size) -> ::musa::dnn::MemoryHandler {
    if (size == 0) {
      return ::musa::dnn::MemoryHandler(nullptr, [](void*) {});
    }
    auto scratch = this->GetScratchBuffer<void>(size, ort_stream);
    void* ptr = scratch.get();
    workspace_buffers.push_back(std::move(scratch));
    return ::musa::dnn::MemoryHandler(ptr, [](void*) {});
  };

  /* Configure MatMul operation */
  ::musa::dnn::MatMul matmul_op;

  auto status = matmul_op.SetTranspose(trans_A_, trans_B_);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set MatMul transpose");
  }

  status = matmul_op.SetAlpha(static_cast<double>(alpha_));
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set MatMul alpha");
  }

  status = matmul_op.SetComputeMode(::musa::dnn::MatMul::ComputeMode::TENSOR);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set MatMul compute mode");
  }

  float effective_beta = (B != nullptr) ? beta_ : 0.0f;

  /* Execute with bias handling strategy (following torch_musa MmCall pattern):
   * Case 1: Full-shape bias (M,N) → RunWithBiasAdd, bias as c parameter with beta
   * Case 2: 1D bias (N,) → RunWithBiasAdd, bias as bias parameter with gamma
   * Case 3: Scalar bias (1,) → Run + Binary ADD_ALPHA (RunWithBiasAdd doesn't support scalar)
   * Case 4: No bias → basic Run
   */
  if (effective_beta != 0.0f && prepare.inputTensors.size() >= 3) {
    const auto& bias_shape = B->Shape();
    if (bias_shape.NumDimensions() == 2 &&
        bias_shape[0] == M && bias_shape[1] == N) {
      // Case 1: Full-shape bias (M, N)
      // d = alpha * a * b + beta * c + gamma * bias(empty)
      // Pass bias as c, empty tensor as bias
      status = matmul_op.SetBeta(static_cast<double>(effective_beta));
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set MatMul beta");
      }

      ::musa::dnn::Tensor empty_tensor;
      status = matmul_op.RunWithBiasAdd(
          prepare.GetHandle(),
          const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]),  // d (output)
          prepare.inputTensors[0],                                     // a
          prepare.inputTensors[1],                                     // b
          prepare.inputTensors[2],                                     // c = bias (full shape, scaled by beta)
          empty_tensor,                                                // bias (empty, no gamma term)
          memory_allocator);
    } else if (bias_shape.NumDimensions() == 1 && bias_shape[0] > 1) {
      // Case 2: 1D bias (N,) where N > 1
      // d = alpha * a * b + 0 * c + gamma * bias
      status = matmul_op.SetGamma(static_cast<double>(effective_beta));
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set MatMul gamma");
      }

      status = matmul_op.RunWithBiasAdd(
          prepare.GetHandle(),
          const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]),  // d (output)
          prepare.inputTensors[0],                                     // a
          prepare.inputTensors[1],                                     // b
          prepare.outputTensors[0],                                    // c (output, beta=0 so ignored)
          prepare.inputTensors[2],                                     // bias (1D)
          memory_allocator);
    } else {
      // Case 3: Scalar bias (1,) or other shapes
      // Two-step: Y = alpha * A * B, then Y = Y + beta * C using Binary Add
      status = matmul_op.Run(
          prepare.GetHandle(),
          const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]),
          prepare.inputTensors[0],
          prepare.inputTensors[1],
          memory_allocator);
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "mudnn MatMul operation failed in scalar bias path, status: " +
                                   std::to_string(static_cast<int>(status)));
      }

      // Add scaled bias: Y = Y + beta * C using Binary ADD_ALPHA
      ::musa::dnn::Binary binary_op;
      auto bin_status = binary_op.SetMode(::musa::dnn::Binary::Mode::ADD_ALPHA);
      if (bin_status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set Binary mode");
      }
      bin_status = binary_op.SetAlpha(static_cast<double>(effective_beta));
      if (bin_status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set Binary alpha");
      }
      // ADD_ALPHA: out = a + alpha * b
      bin_status = binary_op.Run(
          prepare.GetHandle(),
          const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]),  // out = Y
          prepare.outputTensors[0],                                    // a = Y (matmul result)
          prepare.inputTensors[2]);                                    // b = C (bias, to be scaled by alpha=beta)
      if (bin_status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "mudnn Binary ADD_ALPHA failed for scalar bias, status: " +
                                   std::to_string(static_cast<int>(bin_status)));
      }
      // Already handled, skip the common status check below
      return ApplyActivationInPlace(prepare);
    }
  } else {
    // No bias case
    status = matmul_op.Run(
        prepare.GetHandle(),
        const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]),
        prepare.inputTensors[0],
        prepare.inputTensors[1],
        memory_allocator);
  }

  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "mudnn MatMul operation failed, status: " +
                               std::to_string(static_cast<int>(status)));
  }

  return ApplyActivationInPlace(prepare);
}

// Macro for registering typed kernel
#define REGISTER_MUSA_GEMM_TYPED_KERNEL(ver, T)                   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Gemm, kOnnxDomain, ver, T, kMusaExecutionProvider,          \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Gemm<T>);

#define REGISTER_MUSA_GEMM_VERSIONED_TYPED_KERNEL(startver, endver, T) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                             \
      Gemm, kOnnxDomain, startver, endver, T, kMusaExecutionProvider,  \
      (*KernelDefBuilder::Create())                                    \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),      \
      Gemm<T>);

// Combined macro for both kernel and compute registration
#define REGISTER_MUSA_GEMM_TYPED(ver, T) \
  REGISTER_MUSA_GEMM_TYPED_KERNEL(ver, T)

// Register operations for versions 11-12
REGISTER_MUSA_GEMM_VERSIONED_TYPED_KERNEL(11, 12, float)
REGISTER_MUSA_GEMM_VERSIONED_TYPED_KERNEL(11, 12, MLFloat16)
// Register operations for version 13
REGISTER_MUSA_GEMM_TYPED(13, float)
REGISTER_MUSA_GEMM_TYPED(13, MLFloat16)
// Register operations for version 14
REGISTER_MUSA_GEMM_TYPED(14, float)
REGISTER_MUSA_GEMM_TYPED(14, MLFloat16)

}  // namespace musa
}  // namespace onnxruntime
