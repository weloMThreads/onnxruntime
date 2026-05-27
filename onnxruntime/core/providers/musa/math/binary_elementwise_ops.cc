// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/common.h"
#include "core/providers/musa/math/binary_elementwise_ops.h"
#include "core/providers/musa/math/elementwise_safe_impl.h"
#include "core/providers/musa/musa_call.h"
#include "core/providers/musa/musa_execution_provider.h"  // For PerThreadContext
#include "core/providers/musa/musa_fwd.h"
#include <algorithm>
#include <musa_runtime.h>
#include <mudnn.h>
#include <string>
#include <type_traits>
#include <vector>

using onnxruntime::common::Status;
namespace onnxruntime {
namespace musa {

namespace {

void FillOutputStrides(const std::vector<int64_t>& dims,
                       int64_t* strides) {
  int64_t running_stride = 1;
  for (int64_t dim = static_cast<int64_t>(dims.size()) - 1; dim >= 0; --dim) {
    strides[dim] = running_stride;
    running_stride *= dims[dim];
  }
}

void FillInputStrides(const std::vector<int64_t>& dims,
                      int64_t* strides) {
  int64_t running_stride = 1;
  for (int64_t dim = static_cast<int64_t>(dims.size()) - 1; dim >= 0; --dim) {
    strides[dim] = (dims[dim] == 1) ? 0 : running_stride;
    running_stride *= dims[dim];
  }
}

Status BuildPowSameTypeParams(const TensorShape& lhs_shape,
                              const TensorShape& rhs_shape,
                              const TensorShape& output_shape,
                              PowSameTypeParams* params) {
  ORT_RETURN_IF_NOT(params != nullptr, "PowSameTypeParams must not be null");

  const int32_t rank = static_cast<int32_t>(output_shape.NumDimensions());
  ORT_RETURN_IF_NOT(rank <= kPowSameTypeMaxDims,
                    "Pow safe kernel supports rank up to ", kPowSameTypeMaxDims,
                    ", got ", rank);

  params->rank = rank;
  params->total_elements = output_shape.Size();
  std::fill_n(params->output_strides, kPowSameTypeMaxDims, 0);
  std::fill_n(params->lhs_strides, kPowSameTypeMaxDims, 0);
  std::fill_n(params->rhs_strides, kPowSameTypeMaxDims, 0);

  if (rank == 0 || params->total_elements == 0) {
    return Status::OK();
  }

  std::vector<int64_t> padded_output(rank, 1);
  std::vector<int64_t> padded_lhs(rank, 1);
  std::vector<int64_t> padded_rhs(rank, 1);

  const auto output_dims = output_shape.GetDims();
  std::copy(output_dims.begin(), output_dims.end(), padded_output.begin());

  const auto lhs_dims = lhs_shape.GetDims();
  const auto rhs_dims = rhs_shape.GetDims();
  const size_t lhs_offset = static_cast<size_t>(rank - static_cast<int32_t>(lhs_shape.NumDimensions()));
  const size_t rhs_offset = static_cast<size_t>(rank - static_cast<int32_t>(rhs_shape.NumDimensions()));
  std::copy(lhs_dims.begin(), lhs_dims.end(), padded_lhs.begin() + lhs_offset);
  std::copy(rhs_dims.begin(), rhs_dims.end(), padded_rhs.begin() + rhs_offset);

  FillOutputStrides(padded_output, params->output_strides);
  FillInputStrides(padded_lhs, params->lhs_strides);
  FillInputStrides(padded_rhs, params->rhs_strides);

  return Status::OK();
}

// Check if a pointer resides on host memory (graph initializers may stay on CPU).
// Returns true if the pointer is host memory and needs device copy.
static bool IsHostPointer(const void* ptr) {
  if (!ptr) return false;
  musaPointerAttributes attr;
  auto err = musaPointerGetAttributes(&attr, ptr);
  if (err != musaSuccess) {
    // musaPointerGetAttributes fails for host-allocated memory (not registered
    // with MUSA runtime). Clear the error and treat as host pointer.
    musaGetLastError();  // clear sticky error
    return true;
  }
  return attr.type == musaMemoryTypeHost ||
         attr.type == musaMemoryTypeUnregistered;
}

template <typename T>
Status LaunchPowSameType(const MusaPreparation& prepare,
                         musaStream_t stream) {
  PowSameTypeParams params{};
  ORT_RETURN_IF_ERROR(BuildPowSameTypeParams(prepare.input_a_shape,
                                             prepare.input_b_shape,
                                             prepare.output_shape,
                                             &params));

  // Graph initializers (e.g. constant exponent 2.0 in LayerNorm Pow) may
  // reside in CPU memory. Copy to a temporary GPU buffer before kernel launch.
  const void* gpu_a = prepare.input_a_ptr;
  const void* gpu_b = prepare.input_b_ptr;
  void* temp_a = nullptr;
  void* temp_b = nullptr;

  auto cleanup = [&]() {
    if (temp_a) musaFree(temp_a);
    if (temp_b) musaFree(temp_b);
  };

  if (IsHostPointer(prepare.input_a_ptr)) {
    size_t bytes = prepare.input_a_shape.Size() * sizeof(T);
    MUSA_RETURN_IF_ERROR(musaMalloc(&temp_a, bytes));
    MUSA_RETURN_IF_ERROR(musaMemcpyAsync(temp_a, prepare.input_a_ptr, bytes,
                                         musaMemcpyHostToDevice, stream));
    gpu_a = temp_a;
  }
  if (IsHostPointer(prepare.input_b_ptr)) {
    size_t bytes = prepare.input_b_shape.Size() * sizeof(T);
    MUSA_RETURN_IF_ERROR(musaMalloc(&temp_b, bytes));
    MUSA_RETURN_IF_ERROR(musaMemcpyAsync(temp_b, prepare.input_b_ptr, bytes,
                                         musaMemcpyHostToDevice, stream));
    gpu_b = temp_b;
  }

  if constexpr (std::is_same_v<T, MLFloat16>) {
    LaunchPowSameTypeKernelHalf(stream, gpu_a, gpu_b,
                                prepare.output_ptr, params);
  } else {
    LaunchPowSameTypeKernel<T>(stream,
                               reinterpret_cast<const T*>(gpu_a),
                               reinterpret_cast<const T*>(gpu_b),
                               reinterpret_cast<T*>(prepare.output_ptr),
                               params);
  }

  MUSA_RETURN_IF_ERROR(musaGetLastError());
  // Sync before freeing temp buffers (kernel uses them asynchronously)
  if (temp_a || temp_b) {
    MUSA_RETURN_IF_ERROR(musaStreamSynchronize(stream));
  }
  cleanup();
  return Status::OK();
}

}  // namespace

template <typename T>
Status SimpleMusaBinaryOp(const MusaPreparation& prepare,
                          size_t size,
                          musaStream_t stream,
                          const std::string& op_name);

// Simplified approach: Just fail for mixed types for now and implement step by step
template <typename T1, typename T2>
Status SimpleMusaBinaryOpMixed(OpKernelContext* ctx,
                               const MusaExecutionProvider* ep,
                               const std::string& op_name) {

  // For now, only handle same types
  if constexpr (std::is_same_v<T1, T2>) {
    // Use standard single-type path
    MusaPreparation prepare(ep);
    const auto musaType = GetMusaDataType<T1>();
    
    // Get input tensors A and B
    const Tensor *A = ctx->Input<Tensor>(0);
    const Tensor *B = ctx->Input<Tensor>(1);
    
    if (!A || !B) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input tensors");
    }
    
    // Check if shapes are broadcastable and compute output shape
    TensorShape output_shape;
    ORT_RETURN_IF_ERROR(ComputeBroadcastOutputShape(op_name, A->Shape(), B->Shape(), output_shape));

    // Create output tensor (output type follows first input type T1)
    Tensor *C = ctx->Output(0, output_shape);
    if (!C) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor");
    }
    
    // Store basic tensor info
    prepare.input_a_ptr = A->DataRaw();
    prepare.input_b_ptr = B->DataRaw(); 
    prepare.output_ptr = C->MutableDataRaw();
    prepare.output_size = output_shape.Size();
    prepare.input_a_shape = A->Shape();
    prepare.input_b_shape = B->Shape();
    prepare.output_shape = output_shape;
    
    // Setup tensors  
    musaStream_t stream = nullptr;
    ORT_TRY {
      // Get MUSA stream
      auto* ort_stream = ctx->GetComputeStream();
      if (ort_stream) {
        stream = static_cast<musaStream_t>(ort_stream->GetHandle());
      }
      
      if (prepare.handle && stream) {
        auto status = prepare.handle->SetStream(stream);
        if (status != ::musa::dnn::Status::SUCCESS) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set MUSA stream");
        }
      }
      
      prepare.inputTensors.resize(2);
      prepare.outputTensors.resize(1);
      
      ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.inputTensors[0], A, musaType, &prepare));
      ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.inputTensors[1], B, musaType, &prepare));
      ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.outputTensors[0], C, musaType, &prepare));
    }
    ORT_CATCH(const std::exception &e) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
    }
    
    // Call MUSA operation
    return SimpleMusaBinaryOp<T1>(prepare, prepare.output_size, stream, op_name);
  } else {
           
    // For integer base, MUSA doesn't support, so return error immediately
    if constexpr (std::is_integral_v<T1>) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, 
                             "MUSA Binary POW does not support integer base type");
    }
    
    // For float base + integer exponent: implement simple CPU-GPU hybrid approach
    if constexpr (std::is_floating_point_v<T1> && std::is_integral_v<T2>) {
      // For mixed type operations, we need to handle the type conversion properly
      // MUSA Binary POW with mixed types is currently problematic, needs further investigation

      MusaPreparation prepare(ep);
      const auto musaType = GetMusaDataType<T1>();
      
      // Get input tensors A and B
      const Tensor *A = ctx->Input<Tensor>(0);
      const Tensor *B = ctx->Input<Tensor>(1);
      
      if (!A || !B) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input tensors");
      }
      
      // Check if shapes are broadcastable and compute output shape
      TensorShape output_shape;
      ORT_RETURN_IF_ERROR(ComputeBroadcastOutputShape("Pow", A->Shape(), B->Shape(), output_shape));

      // Create output tensor (output type follows first input type T1)
      Tensor *C = ctx->Output(0, output_shape);
      if (!C) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor");
      }
      
      // Store basic tensor info
      prepare.input_a_ptr = A->DataRaw();
      prepare.input_b_ptr = B->DataRaw(); // Use original integer data for now
      prepare.output_ptr = C->MutableDataRaw();
      prepare.output_size = output_shape.Size();
      
      // Store shape information 
      prepare.input_a_shape = A->Shape();
      prepare.input_b_shape = B->Shape();
      prepare.output_shape = output_shape;
      
      // Setup input tensors in the vector
      prepare.inputTensors.resize(2);
      prepare.outputTensors.resize(1);
      
      // Setup first input tensor (A)
      ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.inputTensors[0], A, musaType, &prepare));
      
      // Setup second input tensor (B) with its original integer type
      ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.inputTensors[1], B, GetMusaDataType<T2>(), &prepare));
      
      // Setup output tensor (C)
      ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.outputTensors[0], C, musaType, &prepare));
      
      // For now, fallback to CPU for mixed types - ensures correctness first
      // TODO: Implement MUSA device memory allocation and type conversion kernel for better performance
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, 
                             "Mixed type Pow operations temporarily fall back to CPU for correctness");
    }
    
    // For other mixed type cases, fallback to CPU for now
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, 
                           "Mixed type Pow operations temporarily fall back to CPU");
  }
}

// Pow operation dual dispatch implementation
namespace pow_internal {

template <typename T1>
Status DispatchOnSecondArg(OpKernelContext* ctx,
                           const MusaExecutionProvider* ep,
                           const std::string& op_name) {
  namespace on = ONNX_NAMESPACE;

  const Tensor* rhs_tensor = ctx->Input<Tensor>(1);
  
  if (!rhs_tensor) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Second input tensor is null");
  }
  
  Status result;
  switch (rhs_tensor->GetElementType()) {
    case on::TensorProto_DataType_INT32:
      result = SimpleMusaBinaryOpMixed<T1, int32_t>(ctx, ep, op_name);
      break;
    case on::TensorProto_DataType_INT64:
      result = SimpleMusaBinaryOpMixed<T1, int64_t>(ctx, ep, op_name);
      break;
    case on::TensorProto_DataType_FLOAT:
      result = SimpleMusaBinaryOpMixed<T1, float>(ctx, ep, op_name);
      break;
    case on::TensorProto_DataType_FLOAT16:
      result = SimpleMusaBinaryOpMixed<T1, MLFloat16>(ctx, ep, op_name);
      break;
    case on::TensorProto_DataType_DOUBLE:
      result = SimpleMusaBinaryOpMixed<T1, double>(ctx, ep, op_name);
      break;
    default:
      result = ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, 
                              "Unsupported second input type for Pow operation: " +
                              std::to_string(static_cast<int>(rhs_tensor->GetElementType())));
  }
  return result;
}

template <typename T1>
Status DispatchOnFirstArg(OpKernelContext* ctx,
                          const MusaExecutionProvider* ep,
                          const std::string& op_name) {
  return DispatchOnSecondArg<T1>(ctx, ep, op_name);
}

} // namespace pow_internal

// Helper function to map operation names to MUSA Binary modes
static ::musa::dnn::Binary::Mode GetBinaryMode(const std::string &op_name) {
  if (op_name == "Add") {
    return ::musa::dnn::Binary::Mode::ADD;
  } else if (op_name == "Sub") {
    return ::musa::dnn::Binary::Mode::SUB;
  } else if (op_name == "Mul") {
    return ::musa::dnn::Binary::Mode::MUL;
  } else if (op_name == "Div") {
    return ::musa::dnn::Binary::Mode::DIV;
  } else if (op_name == "Pow") {
    return ::musa::dnn::Binary::Mode::POW;
  } else if (op_name == "Min") {
    return ::musa::dnn::Binary::Mode::MIN;
  } else if (op_name == "Max") {
    return ::musa::dnn::Binary::Mode::MAX;
  } else if (op_name == "PRelu") {
    return ::musa::dnn::Binary::Mode::PRELU;
  } else {
    // This should not happen as we validate in the calling function
    return ::musa::dnn::Binary::Mode::ADD;
  }
}

// MUSA device-based implementation using MusaPreparation and mudnn library
template <typename T>
Status SimpleMusaBinaryOp(const MusaPreparation &prepare, size_t size,
                          musaStream_t stream,
                          const std::string &op_name) {
  // Support Add, Sub, Mul, Div, Pow, Min, Max, PRelu operations
  if (op_name != "Add" && op_name != "Sub" && op_name != "Mul" && op_name != "Div" &&
      op_name != "Pow" && op_name != "Min" && op_name != "Max" && op_name != "PRelu") {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Unsupported operation: " + op_name);
  }

  if (size == 0) {
    return Status::OK();
  }


  // Get tensor data from prepared MUSA tensors
  const T *input_a = reinterpret_cast<const T *>(prepare.input_a_ptr);
  const T *input_b = reinterpret_cast<const T *>(prepare.input_b_ptr);
  T *output = reinterpret_cast<T *>(prepare.output_ptr);

  // Validate prepared tensors
  if (!input_a || !input_b || !output) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid prepared tensor pointers");
  }

  if (prepare.inputTensors.size() < 2 || prepare.outputTensors.size() < 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid prepared tensor configuration");
  }

  if (op_name == "Pow") {
    return LaunchPowSameType<T>(prepare, stream);
  }

  // Use mudnn Binary class for device computation with broadcasting support
  try {
    // Create mudnn Binary operation
    ::musa::dnn::Binary binary_op;

    // Set the operation mode based on op_name
    auto mode = GetBinaryMode(op_name);
    auto status = binary_op.SetMode(mode);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set mudnn Binary mode to " + op_name);
    }

    // Run the binary operation directly on device with automatic broadcasting
    // Use GetHandle() for unified access (works for both legacy and EP modes)
    status = binary_op.Run(prepare.GetHandle(),
                          const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]),  // output tensor
                          prepare.inputTensors[0],   // input A tensor
                          prepare.inputTensors[1]);  // input B tensor


    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "mudnn Binary " + op_name + " operation failed, status: " +
                             std::to_string(static_cast<int>(status)));
    }


  } catch (const std::exception &e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Exception in mudnn Binary operation: " +
                           std::string(e.what()));
  }

  return Status::OK();
}

template <typename T>
Status BinaryElementwise::Prepare(OpKernelContext *ctx,
                                  MusaPreparation &prepare) const {
  // 1. Get input tensors A and B
  const Tensor *A = ctx->Input<Tensor>(0);
  const Tensor *B = ctx->Input<Tensor>(1);

  if (!A || !B) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid input tensors");
  }

  // 2. Check if shapes are broadcastable and compute output shape
  TensorShape output_shape;
  ORT_RETURN_IF_ERROR(ComputeBroadcastOutputShape(Node().Name(), A->Shape(),
                                                  B->Shape(), output_shape));

  // 3. Create output tensor
  Tensor *C = ctx->Output(0, output_shape);
  if (!C) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor");
  }

  // 4. Store tensor pointers and shapes in preparation for use in
  // ComputeInternal
  prepare.input_a_ptr = A->DataRaw();
  prepare.input_b_ptr = B->DataRaw();
  prepare.output_ptr = C->MutableDataRaw();
  prepare.output_size = output_shape.Size();
  prepare.input_a_shape = A->Shape();
  prepare.input_b_shape = B->Shape();
  prepare.output_shape = output_shape;

  if (prepare.output_size > 0 && (!prepare.input_a_ptr || !prepare.input_b_ptr || !prepare.output_ptr)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid tensor data pointers");
  }

  // 6. Get MUSA data type
  const auto musaType = GetMusaDataType<T>();

  // 7. Prepare MUSA tensors - using ORT_TRY/ORT_CATCH like CANN
  ORT_TRY {
    // Set up MUSA stream for asynchronous execution
    // In EP mode (prepare.handle is nullptr), stream is already set in PerThreadContext constructor
    // In legacy mode (prepare.handle is not nullptr), we need to set stream here
    if (prepare.handle) {
      // Legacy mode - need to set stream
      auto* stream = Stream(ctx);
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
    // EP mode - stream already set in PerThreadContext, skip SetStream to avoid race condition

    // Initialize tensors vectors
    prepare.inputTensors.resize(2);
    prepare.outputTensors.resize(1);

    // Setup input tensor A
    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.inputTensors[0], A, musaType, &prepare));

    // Setup input tensor B
    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.inputTensors[1], B, musaType, &prepare));

    // Setup output tensor C
    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.outputTensors[0], C, musaType, &prepare));
  }
  ORT_CATCH(const std::exception &e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  return Status::OK();
}

// Forward declaration for Pow's dual dispatch
namespace pow_internal {
template <typename T1>
Status DispatchOnFirstArg(OpKernelContext* ctx,
                          const MusaExecutionProvider* ep,
                          const std::string& op_name);
}

// Macro for registering typed compute function with MUSA implementation
#define REGISTER_MUSA_ELEMENTWISE_TYPED_COMPUTE(x, T)                          \
  template <> Status x<T>::ComputeInternal(OpKernelContext *ctx) const {       \
    /* Check input count - only support binary operations for now */           \
    const int input_count = ctx->InputCount();                                 \
    if (input_count > 2) {                                                     \
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,                               \
        "MusaEP " #x " only supports binary operations (2 inputs). "           \
        "Found " + std::to_string(input_count) + " inputs. "                   \
        "Multi-input " #x " operations will fallback to CPU.");                \
    }                                                                           \
                                                                               \
    /* Get EP for PerThreadContext-based Handle */                             \
    const auto* ep = static_cast<const MusaExecutionProvider*>(                \
        Info().GetExecutionProvider());                                        \
                                                                               \
    /* Prepare MUSA operation with EP (uses PerThreadContext) */               \
    MusaPreparation prepare(ep);                                               \
    ORT_RETURN_IF_ERROR(Prepare<T>(ctx, prepare));                             \
                                                                               \
    /* Call MUSA device binary operation using prepared data */                \
    ORT_RETURN_IF_ERROR(                                                       \
        SimpleMusaBinaryOp<T>(prepare, prepare.output_size, Stream(ctx), #x)); \
                                                                               \
    return Status::OK();                                                       \
  }

// Special compute implementation for Pow with dual type dispatch
#define REGISTER_MUSA_POW_TYPED_COMPUTE(T)                                     \
  template <> Status Pow<T>::ComputeInternal(OpKernelContext *ctx) const {     \
    /* Get EP for thread-safe Handle management */                            \
    const auto* ep = static_cast<const MusaExecutionProvider*>(               \
        Info().GetExecutionProvider());                                        \
    /* Use dual dispatch for Pow operation */                                 \
    return pow_internal::DispatchOnFirstArg<T>(ctx, ep, "Pow");                \
  }

// Macro for registering typed kernel
#define REGISTER_MUSA_ELEMENTWISE_TYPED_KERNEL_IN_DOMAIN(x, domain, ver, T)    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                               \
      x, domain, ver, T, kMusaExecutionProvider,                               \
      (*KernelDefBuilder::Create())                                            \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),              \
      x<T>);

#define REGISTER_MUSA_ELEMENTWISE_TYPED_KERNEL(x, ver, T)                      \
  REGISTER_MUSA_ELEMENTWISE_TYPED_KERNEL_IN_DOMAIN(x, kOnnxDomain, ver, T)

// Macro for registering versioned typed kernel
#define REGISTER_MUSA_ELEMENTWISE_VERSIONED_TYPED_KERNEL_IN_DOMAIN(x, domain, startver, endver, T) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                     \
      x, domain, startver, endver, T, kMusaExecutionProvider,                  \
      (*KernelDefBuilder::Create())                                            \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),              \
      x<T>);

#define REGISTER_MUSA_ELEMENTWISE_VERSIONED_TYPED_KERNEL(x, startver, endver, T) \
  REGISTER_MUSA_ELEMENTWISE_VERSIONED_TYPED_KERNEL_IN_DOMAIN(x, kOnnxDomain, startver, endver, T)

// Macro for versioned typed registration (kernel only, no compute)
#define REGISTER_MUSA_ELEMENTWISE_VERSIONED_TYPED(name, startver, endver, T)   \
  REGISTER_MUSA_ELEMENTWISE_VERSIONED_TYPED_KERNEL(name, startver, endver, T)

// Combined macro for both kernel and compute registration
#define REGISTER_MUSA_ELEMENTWISE_TYPED(name, ver, T)                          \
  REGISTER_MUSA_ELEMENTWISE_TYPED_KERNEL(name, ver, T)                         \
  REGISTER_MUSA_ELEMENTWISE_TYPED_COMPUTE(name, T)

// Register operations
#define REGISTER_MUSA_ELEMENTWISE_ILHFD(name, ver)                   \
  REGISTER_MUSA_ELEMENTWISE_TYPED(name, ver, int32_t)                          \
  REGISTER_MUSA_ELEMENTWISE_TYPED(name, ver, int64_t)                          \
  REGISTER_MUSA_ELEMENTWISE_TYPED(name, ver, MLFloat16)                        \
  REGISTER_MUSA_ELEMENTWISE_TYPED(name, ver, float)

// Register versioned operations (kernel only, for ILHFD types)
#define REGISTER_MUSA_ELEMENTWISE_VERSIONED_ILHFD(name, startver, endver)       \
  REGISTER_MUSA_ELEMENTWISE_VERSIONED_TYPED(name, startver, endver, int32_t)    \
  REGISTER_MUSA_ELEMENTWISE_VERSIONED_TYPED(name, startver, endver, int64_t)    \
  REGISTER_MUSA_ELEMENTWISE_VERSIONED_TYPED(name, startver, endver, MLFloat16)  \
  REGISTER_MUSA_ELEMENTWISE_VERSIONED_TYPED(name, startver, endver, float)

// Register Add operations
REGISTER_MUSA_ELEMENTWISE_VERSIONED_ILHFD(Add, 7, 13)
REGISTER_MUSA_ELEMENTWISE_ILHFD(Add, 14)

// Register Sub operations
REGISTER_MUSA_ELEMENTWISE_VERSIONED_ILHFD(Sub, 7, 13)
REGISTER_MUSA_ELEMENTWISE_ILHFD(Sub, 14)

// Register Mul operations
REGISTER_MUSA_ELEMENTWISE_VERSIONED_ILHFD(Mul, 7, 13)
REGISTER_MUSA_ELEMENTWISE_ILHFD(Mul, 14)

#ifdef ENABLE_MUSA_NHWC_OPS
REGISTER_MUSA_ELEMENTWISE_VERSIONED_TYPED_KERNEL_IN_DOMAIN(Add, kMSInternalNHWCDomain, 7, 13, int32_t)
REGISTER_MUSA_ELEMENTWISE_VERSIONED_TYPED_KERNEL_IN_DOMAIN(Add, kMSInternalNHWCDomain, 7, 13, int64_t)
REGISTER_MUSA_ELEMENTWISE_VERSIONED_TYPED_KERNEL_IN_DOMAIN(Add, kMSInternalNHWCDomain, 7, 13, MLFloat16)
REGISTER_MUSA_ELEMENTWISE_VERSIONED_TYPED_KERNEL_IN_DOMAIN(Add, kMSInternalNHWCDomain, 7, 13, float)
REGISTER_MUSA_ELEMENTWISE_TYPED_KERNEL_IN_DOMAIN(Add, kMSInternalNHWCDomain, 14, int32_t)
REGISTER_MUSA_ELEMENTWISE_TYPED_KERNEL_IN_DOMAIN(Add, kMSInternalNHWCDomain, 14, int64_t)
REGISTER_MUSA_ELEMENTWISE_TYPED_KERNEL_IN_DOMAIN(Add, kMSInternalNHWCDomain, 14, MLFloat16)
REGISTER_MUSA_ELEMENTWISE_TYPED_KERNEL_IN_DOMAIN(Add, kMSInternalNHWCDomain, 14, float)

REGISTER_MUSA_ELEMENTWISE_VERSIONED_TYPED_KERNEL_IN_DOMAIN(Mul, kMSInternalNHWCDomain, 7, 13, int32_t)
REGISTER_MUSA_ELEMENTWISE_VERSIONED_TYPED_KERNEL_IN_DOMAIN(Mul, kMSInternalNHWCDomain, 7, 13, int64_t)
REGISTER_MUSA_ELEMENTWISE_VERSIONED_TYPED_KERNEL_IN_DOMAIN(Mul, kMSInternalNHWCDomain, 7, 13, MLFloat16)
REGISTER_MUSA_ELEMENTWISE_VERSIONED_TYPED_KERNEL_IN_DOMAIN(Mul, kMSInternalNHWCDomain, 7, 13, float)
REGISTER_MUSA_ELEMENTWISE_TYPED_KERNEL_IN_DOMAIN(Mul, kMSInternalNHWCDomain, 14, int32_t)
REGISTER_MUSA_ELEMENTWISE_TYPED_KERNEL_IN_DOMAIN(Mul, kMSInternalNHWCDomain, 14, int64_t)
REGISTER_MUSA_ELEMENTWISE_TYPED_KERNEL_IN_DOMAIN(Mul, kMSInternalNHWCDomain, 14, MLFloat16)
REGISTER_MUSA_ELEMENTWISE_TYPED_KERNEL_IN_DOMAIN(Mul, kMSInternalNHWCDomain, 14, float)
#endif

// Register Div operations
REGISTER_MUSA_ELEMENTWISE_VERSIONED_ILHFD(Div, 7, 13)
REGISTER_MUSA_ELEMENTWISE_ILHFD(Div, 14)

// Register Pow operations (version 7 to 11 are versioned with same type constraint, 12+ support multi-type)
// Note: We only need one ComputeInternal per type, so use versioned registration for kernels only
REGISTER_MUSA_ELEMENTWISE_VERSIONED_ILHFD(Pow, 7, 11)

// For v12+, use the typed kernel registration (kernel only, compute is shared)
#define REGISTER_MUSA_POW_TYPED_KERNEL_ONLY(ver, T) \
  REGISTER_MUSA_ELEMENTWISE_TYPED_KERNEL(Pow, ver, T)

REGISTER_MUSA_POW_TYPED_KERNEL_ONLY(12, int32_t)
REGISTER_MUSA_POW_TYPED_KERNEL_ONLY(12, int64_t)
REGISTER_MUSA_POW_TYPED_KERNEL_ONLY(12, MLFloat16)
REGISTER_MUSA_POW_TYPED_KERNEL_ONLY(12, float)

REGISTER_MUSA_POW_TYPED_KERNEL_ONLY(13, int32_t)
REGISTER_MUSA_POW_TYPED_KERNEL_ONLY(13, int64_t)
REGISTER_MUSA_POW_TYPED_KERNEL_ONLY(13, MLFloat16)
REGISTER_MUSA_POW_TYPED_KERNEL_ONLY(13, float)

REGISTER_MUSA_POW_TYPED_KERNEL_ONLY(14, int32_t)
REGISTER_MUSA_POW_TYPED_KERNEL_ONLY(14, int64_t)
REGISTER_MUSA_POW_TYPED_KERNEL_ONLY(14, MLFloat16)
REGISTER_MUSA_POW_TYPED_KERNEL_ONLY(14, float)

REGISTER_MUSA_POW_TYPED_KERNEL_ONLY(15, int32_t)
REGISTER_MUSA_POW_TYPED_KERNEL_ONLY(15, int64_t)
REGISTER_MUSA_POW_TYPED_KERNEL_ONLY(15, MLFloat16)
REGISTER_MUSA_POW_TYPED_KERNEL_ONLY(15, float)

// Register ComputeInternal implementations (only once per type)
REGISTER_MUSA_POW_TYPED_COMPUTE(int32_t)
REGISTER_MUSA_POW_TYPED_COMPUTE(int64_t)
REGISTER_MUSA_POW_TYPED_COMPUTE(MLFloat16)
REGISTER_MUSA_POW_TYPED_COMPUTE(float)

// Register Min operations
// Based on ONNX Min specification: supports versions 6-11, 12, 13
REGISTER_MUSA_ELEMENTWISE_VERSIONED_ILHFD(Min, 6, 11)

REGISTER_MUSA_ELEMENTWISE_TYPED_KERNEL(Min, 12, int32_t)
REGISTER_MUSA_ELEMENTWISE_TYPED_KERNEL(Min, 12, int64_t)
REGISTER_MUSA_ELEMENTWISE_TYPED_KERNEL(Min, 12, MLFloat16)
REGISTER_MUSA_ELEMENTWISE_TYPED_KERNEL(Min, 12, float)

REGISTER_MUSA_ELEMENTWISE_TYPED_KERNEL(Min, 13, int32_t)
REGISTER_MUSA_ELEMENTWISE_TYPED_KERNEL(Min, 13, int64_t)
REGISTER_MUSA_ELEMENTWISE_TYPED_KERNEL(Min, 13, MLFloat16)
REGISTER_MUSA_ELEMENTWISE_TYPED_KERNEL(Min, 13, float)

REGISTER_MUSA_ELEMENTWISE_TYPED_COMPUTE(Min, int32_t)
REGISTER_MUSA_ELEMENTWISE_TYPED_COMPUTE(Min, int64_t)
REGISTER_MUSA_ELEMENTWISE_TYPED_COMPUTE(Min, MLFloat16)
REGISTER_MUSA_ELEMENTWISE_TYPED_COMPUTE(Min, float)

// Register Max operations
// Based on ONNX Max specification: supports versions 6-11, 12, 13
REGISTER_MUSA_ELEMENTWISE_VERSIONED_ILHFD(Max, 6, 11)

// For v12+, use the typed kernel registration (kernel only, compute is shared)
#define REGISTER_MUSA_MAX_TYPED_KERNEL_ONLY(ver, T) \
  REGISTER_MUSA_ELEMENTWISE_TYPED_KERNEL(Max, ver, T)

REGISTER_MUSA_MAX_TYPED_KERNEL_ONLY(12, int32_t)
REGISTER_MUSA_MAX_TYPED_KERNEL_ONLY(12, int64_t)
REGISTER_MUSA_MAX_TYPED_KERNEL_ONLY(12, MLFloat16)
REGISTER_MUSA_MAX_TYPED_KERNEL_ONLY(12, float)

REGISTER_MUSA_MAX_TYPED_KERNEL_ONLY(13, int32_t)
REGISTER_MUSA_MAX_TYPED_KERNEL_ONLY(13, int64_t)
REGISTER_MUSA_MAX_TYPED_KERNEL_ONLY(13, MLFloat16)
REGISTER_MUSA_MAX_TYPED_KERNEL_ONLY(13, float)

// Register ComputeInternal implementations (only once per type)
REGISTER_MUSA_ELEMENTWISE_TYPED_COMPUTE(Max, int32_t)
REGISTER_MUSA_ELEMENTWISE_TYPED_COMPUTE(Max, int64_t)
REGISTER_MUSA_ELEMENTWISE_TYPED_COMPUTE(Max, MLFloat16)
REGISTER_MUSA_ELEMENTWISE_TYPED_COMPUTE(Max, float)

// Register PRelu operations
ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(
    PRelu, kOnnxDomain, 7, 8, float, kMusaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    PRelu<float>);

ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(
    PRelu, kOnnxDomain, 7, 8, MLFloat16, kMusaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    PRelu<MLFloat16>);

ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(
    PRelu, kOnnxDomain, 9, 15, float, kMusaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    PRelu<float>);

ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(
    PRelu, kOnnxDomain, 9, 15, MLFloat16, kMusaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    PRelu<MLFloat16>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    PRelu, kOnnxDomain, 16, float, kMusaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    PRelu<float>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    PRelu, kOnnxDomain, 16, MLFloat16, kMusaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    PRelu<MLFloat16>);

template <> Status PRelu<float>::ComputeInternal(OpKernelContext *ctx) const {
  const int input_count = ctx->InputCount();
  if (input_count > 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "MusaEP PRelu only supports binary operations (2 inputs). "
                           "Found " +
                               std::to_string(input_count) +
                               " inputs. Multi-input PRelu operations will fallback to CPU.");
  }

  const auto *ep =
      static_cast<const MusaExecutionProvider *>(Info().GetExecutionProvider());
  MusaPreparation prepare(ep);
  ORT_RETURN_IF_ERROR(Prepare<float>(ctx, prepare));
  ORT_RETURN_IF_ERROR(SimpleMusaBinaryOp<float>(prepare, prepare.output_size, Stream(ctx), "PRelu"));
  return Status::OK();
}

template <> Status PRelu<MLFloat16>::ComputeInternal(OpKernelContext *ctx) const {
  const int input_count = ctx->InputCount();
  if (input_count > 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "MusaEP PRelu only supports binary operations (2 inputs). "
                           "Found " +
                               std::to_string(input_count) +
                               " inputs. Multi-input PRelu operations will fallback to CPU.");
  }

  const auto *ep =
      static_cast<const MusaExecutionProvider *>(Info().GetExecutionProvider());
  MusaPreparation prepare(ep);
  ORT_RETURN_IF_ERROR(Prepare<MLFloat16>(ctx, prepare));
  ORT_RETURN_IF_ERROR(
      SimpleMusaBinaryOp<MLFloat16>(prepare, prepare.output_size, Stream(ctx), "PRelu"));
  return Status::OK();
}

// Note: MusaEP does not support double and bfloat16 types as per requirements
// Unsigned integer types will use CPU fallback if not supported by MUSA Binary
// These types will automatically fallback to CPU execution when needed

} // namespace musa
} // namespace onnxruntime
