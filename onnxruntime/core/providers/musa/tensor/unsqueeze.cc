// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/tensor/unsqueeze.h"
#include "core/providers/common.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/musa_execution_provider.h"
#include <algorithm>
#include <musa_runtime.h>
#include <mudnn.h>
#include <string>
#include <vector>

using onnxruntime::common::Status;
namespace onnxruntime {
namespace musa {

// MUSA device-based implementation using MusaPreparation
template <typename T>
Status SimpleMusaUnsqueezeOp(const MusaPreparation& prepare, musaStream_t stream) {
  // Get tensor data from prepared MUSA tensors
  const T* input_data = reinterpret_cast<const T*>(prepare.input_a_ptr);
  T* output_data = reinterpret_cast<T*>(prepare.output_ptr);

  // Validate prepared tensors
  if (!input_data || !output_data) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid prepared tensor pointers");
  }

  if (prepare.inputTensors.empty() || prepare.outputTensors.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid prepared tensor configuration");
  }

  // For Unsqueeze, if input and output point to the same memory location,
  // no copy is needed as it's just a view operation
  if (input_data == output_data) {
    return Status::OK();
  }

  // Otherwise, perform async memcpy to copy data
  auto element_size = sizeof(T);
  auto total_bytes = prepare.output_size * element_size;

  try {
    // Use MUSA async memcpy for device-to-device copy
    musaError_t musa_status = musaMemcpyAsync(
        output_data, input_data, total_bytes,
        musaMemcpyDeviceToDevice, stream);

    if (musa_status != musaSuccess) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "MUSA memcpy failed in Unsqueeze operation, status: " +
                                 std::to_string(static_cast<int>(musa_status)));
    }

  } catch (const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Exception in MUSA Unsqueeze operation: " +
                               std::string(e.what()));
  }

  return Status::OK();
}

template <typename T>
Status Unsqueeze<T>::Prepare(OpKernelContext* ctx, MusaPreparation& prepare) const {
  // 1. Use base class PrepareCompute to get input/output tensors with correct shapes
  UnsqueezeBase::Prepare base_prepare;
  ORT_RETURN_IF_ERROR(PrepareCompute(ctx, base_prepare));

  const Tensor* input_tensor = base_prepare.input_tensor;
  Tensor* output_tensor = base_prepare.output_tensor;

  if (input_tensor == nullptr || output_tensor == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid input or output tensor from base PrepareCompute");
  }

  // 2. Store tensor pointers and shapes in preparation
  prepare.input_a_ptr = input_tensor->DataRaw();
  prepare.output_ptr = output_tensor->MutableDataRaw();
  prepare.output_size = output_tensor->Shape().Size();
  prepare.input_a_shape = input_tensor->Shape();
  prepare.output_shape = output_tensor->Shape();

  if (prepare.output_size == 0) {
    return Status::OK();
  }

  if (prepare.input_a_ptr == nullptr || prepare.output_ptr == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid tensor data pointers");
  }

  // 3. Get MUSA data type
  const auto musaType = GetMusaDataType<T>();

  // 4. Prepare MUSA tensors - using ORT_TRY/ORT_CATCH like other MusaEP operations
  ORT_TRY {
    // Set up MUSA stream for asynchronous execution
    auto* stream = Stream(ctx);
    if (prepare.handle) {
      if (stream) {
        auto musa_status = prepare.handle->SetStream(stream);
        if (musa_status != ::musa::dnn::Status::SUCCESS) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                 "Failed to set MUSA stream, status: " +
                                     std::to_string(static_cast<int>(musa_status)));
        }
      } else {
        // Use default stream for backward compatibility
        LOGS_DEFAULT(WARNING) << "No stream provided, using default MUSA stream";
      }
    }

    // Initialize tensors vectors
    prepare.inputTensors.resize(1);
    prepare.outputTensors.resize(1);

    // Setup input tensor
    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.inputTensors[0], input_tensor, musaType, &prepare));

    // Setup output tensor
    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.outputTensors[0], output_tensor, musaType, &prepare));
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  return Status::OK();
}

template <typename T>
Status Unsqueeze<T>::ComputeInternal(OpKernelContext* ctx) const {
  // Prepare MUSA operation
  const auto* ep = static_cast<const MusaExecutionProvider*>(
      Info().GetExecutionProvider());
  MusaPreparation prepare(ep);
  ORT_RETURN_IF_ERROR(Prepare(ctx, prepare));

  if (prepare.output_size == 0) {
    return Status::OK();
  }

  // Call MUSA device unsqueeze operation using prepared data
  musaStream_t stream = Stream(ctx);
  ORT_RETURN_IF_ERROR(SimpleMusaUnsqueezeOp<T>(prepare, stream));

  return Status::OK();
}

// Macro for registering typed compute function with MUSA implementation
#define REGISTER_MUSA_UNSQUEEZE_TYPED_COMPUTE(T) \
  template Status Unsqueeze<T>::ComputeInternal(OpKernelContext* ctx) const;

// Macro for registering versioned typed kernel
#define REGISTER_MUSA_UNSQUEEZE_VERSIONED_TYPED_KERNEL(startver, endver, T) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                  \
      Unsqueeze, kOnnxDomain, startver, endver, T, kMusaExecutionProvider,  \
      (*KernelDefBuilder::Create())                                         \
          .Alias(0, 0)                                                      \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),           \
      Unsqueeze<T>);

// Macro for registering versioned typed kernel with input memory type
#define REGISTER_MUSA_UNSQUEEZE_VERSIONED_TYPED_KERNEL_WITH_INPUT(startver, endver, T) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                             \
      Unsqueeze, kOnnxDomain, startver, endver, T, kMusaExecutionProvider,             \
      (*KernelDefBuilder::Create())                                                    \
          .Alias(0, 0)                                                                 \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                       \
          .InputMemoryType(OrtMemTypeCPUInput, 1),                                     \
      Unsqueeze<T>);

// Macro for registering non-versioned typed kernel with input memory type (opset 13+)
#define REGISTER_MUSA_UNSQUEEZE_TYPED_KERNEL_WITH_INPUT(ver, T)  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                 \
      Unsqueeze, kOnnxDomain, ver, T, kMusaExecutionProvider,    \
      (*KernelDefBuilder::Create())                              \
          .Alias(0, 0)                                           \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()) \
          .InputMemoryType(OrtMemTypeCPUInput, 1),               \
      Unsqueeze<T>);

// Register Unsqueeze operations for different versions and compute functions
// Version 1-10: axes as attribute
REGISTER_MUSA_UNSQUEEZE_VERSIONED_TYPED_KERNEL(1, 10, uint8_t)
REGISTER_MUSA_UNSQUEEZE_VERSIONED_TYPED_KERNEL(1, 10, uint16_t)
REGISTER_MUSA_UNSQUEEZE_VERSIONED_TYPED_KERNEL(1, 10, uint32_t)
REGISTER_MUSA_UNSQUEEZE_VERSIONED_TYPED_KERNEL(1, 10, uint64_t)
REGISTER_MUSA_UNSQUEEZE_VERSIONED_TYPED_KERNEL(1, 10, int8_t)
REGISTER_MUSA_UNSQUEEZE_VERSIONED_TYPED_KERNEL(1, 10, int16_t)
REGISTER_MUSA_UNSQUEEZE_VERSIONED_TYPED_KERNEL(1, 10, int32_t)
REGISTER_MUSA_UNSQUEEZE_VERSIONED_TYPED_KERNEL(1, 10, int64_t)
REGISTER_MUSA_UNSQUEEZE_VERSIONED_TYPED_KERNEL(1, 10, MLFloat16)
REGISTER_MUSA_UNSQUEEZE_VERSIONED_TYPED_KERNEL(1, 10, float)
REGISTER_MUSA_UNSQUEEZE_VERSIONED_TYPED_KERNEL(1, 10, bool)

// Version 11-12: axes as attribute, explicitly support negative axis
REGISTER_MUSA_UNSQUEEZE_VERSIONED_TYPED_KERNEL(11, 12, uint8_t)
REGISTER_MUSA_UNSQUEEZE_VERSIONED_TYPED_KERNEL(11, 12, uint16_t)
REGISTER_MUSA_UNSQUEEZE_VERSIONED_TYPED_KERNEL(11, 12, uint32_t)
REGISTER_MUSA_UNSQUEEZE_VERSIONED_TYPED_KERNEL(11, 12, uint64_t)
REGISTER_MUSA_UNSQUEEZE_VERSIONED_TYPED_KERNEL(11, 12, int8_t)
REGISTER_MUSA_UNSQUEEZE_VERSIONED_TYPED_KERNEL(11, 12, int16_t)
REGISTER_MUSA_UNSQUEEZE_VERSIONED_TYPED_KERNEL(11, 12, int32_t)
REGISTER_MUSA_UNSQUEEZE_VERSIONED_TYPED_KERNEL(11, 12, int64_t)
REGISTER_MUSA_UNSQUEEZE_VERSIONED_TYPED_KERNEL(11, 12, MLFloat16)
REGISTER_MUSA_UNSQUEEZE_VERSIONED_TYPED_KERNEL(11, 12, float)
REGISTER_MUSA_UNSQUEEZE_VERSIONED_TYPED_KERNEL(11, 12, bool)

// Version 13+: axes as input instead of attribute, axes on CPU
REGISTER_MUSA_UNSQUEEZE_TYPED_KERNEL_WITH_INPUT(13, uint8_t)
REGISTER_MUSA_UNSQUEEZE_TYPED_KERNEL_WITH_INPUT(13, uint16_t)
REGISTER_MUSA_UNSQUEEZE_TYPED_KERNEL_WITH_INPUT(13, uint32_t)
REGISTER_MUSA_UNSQUEEZE_TYPED_KERNEL_WITH_INPUT(13, uint64_t)
REGISTER_MUSA_UNSQUEEZE_TYPED_KERNEL_WITH_INPUT(13, int8_t)
REGISTER_MUSA_UNSQUEEZE_TYPED_KERNEL_WITH_INPUT(13, int16_t)
REGISTER_MUSA_UNSQUEEZE_TYPED_KERNEL_WITH_INPUT(13, int32_t)
REGISTER_MUSA_UNSQUEEZE_TYPED_KERNEL_WITH_INPUT(13, int64_t)
REGISTER_MUSA_UNSQUEEZE_TYPED_KERNEL_WITH_INPUT(13, MLFloat16)
REGISTER_MUSA_UNSQUEEZE_TYPED_KERNEL_WITH_INPUT(13, float)
REGISTER_MUSA_UNSQUEEZE_TYPED_KERNEL_WITH_INPUT(13, bool)

#define REGISTER_MUSA_EXPAND_DIMS_TYPED_KERNEL(ver, T)                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                          \
      ExpandDims, kOnnxDomain, ver, T, kMusaExecutionProvider,            \
      (*KernelDefBuilder::Create())                                       \
          .Alias(0, 0)                                                    \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("Tdim", DataTypeImpl::GetTensorType<int64_t>()) \
          .InputMemoryType(OrtMemTypeCPUInput, 1),                        \
      Unsqueeze<T>);

REGISTER_MUSA_EXPAND_DIMS_TYPED_KERNEL(1, float)
REGISTER_MUSA_EXPAND_DIMS_TYPED_KERNEL(1, int32_t)
REGISTER_MUSA_EXPAND_DIMS_TYPED_KERNEL(1, int64_t)
REGISTER_MUSA_EXPAND_DIMS_TYPED_KERNEL(1, MLFloat16)
REGISTER_MUSA_EXPAND_DIMS_TYPED_KERNEL(1, bool)

#undef REGISTER_MUSA_EXPAND_DIMS_TYPED_KERNEL

// Register compute implementations for all supported types (only once to avoid duplication)
REGISTER_MUSA_UNSQUEEZE_TYPED_COMPUTE(uint8_t)
REGISTER_MUSA_UNSQUEEZE_TYPED_COMPUTE(uint16_t)
REGISTER_MUSA_UNSQUEEZE_TYPED_COMPUTE(uint32_t)
REGISTER_MUSA_UNSQUEEZE_TYPED_COMPUTE(uint64_t)
REGISTER_MUSA_UNSQUEEZE_TYPED_COMPUTE(int8_t)
REGISTER_MUSA_UNSQUEEZE_TYPED_COMPUTE(int16_t)
REGISTER_MUSA_UNSQUEEZE_TYPED_COMPUTE(int32_t)
REGISTER_MUSA_UNSQUEEZE_TYPED_COMPUTE(int64_t)
REGISTER_MUSA_UNSQUEEZE_TYPED_COMPUTE(MLFloat16)
REGISTER_MUSA_UNSQUEEZE_TYPED_COMPUTE(float)
REGISTER_MUSA_UNSQUEEZE_TYPED_COMPUTE(bool)

}  // namespace musa
}  // namespace onnxruntime
