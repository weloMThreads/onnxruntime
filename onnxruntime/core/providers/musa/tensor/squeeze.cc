// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/tensor/squeeze.h"
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
Status SimpleMusaSqueezeOp(const MusaPreparation& prepare, musaStream_t stream) {
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

  // For Squeeze, if input and output point to the same memory location,
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
                             "MUSA memcpy failed in Squeeze operation, status: " +
                             std::to_string(static_cast<int>(musa_status)));
    }


  } catch (const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Exception in MUSA Squeeze operation: " +
                           std::string(e.what()));
  }

  return Status::OK();
}

template <typename T>
Status Squeeze<T>::Prepare(OpKernelContext* ctx, MusaPreparation& prepare) const {
  // 1. Get input tensor and validate
  const Tensor* input_tensor = ctx->Input<Tensor>(0);
  if (input_tensor == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input tensor is null");
  }

  const TensorShape& input_shape = input_tensor->Shape();

  // 2. Handle axes parameter (can be from attribute or input)
  TensorShapeVector axes;
  size_t num_inputs = ctx->InputCount();
  if (num_inputs == 2) {  // axes is an input
    const Tensor* axes_tensor = ctx->Input<Tensor>(1);
    ORT_ENFORCE(axes_tensor != nullptr, "Axes input is null");
    ORT_ENFORCE(axes_tensor->Shape().NumDimensions() == 1,
                "An axes tensor must be a vector tensor.");
    auto nDims = static_cast<size_t>(axes_tensor->Shape()[0]);
    const auto* data = axes_tensor->Data<int64_t>();
    axes.resize(nDims);
    for (size_t i = 0; i < nDims; ++i) {
      axes[i] = data[i];
    }
  } else {
    axes.assign(axes_.begin(), axes_.end());
  }

  // 3. Compute output shape using the base class function
  TensorShapeVector output_shape = ComputeOutputShape(input_shape, axes);

  // 4. Create output tensor
  Tensor* output_tensor = ctx->Output(0, TensorShape(output_shape));
  if (output_tensor == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Failed to create output tensor");
  }

  // 5. Store tensor pointers and shapes in preparation
  prepare.input_a_ptr = input_tensor->DataRaw();
  prepare.output_ptr = output_tensor->MutableDataRaw();
  prepare.output_size = output_tensor->Shape().Size();
  prepare.input_a_shape = input_tensor->Shape();
  prepare.output_shape = output_tensor->Shape();

  if (prepare.input_a_ptr == nullptr || prepare.output_ptr == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid tensor data pointers");
  }

  // 6. Get MUSA data type
  const auto musaType = GetMusaDataType<T>();

  // 7. Prepare MUSA tensors - using ORT_TRY/ORT_CATCH like other MusaEP operations
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
Status Squeeze<T>::ComputeInternal(OpKernelContext* ctx) const {
  // Prepare MUSA operation
  const auto* ep = static_cast<const MusaExecutionProvider*>(
      Info().GetExecutionProvider());
  MusaPreparation prepare(ep);
  ORT_RETURN_IF_ERROR(Prepare(ctx, prepare));

  // Call MUSA device squeeze operation using prepared data
  musaStream_t stream = Stream(ctx);
  ORT_RETURN_IF_ERROR(SimpleMusaSqueezeOp<T>(prepare, stream));

  return Status::OK();
}

// Macro for registering typed compute function with MUSA implementation
#define REGISTER_MUSA_SQUEEZE_TYPED_COMPUTE(T)                               \
  template Status Squeeze<T>::ComputeInternal(OpKernelContext* ctx) const;

// Macro for registering versioned typed kernel
#define REGISTER_MUSA_SQUEEZE_VERSIONED_TYPED_KERNEL(startver, endver, T)   \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                  \
      Squeeze, kOnnxDomain, startver, endver, T, kMusaExecutionProvider,    \
      (*KernelDefBuilder::Create())                                          \
          .Alias(0, 0)                                                       \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),            \
      Squeeze<T>);

// Macro for registering versioned typed kernel with input memory type
#define REGISTER_MUSA_SQUEEZE_VERSIONED_TYPED_KERNEL_WITH_INPUT(startver, endver, T) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                           \
      Squeeze, kOnnxDomain, startver, endver, T, kMusaExecutionProvider,            \
      (*KernelDefBuilder::Create())                                                  \
          .Alias(0, 0)                                                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                    \
          .InputMemoryType(OrtMemTypeCPUInput, 1),                                   \
      Squeeze<T>);

// Macro for registering non-versioned typed kernel (version 13 and above)
#define REGISTER_MUSA_SQUEEZE_TYPED_KERNEL(T)                                \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                             \
      Squeeze, kOnnxDomain, 13, T, kMusaExecutionProvider,                  \
      (*KernelDefBuilder::Create())                                          \
          .Alias(0, 0)                                                       \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())             \
          .InputMemoryType(OrtMemTypeCPUInput, 1),                           \
      Squeeze<T>);

// Register Squeeze operations for different versions and data types
// Version 1-10: axes as attribute
REGISTER_MUSA_SQUEEZE_VERSIONED_TYPED_KERNEL(1, 10, uint8_t)
REGISTER_MUSA_SQUEEZE_VERSIONED_TYPED_KERNEL(1, 10, uint16_t)
REGISTER_MUSA_SQUEEZE_VERSIONED_TYPED_KERNEL(1, 10, uint32_t)
REGISTER_MUSA_SQUEEZE_VERSIONED_TYPED_KERNEL(1, 10, uint64_t)
REGISTER_MUSA_SQUEEZE_VERSIONED_TYPED_KERNEL(1, 10, int8_t)
REGISTER_MUSA_SQUEEZE_VERSIONED_TYPED_KERNEL(1, 10, int16_t)
REGISTER_MUSA_SQUEEZE_VERSIONED_TYPED_KERNEL(1, 10, int32_t)
REGISTER_MUSA_SQUEEZE_VERSIONED_TYPED_KERNEL(1, 10, int64_t)
REGISTER_MUSA_SQUEEZE_VERSIONED_TYPED_KERNEL(1, 10, MLFloat16)
REGISTER_MUSA_SQUEEZE_VERSIONED_TYPED_KERNEL(1, 10, float)
REGISTER_MUSA_SQUEEZE_VERSIONED_TYPED_KERNEL(1, 10, bool)

// Version 11-12: axes as attribute, explicitly support negative axis
REGISTER_MUSA_SQUEEZE_VERSIONED_TYPED_KERNEL(11, 12, uint8_t)
REGISTER_MUSA_SQUEEZE_VERSIONED_TYPED_KERNEL(11, 12, uint16_t)
REGISTER_MUSA_SQUEEZE_VERSIONED_TYPED_KERNEL(11, 12, uint32_t)
REGISTER_MUSA_SQUEEZE_VERSIONED_TYPED_KERNEL(11, 12, uint64_t)
REGISTER_MUSA_SQUEEZE_VERSIONED_TYPED_KERNEL(11, 12, int8_t)
REGISTER_MUSA_SQUEEZE_VERSIONED_TYPED_KERNEL(11, 12, int16_t)
REGISTER_MUSA_SQUEEZE_VERSIONED_TYPED_KERNEL(11, 12, int32_t)
REGISTER_MUSA_SQUEEZE_VERSIONED_TYPED_KERNEL(11, 12, int64_t)
REGISTER_MUSA_SQUEEZE_VERSIONED_TYPED_KERNEL(11, 12, MLFloat16)
REGISTER_MUSA_SQUEEZE_VERSIONED_TYPED_KERNEL(11, 12, float)
REGISTER_MUSA_SQUEEZE_VERSIONED_TYPED_KERNEL(11, 12, bool)

// Version 13: axes as input instead of attribute, axes on CPU
REGISTER_MUSA_SQUEEZE_TYPED_KERNEL(uint8_t)
REGISTER_MUSA_SQUEEZE_TYPED_KERNEL(uint16_t)
REGISTER_MUSA_SQUEEZE_TYPED_KERNEL(uint32_t)
REGISTER_MUSA_SQUEEZE_TYPED_KERNEL(uint64_t)
REGISTER_MUSA_SQUEEZE_TYPED_KERNEL(int8_t)
REGISTER_MUSA_SQUEEZE_TYPED_KERNEL(int16_t)
REGISTER_MUSA_SQUEEZE_TYPED_KERNEL(int32_t)
REGISTER_MUSA_SQUEEZE_TYPED_KERNEL(int64_t)
REGISTER_MUSA_SQUEEZE_TYPED_KERNEL(MLFloat16)
REGISTER_MUSA_SQUEEZE_TYPED_KERNEL(float)
REGISTER_MUSA_SQUEEZE_TYPED_KERNEL(bool)

// Register compute implementations for all supported types (only once to avoid duplication)
REGISTER_MUSA_SQUEEZE_TYPED_COMPUTE(uint8_t)
REGISTER_MUSA_SQUEEZE_TYPED_COMPUTE(uint16_t)
REGISTER_MUSA_SQUEEZE_TYPED_COMPUTE(uint32_t)
REGISTER_MUSA_SQUEEZE_TYPED_COMPUTE(uint64_t)
REGISTER_MUSA_SQUEEZE_TYPED_COMPUTE(int8_t)
REGISTER_MUSA_SQUEEZE_TYPED_COMPUTE(int16_t)
REGISTER_MUSA_SQUEEZE_TYPED_COMPUTE(int32_t)
REGISTER_MUSA_SQUEEZE_TYPED_COMPUTE(int64_t)
REGISTER_MUSA_SQUEEZE_TYPED_COMPUTE(MLFloat16)
REGISTER_MUSA_SQUEEZE_TYPED_COMPUTE(float)
REGISTER_MUSA_SQUEEZE_TYPED_COMPUTE(bool)

}  // namespace musa
}  // namespace onnxruntime
