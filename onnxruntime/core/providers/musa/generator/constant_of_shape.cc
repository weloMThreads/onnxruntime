// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/common.h"
#include "constant_of_shape.h"
#include "core/providers/musa/musa_fwd.h"
#include <musa_runtime.h>
#include <mudnn.h>

using onnxruntime::common::Status;
namespace onnxruntime {
namespace musa {

// Get mudnn data type for template type T
template <typename T>
::musa::dnn::Tensor::Type GetMudnnDataType() {
  if constexpr (std::is_same_v<T, int8_t>) {
    return ::musa::dnn::Tensor::Type::INT8;
  }
  if constexpr (std::is_same_v<T, int16_t>) {
    return ::musa::dnn::Tensor::Type::INT16;
  }
  if constexpr (std::is_same_v<T, int32_t>) {
    return ::musa::dnn::Tensor::Type::INT32;
  }
  if constexpr (std::is_same_v<T, int64_t>) {
    return ::musa::dnn::Tensor::Type::INT64;
  }
  // Default to FLOAT for float and unsupported types
  return ::musa::dnn::Tensor::Type::FLOAT;
}

// Get mudnn tensor format based on shape dimensions
::musa::dnn::Tensor::Format GetMudnnTensorFormat(const TensorShape& shape) {
  if (shape.NumDimensions() <= 1) {
    return ::musa::dnn::Tensor::Format::NCW;
  }
  if (shape.NumDimensions() <= 4) {
    return ::musa::dnn::Tensor::Format::NCHW;
  }
  return ::musa::dnn::Tensor::Format::NCDHW;
}

// Setup mudnn tensor with given parameters
Status SetupMudnnTensor(::musa::dnn::Tensor& musa_tensor, void* output_data,
                        const TensorShape& shape, ::musa::dnn::Tensor::Type data_type) {
  auto status = musa_tensor.SetType(data_type);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set tensor type");
  }

  status = musa_tensor.SetAddr(output_data);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set tensor address");
  }

  auto format = GetMudnnTensorFormat(shape);
  status = musa_tensor.SetFormat(format);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set tensor format");
  }

  std::vector<int64_t> dims;
  if (shape.NumDimensions() == 0) {
    dims = {1}; // scalar case
  } else {
    for (size_t i = 0; i < shape.NumDimensions(); ++i) {
      dims.push_back(shape[i]);
    }
  }

  status = musa_tensor.SetNdInfo(static_cast<int>(dims.size()), dims.data());
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set tensor shape");
  }

  return Status::OK();
}

// Helper function to fill tensor data using mudnn Fill API
template <typename T>
Status MusaFillTensor(void* output_data, T value, size_t element_count,
                      const TensorShape& shape, musaStream_t stream, int device_id) {
  if (element_count == 0) {
    return Status::OK();
  }

  try {
    // Create mudnn handle
    ::musa::dnn::Handle handle(device_id);
    if (stream) {
      auto status = handle.SetStream(stream);
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Failed to set mudnn stream, status: ",
                               static_cast<int>(status));
      }
    }

    // Create and setup mudnn tensor
    ::musa::dnn::Tensor musa_tensor;
    auto data_type = GetMudnnDataType<T>();
    ORT_RETURN_IF_ERROR(SetupMudnnTensor(musa_tensor, output_data, shape, data_type));

    // Create and run Fill operation
    ::musa::dnn::Fill fill_op;

    // Set fill value based on data type
    ::musa::dnn::Status status = ::musa::dnn::Status::SUCCESS;  // default
    if constexpr (std::is_integral_v<T>) {
      status = fill_op.SetValue(static_cast<int64_t>(value));
    } else {
      status = fill_op.SetValue(static_cast<double>(value));
    }

    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set fill value");
    }

    // Run the fill operation
    status = fill_op.Run(handle, musa_tensor);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "mudnn Fill operation failed, status: ",
                             static_cast<int>(status));
    }


  } catch (const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Exception in mudnn Fill operation: ",
                           std::string(e.what()));
  }

  return Status::OK();
}

// Helper function to dispatch fill operation based on element size
Status DispatchFillOperation(void* output_data, const void* value_ptr, size_t element_size_bytes,
                             musaStream_t stream, size_t element_count, const TensorShape& shape, int device_id) {
  if (element_count == 0) {
    return Status::OK();
  }

  switch (element_size_bytes) {
    case sizeof(int8_t):
      return MusaFillTensor<int8_t>(output_data, *(reinterpret_cast<const int8_t*>(value_ptr)),
                                    element_count, shape, stream, device_id);
    case sizeof(int16_t):
      return MusaFillTensor<int16_t>(output_data, *(reinterpret_cast<const int16_t*>(value_ptr)),
                                     element_count, shape, stream, device_id);
    case sizeof(int32_t):
      return MusaFillTensor<int32_t>(output_data, *(reinterpret_cast<const int32_t*>(value_ptr)),
                                     element_count, shape, stream, device_id);
    case sizeof(int64_t):
      return MusaFillTensor<int64_t>(output_data, *(reinterpret_cast<const int64_t*>(value_ptr)),
                                     element_count, shape, stream, device_id);
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported value attribute datatype with sizeof=: ", element_size_bytes);
  }
}

Status ConstantOfShape::ComputeInternal(OpKernelContext* ctx) const {
  Tensor* output_tensor = nullptr;
  ORT_RETURN_IF_ERROR(PrepareCompute(ctx, &output_tensor));

  auto* output_data = output_tensor->MutableDataRaw();
  const auto size = output_tensor->Shape().Size();
  const void* value_ptr = GetValuePtr();
  const auto element_size = output_tensor->DataType()->Size();
  const auto& shape = output_tensor->Shape();

  // Get stream from context (may be null)
  auto* stream = Stream(ctx);

  return DispatchFillOperation(output_data, value_ptr, element_size, stream, size, shape,
                               Info().GetExecutionProvider()->GetDeviceId());
}

// Register the operator using the same pattern as other MUSA operators
#define REGISTER_MUSA_CONSTANT_OF_SHAPE_KERNEL(ver)                                    \
  ONNX_OPERATOR_KERNEL_EX(                                                             \
      ConstantOfShape,                                                                 \
      kOnnxDomain,                                                                     \
      ver,                                                                             \
      kMusaExecutionProvider,                                                          \
      (*KernelDefBuilder::Create())                                                    \
          .InputMemoryType(OrtMemTypeCPUInput, 0)                                      \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>())                \
          .TypeConstraint("T2", DataTypeImpl::AllFixedSizeTensorTypes()),              \
      ConstantOfShape);

// Register for opset 9 and later versions
REGISTER_MUSA_CONSTANT_OF_SHAPE_KERNEL(9)

}  // namespace musa
}  // namespace onnxruntime
