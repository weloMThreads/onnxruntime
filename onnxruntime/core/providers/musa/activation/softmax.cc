// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/activation/softmax.h"
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/common.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/musa_execution_provider.h"
#include <musa_runtime.h>
#include <mudnn.h>

using onnxruntime::common::Status;
namespace onnxruntime {
namespace musa {

// Specialized tensor setup for Softmax to handle format and axis mapping correctly
Status SetupSoftmaxMusaTensor(::musa::dnn::Tensor &musa_tensor,
                              const Tensor *ort_tensor,
                              ::musa::dnn::Tensor::Type data_type,
                              bool is_output) {
  
  // Set tensor type
  auto status = musa_tensor.SetType(data_type);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set MUSA tensor type for Softmax");
  }

  // Set data address
  const void *data_ptr = ort_tensor->DataRaw();
  const auto &shape = ort_tensor->Shape();

  if (!data_ptr && shape.Size() > 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "ORT tensor data pointer is null for non-empty tensor");
  }

  if (data_ptr) {
    status = musa_tensor.SetAddr(data_ptr);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set MUSA tensor address for Softmax");
    }
  }

  // Choose format based on tensor dimensions for Softmax
  ::musa::dnn::Tensor::Format format;
  if (shape.NumDimensions() == 1) {
    format = ::musa::dnn::Tensor::Format::NCW;
  } else if (shape.NumDimensions() == 3) {
    // For 3D tensors, use NCW format (N=batch, C=channels, W=width)
    format = ::musa::dnn::Tensor::Format::NCW;
  } else if (shape.NumDimensions() == 4) {
    format = ::musa::dnn::Tensor::Format::NCHW;
  } else {
    // Default format for other cases
    format = ::musa::dnn::Tensor::Format::NCHW;
  }

  status = musa_tensor.SetFormat(format);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set MUSA tensor format for Softmax");
  }

  // Set tensor shape
  if (shape.NumDimensions() == 0) {
    std::vector<int64_t> scalar_dims = {1};
    status = musa_tensor.SetNdInfo(static_cast<int>(scalar_dims.size()), scalar_dims.data());
  } else {
    std::vector<int64_t> dims;
    for (size_t i = 0; i < shape.NumDimensions(); ++i) {
      dims.push_back(shape[i]);
    }
    status = musa_tensor.SetNdInfo(static_cast<int>(dims.size()), dims.data());
  }

  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set MUSA tensor shape for Softmax");
  }

  return Status::OK();
}

// MUSA device-based Softmax implementation using MUSA Softmax API
template <typename T>
Status SimpleMusaSoftmaxOp(const MusaPreparation& prepare, 
                           int axis, 
                           bool is_log_softmax) {
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

  // Use mudnn Softmax class for device computation
  try {
    // Create mudnn Softmax operation
    ::musa::dnn::Softmax softmax_op;

    // Set the operation mode
    auto mode = is_log_softmax ? ::musa::dnn::Softmax::Mode::LOGSOFTMAX 
                               : ::musa::dnn::Softmax::Mode::SOFTMAX;
    auto status = softmax_op.SetMode(mode);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set mudnn Softmax mode");
    }

    // Set the algorithm to ACCURATE (following torch_musa's successful pattern)
    status = softmax_op.SetAlgorithm(::musa::dnn::Softmax::Algorithm::ACCURATE);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set mudnn Softmax algorithm");
    }

    // Set the axis (dimension) for softmax computation
    status = softmax_op.SetDim(axis);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set mudnn Softmax dimension");
    }

    // Run the softmax operation - correct parameter order: handle, output, input
    status = softmax_op.Run(prepare.GetHandle(),
                           const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]),  // output tensor (non-const)
                           prepare.inputTensors[0]);  // input tensor (const)

    if (status != ::musa::dnn::Status::SUCCESS) {
      std::string op_name = is_log_softmax ? "LogSoftmax" : "Softmax";
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "mudnn " + op_name + " operation failed, status: " +
                             std::to_string(static_cast<int>(status)));
    }

  } catch (const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Exception in mudnn Softmax operation: " +
                           std::string(e.what()));
  }

  return Status::OK();
}

template <typename T>
Status Softmax<T>::Prepare(OpKernelContext* ctx, MusaPreparation& prepare) const {
  // Get input and output tensors
  const auto* input_tensor = ctx->Input<Tensor>(0);
  auto* output_tensor = ctx->Output(0, input_tensor->Shape());

  if (!input_tensor || !output_tensor) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input or output tensor");
  }

  // Verify input and output tensors have the same shape
  if (input_tensor->Shape() != output_tensor->Shape()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input and output tensor shapes must match");
  }

  // Setup tensor data pointers and shapes
  prepare.input_a_ptr = input_tensor->Data<T>();
  prepare.output_ptr = output_tensor->MutableData<T>();
  prepare.input_a_shape = input_tensor->Shape();
  prepare.output_shape = output_tensor->Shape();

  if (!prepare.input_a_ptr || !prepare.output_ptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid tensor data pointers");
  }

  // Convert input tensor to MUSA tensor format - use specialized setup for Softmax
  ::musa::dnn::Tensor input_musa_tensor;
  ORT_RETURN_IF_ERROR(SetupSoftmaxMusaTensor(input_musa_tensor, input_tensor, GetMusaDataType<T>(), false));
  prepare.inputTensors.push_back(input_musa_tensor);

  // Convert output tensor to MUSA tensor format - use specialized setup for Softmax
  ::musa::dnn::Tensor output_musa_tensor;
  ORT_RETURN_IF_ERROR(SetupSoftmaxMusaTensor(output_musa_tensor, output_tensor, GetMusaDataType<T>(), true));
  prepare.outputTensors.push_back(output_musa_tensor);

  return Status::OK();
}

template <typename T>
Status Softmax<T>::ComputeInternal(OpKernelContext* ctx) const {
  // Prepare MUSA resources and tensors
  const auto* ep = static_cast<const MusaExecutionProvider*>(
      Info().GetExecutionProvider());
  MusaPreparation prepare(ep);
  ORT_RETURN_IF_ERROR(Prepare(ctx, prepare));

  // Get input tensor to determine axis handling
  const auto* input_tensor = ctx->Input<Tensor>(0);
  int64_t rank = static_cast<int64_t>(input_tensor->Shape().NumDimensions());
  
  // Handle negative axis and normalize to positive axis
  int normalized_axis = axis_;
  if (normalized_axis < 0) {
    normalized_axis += static_cast<int>(rank);
  }
  
  // Validate axis range
  if (normalized_axis < 0 || normalized_axis >= rank) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Softmax axis " + std::to_string(axis_) +
                           " is out of range for tensor of rank " + std::to_string(rank));
  }

  // MUSA Softmax with mudnn::Softmax::SetDim naturally supports opset-13+ semantics
  // For opset < 13, only last axis is supported due to semantic differences:
  // - opset < 13: flatten-reshape semantics (softmax_2d on reshaped tensor)
  // - opset >= 13: direct axis computation (what mudnn::Softmax::SetDim does)
  if (opset_version_ < 13 && normalized_axis != rank - 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "MUSA Softmax supports arbitrary axis only for opset >= 13. "
                           "For opset < 13, only the last axis (axis=" + std::to_string(rank - 1) +
                           ") is supported, requested axis=" + std::to_string(normalized_axis) +
                           ". This is due to different ONNX opset semantics.");
  }

  // Call the device implementation with arbitrary axis support (opset >= 13 or last axis)
  return SimpleMusaSoftmaxOp<T>(prepare, normalized_axis, log_softmax_);
}

// Explicit template instantiations for supported types
// Note: MUSA doesn't support double and bfloat16 for Softmax
template class Softmax<float>;
template class Softmax<MLFloat16>;

// Register MUSA Softmax kernels
#define REGISTER_MUSA_SOFTMAX_TYPED_KERNEL(ver, T) \
  ONNX_OPERATOR_TYPED_KERNEL_EX( \
      Softmax, kOnnxDomain, ver, T, kMusaExecutionProvider, \
      (*KernelDefBuilder::Create()) \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Softmax<T>);

#define REGISTER_MUSA_SOFTMAX_VERSIONED_TYPED_KERNEL(start_ver, end_ver, T) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX( \
      Softmax, kOnnxDomain, start_ver, end_ver, T, kMusaExecutionProvider, \
      (*KernelDefBuilder::Create()) \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Softmax<T>);

// Register LogSoftmax kernels
#define REGISTER_MUSA_LOGSOFTMAX_TYPED_KERNEL(ver, T) \
  ONNX_OPERATOR_TYPED_KERNEL_EX( \
      LogSoftmax, kOnnxDomain, ver, T, kMusaExecutionProvider, \
      (*KernelDefBuilder::Create()) \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Softmax<T>);

#define REGISTER_MUSA_LOGSOFTMAX_VERSIONED_TYPED_KERNEL(start_ver, end_ver, T) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX( \
      LogSoftmax, kOnnxDomain, start_ver, end_ver, T, kMusaExecutionProvider, \
      (*KernelDefBuilder::Create()) \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Softmax<T>);

// Register all Softmax versions and types
REGISTER_MUSA_SOFTMAX_VERSIONED_TYPED_KERNEL(1, 10, float)
REGISTER_MUSA_SOFTMAX_VERSIONED_TYPED_KERNEL(1, 10, MLFloat16)
REGISTER_MUSA_SOFTMAX_VERSIONED_TYPED_KERNEL(11, 12, float)
REGISTER_MUSA_SOFTMAX_VERSIONED_TYPED_KERNEL(11, 12, MLFloat16)
REGISTER_MUSA_SOFTMAX_TYPED_KERNEL(13, float)
REGISTER_MUSA_SOFTMAX_TYPED_KERNEL(13, MLFloat16)

// Register all LogSoftmax versions and types  
REGISTER_MUSA_LOGSOFTMAX_VERSIONED_TYPED_KERNEL(1, 10, float)
REGISTER_MUSA_LOGSOFTMAX_VERSIONED_TYPED_KERNEL(1, 10, MLFloat16)
REGISTER_MUSA_LOGSOFTMAX_VERSIONED_TYPED_KERNEL(11, 12, float) 
REGISTER_MUSA_LOGSOFTMAX_VERSIONED_TYPED_KERNEL(11, 12, MLFloat16)
REGISTER_MUSA_LOGSOFTMAX_TYPED_KERNEL(13, float)
REGISTER_MUSA_LOGSOFTMAX_TYPED_KERNEL(13, MLFloat16)

}  // namespace musa
}  // namespace onnxruntime