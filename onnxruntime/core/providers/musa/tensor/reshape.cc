// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/tensor/reshape.h"
#include "core/providers/common.h"
#include "core/providers/musa/musa_fwd.h"
#include <musa_runtime.h>
#include <mudnn.h>

using onnxruntime::common::Status;
namespace onnxruntime {
namespace musa {

Status Reshape::ComputeInternal(OpKernelContext* ctx) const {
  // 1. Get input tensors
  const Tensor* data_tensor = ctx->Input<Tensor>(0);
  const Tensor* shape_tensor = ctx->Input<Tensor>(1);
  
  if (!data_tensor || !shape_tensor) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, 
                           "Reshape: data_tensor or shape_tensor is null");
  }
  
  // 2. Infer output shape
  TensorShape output_shape;
  ORT_RETURN_IF_ERROR(InferOutputShape(data_tensor, shape_tensor, output_shape));
  
  // 3. Allocate output tensor
  Tensor* output_tensor = ctx->Output(0, output_shape);
  if (!output_tensor) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor");
  }
  
  // 4. Execute reshape operation
  return ExecuteMusaReshape(ctx, data_tensor, output_tensor, output_shape);
}

Status Reshape::InferOutputShape(const Tensor* data, const Tensor* shape,
                                TensorShape& output_shape) const {
  // Input validation
  ORT_ENFORCE(shape != nullptr, "Shape tensor cannot be null");
  ORT_ENFORCE(shape->Shape().NumDimensions() == 1, 
              "Shape tensor must be 1-dimensional");
  ORT_ENFORCE(shape->Location().device.Type() == OrtDevice::CPU, 
              "Shape tensor must be on CPU");
  
  // Use ReshapeHelper for shape inference
  auto shape_span = shape->DataAsSpan<int64_t>();
  TensorShapeVector shape_vector(shape_span.begin(), shape_span.end());
  ReshapeHelper helper(data->Shape(), shape_vector, allow_zero_);
  
  output_shape = TensorShape(shape_vector);
  return Status::OK();
}

bool Reshape::CanReshapeInPlace(const Tensor* input, 
                               const TensorShape& target_shape) const {
  // Check element count
  if (input->Shape().Size() != target_shape.Size()) {
    return false;
  }
  
  // For MUSA devices, most reshapes can be done in-place (zero-copy)
  return true;
}

Status Reshape::ExecuteMusaReshape(OpKernelContext* ctx, const Tensor* input,
                                  Tensor* output, const TensorShape& target_shape) const {
  // Check if pointers are the same (true zero-copy via Alias)
  if (input->DataRaw() == output->DataRaw()) {
    // True zero-copy case
    return Status::OK();
  }
  
  // Alias mechanism failed - need to copy data manually
  // Use standard ORT memory copy mechanism
  auto* data_transfer = Info().GetDataTransferManager().GetDataTransfer(
      input->Location().device, output->Location().device);
  
  if (!data_transfer) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "No data transfer found for reshape");
  }
  
  // Check if we have a compute stream available for async copy
  auto* stream = ctx->GetComputeStream();
  Status copy_status;
  if (stream) {
    copy_status = data_transfer->CopyTensorAsync(*input, *output, *stream);
  } else {
    copy_status = data_transfer->CopyTensor(*input, *output);
  }
  
  if (!copy_status.IsOK()) {
    return copy_status;
  }
  
  return Status::OK();
}

// Reshape_1 implementation (v1-4 version)
Status Reshape_1::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* input = ctx->Input<Tensor>(0);
  if (!input) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Reshape_1: input is null");
  }
  
  TensorShapeVector target_shape = shape_;
  
  // Use ReshapeHelper to handle shape
  ReshapeHelper helper(input->Shape(), target_shape);
  Tensor* output = ctx->Output(0, TensorShape(target_shape));
  if (!output) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor");
  }
  
  // Simple case: zero-copy through Alias mechanism
  return Status::OK();
}

// Registration macros
#define REGISTER_MUSA_RESHAPE_TYPED_KERNEL(ver_start, ver_end, T) \
    ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX( \
        Reshape, kOnnxDomain, ver_start, ver_end, T, kMusaExecutionProvider, \
        (*KernelDefBuilder::Create()) \
            .Alias(0, 0) \
            .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()) \
            .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>()) \
            .InputMemoryType(OrtMemTypeCPUInput, 1), \
        Reshape);

#define REGISTER_MUSA_RESHAPE_LATEST_TYPED_KERNEL(ver, T) \
    ONNX_OPERATOR_TYPED_KERNEL_EX( \
        Reshape, kOnnxDomain, ver, T, kMusaExecutionProvider, \
        (*KernelDefBuilder::Create()) \
            .Alias(0, 0) \
            .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()) \
            .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>()) \
            .InputMemoryType(OrtMemTypeCPUInput, 1), \
        Reshape);

#define REGISTER_MUSA_RESHAPE_1_TYPED_KERNEL(ver_start, ver_end, T) \
    ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX( \
        Reshape, kOnnxDomain, ver_start, ver_end, T, kMusaExecutionProvider, \
        (*KernelDefBuilder::Create()) \
            .Alias(0, 0) \
            .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
        Reshape_1);

// Register for ONNX v1-4 (Reshape_1)
REGISTER_MUSA_RESHAPE_1_TYPED_KERNEL(1, 4, int32_t)
REGISTER_MUSA_RESHAPE_1_TYPED_KERNEL(1, 4, int64_t)
REGISTER_MUSA_RESHAPE_1_TYPED_KERNEL(1, 4, MLFloat16)
REGISTER_MUSA_RESHAPE_1_TYPED_KERNEL(1, 4, float)

// Register for ONNX v5-12
REGISTER_MUSA_RESHAPE_TYPED_KERNEL(5, 12, int32_t)
REGISTER_MUSA_RESHAPE_TYPED_KERNEL(5, 12, int64_t)
REGISTER_MUSA_RESHAPE_TYPED_KERNEL(5, 12, MLFloat16)
REGISTER_MUSA_RESHAPE_TYPED_KERNEL(5, 12, float)

// Register for ONNX v13
REGISTER_MUSA_RESHAPE_TYPED_KERNEL(13, 13, int32_t)
REGISTER_MUSA_RESHAPE_TYPED_KERNEL(13, 13, int64_t)
REGISTER_MUSA_RESHAPE_TYPED_KERNEL(13, 13, MLFloat16)
REGISTER_MUSA_RESHAPE_TYPED_KERNEL(13, 13, float)

// Register for ONNX v14-18
REGISTER_MUSA_RESHAPE_TYPED_KERNEL(14, 18, int32_t)
REGISTER_MUSA_RESHAPE_TYPED_KERNEL(14, 18, int64_t)
REGISTER_MUSA_RESHAPE_TYPED_KERNEL(14, 18, MLFloat16)
REGISTER_MUSA_RESHAPE_TYPED_KERNEL(14, 18, float)

// Register for ONNX v19+ (latest)
REGISTER_MUSA_RESHAPE_LATEST_TYPED_KERNEL(19, int32_t)
REGISTER_MUSA_RESHAPE_LATEST_TYPED_KERNEL(19, int64_t)
REGISTER_MUSA_RESHAPE_LATEST_TYPED_KERNEL(19, MLFloat16)
REGISTER_MUSA_RESHAPE_LATEST_TYPED_KERNEL(19, float)

} // namespace musa
} // namespace onnxruntime