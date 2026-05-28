// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/tensor/gather.h"
#include "core/providers/musa/tensor/gather_impl.h"
#include "core/providers/common.h"
#include "core/providers/musa/musa_execution_provider.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/musa_utils.h"

#include <algorithm>
#include <vector>

using onnxruntime::common::Status;
namespace onnxruntime {
namespace musa {

namespace {

Status ReadScalarAxis(const Tensor& axis_tensor, int64_t& axis) {
  ORT_RETURN_IF_NOT(axis_tensor.Shape().Size() == 1,
                    "GatherV2 axis must be a scalar, got shape ", axis_tensor.Shape().ToString());
  if (axis_tensor.IsDataType<int32_t>()) {
    axis = static_cast<int64_t>(*axis_tensor.Data<int32_t>());
    return Status::OK();
  }
  if (axis_tensor.IsDataType<int64_t>()) {
    axis = *axis_tensor.Data<int64_t>();
    return Status::OK();
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "GatherV2 axis must be int32 or int64");
}

Status PrepareGatherV2(const Tensor& params,
                       const Tensor& indices,
                       const Tensor& axis_tensor,
                       int64_t batch_dims_attr,
                       int64_t& axis,
                       int64_t& batch_dims,
                       TensorShape& output_shape) {
  ORT_RETURN_IF_ERROR(ReadScalarAxis(axis_tensor, axis));

  const auto& params_shape = params.Shape();
  const auto& indices_shape = indices.Shape();
  const int64_t params_rank = static_cast<int64_t>(params_shape.NumDimensions());
  const int64_t indices_rank = static_cast<int64_t>(indices_shape.NumDimensions());
  ORT_RETURN_IF_NOT(params_rank > 0, "GatherV2 params must have rank >= 1");

  axis = HandleNegativeAxis(axis, params_rank);
  batch_dims = batch_dims_attr;
  if (batch_dims < 0) {
    batch_dims += indices_rank;
  }

  ORT_RETURN_IF_NOT(batch_dims >= 0 && batch_dims <= std::min(axis, indices_rank),
                    "GatherV2 batch_dims must be in [0, min(axis, indices rank)], got batch_dims=",
                    batch_dims, ", axis=", axis, ", indices rank=", indices_rank);

  for (int64_t i = 0; i < batch_dims; ++i) {
    ORT_RETURN_IF_NOT(params_shape[static_cast<size_t>(i)] == indices_shape[static_cast<size_t>(i)],
                      "GatherV2 batch dimension ", i, " mismatch: params=",
                      params_shape[static_cast<size_t>(i)], ", indices=",
                      indices_shape[static_cast<size_t>(i)]);
  }

  TensorShapeVector output_dims;
  output_dims.reserve(static_cast<size_t>(params_rank + indices_rank - batch_dims - 1));
  for (int64_t i = 0; i < axis; ++i) {
    output_dims.push_back(params_shape[static_cast<size_t>(i)]);
  }
  for (int64_t i = batch_dims; i < indices_rank; ++i) {
    output_dims.push_back(indices_shape[static_cast<size_t>(i)]);
  }
  for (int64_t i = axis + 1; i < params_rank; ++i) {
    output_dims.push_back(params_shape[static_cast<size_t>(i)]);
  }
  output_shape = TensorShape(output_dims);
  return Status::OK();
}

}  // namespace

template <typename T>
Status Gather<T>::ComputeInternal(OpKernelContext* ctx) const {
  GatherBase::Prepare p;
  ORT_RETURN_IF_ERROR(PrepareForCompute(ctx, p));

  if (p.output_tensor->Shape().Size() == 0) {
    return Status::OK();
  }

  if (!p.indices_tensor->IsDataType<int32_t>() &&
      !p.indices_tensor->IsDataType<int64_t>()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Type for Tind not supported yet in Gather.");
  }

  const TensorShape& input_shape = p.input_tensor->Shape();
  const int64_t block_size = input_shape.SizeFromDimension(p.axis + 1);
  const int64_t input_block_size = input_shape.SizeFromDimension(p.axis);
  const int64_t output_block_size = p.indices_tensor->Shape().Size() * block_size;
  const int64_t indices_max = input_shape[p.axis];

  GatherImpl(
      Stream(ctx),
      input_block_size,
      indices_max,
      output_block_size,
      block_size,
      p.indices_tensor->DataRaw(),
      p.indices_tensor->DataType()->Size(),
      p.input_tensor->DataRaw(),
      p.input_tensor->DataType()->Size(),
      p.output_tensor->MutableDataRaw(),
      static_cast<size_t>(p.output_tensor->Shape().Size()));

  return Status::OK();
}

template <typename T>
Status GatherV2<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* params = ctx->Input<Tensor>(0);
  const Tensor* indices = ctx->Input<Tensor>(1);
  const Tensor* axis_tensor = ctx->Input<Tensor>(2);
  ORT_RETURN_IF_NOT(params != nullptr && indices != nullptr && axis_tensor != nullptr,
                    "GatherV2 inputs must not be null");
  ORT_RETURN_IF_NOT(indices->IsDataType<int32_t>() || indices->IsDataType<int64_t>(),
                    "GatherV2 indices must be int32 or int64");

  int64_t axis = 0;
  int64_t batch_dims = 0;
  TensorShape output_shape;
  ORT_RETURN_IF_ERROR(PrepareGatherV2(*params, *indices, *axis_tensor, batch_dims_, axis, batch_dims, output_shape));

  Tensor* output = ctx->Output(0, output_shape);
  ORT_RETURN_IF_NOT(output != nullptr, "GatherV2 failed to allocate output tensor");
  if (output->Shape().Size() == 0) {
    return Status::OK();
  }

  const auto* ep = static_cast<const MusaExecutionProvider*>(Info().GetExecutionProvider());
  MusaPreparation prepare(ep);
  ::musa::dnn::Tensor params_tensor;
  ::musa::dnn::Tensor indices_tensor;
  ::musa::dnn::Tensor output_tensor;
  ORT_RETURN_IF_ERROR(SetupMusaTensor(params_tensor, params, GetMusaDataType<T>(), &prepare));
  ORT_RETURN_IF_ERROR(SetupMusaTensor(indices_tensor, indices,
                                      indices->IsDataType<int32_t>() ? GetMusaDataType<int32_t>()
                                                                     : GetMusaDataType<int64_t>(),
                                      &prepare));
  ORT_RETURN_IF_ERROR(SetupMusaTensor(output_tensor, output, GetMusaDataType<T>(), &prepare));

  ::musa::dnn::GatherX gather_op;
  auto status = gather_op.SetMode(::musa::dnn::GatherX::Mode::GATHER);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set GatherV2 mode, status: ", static_cast<int>(status));
  }
  status = gather_op.SetAxis(static_cast<int>(axis));
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set GatherV2 axis, status: ", static_cast<int>(status));
  }
  status = gather_op.SetBatchDims(static_cast<int>(batch_dims));
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set GatherV2 batch_dims, status: ", static_cast<int>(status));
  }

  status = gather_op.Run(prepare.GetHandle(), output_tensor, indices_tensor, params_tensor);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MUSA GatherV2 failed, status: ", static_cast<int>(status));
  }

  return Status::OK();
}

// Macro for registering typed compute function with MUSA implementation
#define REGISTER_MUSA_GATHER_TYPED_COMPUTE(T)                               \
  template Status Gather<T>::ComputeInternal(OpKernelContext* ctx) const;

#define REGISTER_MUSA_GATHER_V2_TYPED_COMPUTE(T)                            \
  template Status GatherV2<T>::ComputeInternal(OpKernelContext* ctx) const;

// Macro for registering typed kernel
#define REGISTER_MUSA_GATHER_V2_TYPED_KERNEL(ver, T)                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                            \
      GatherV2, kOnnxDomain, ver, T, kMusaExecutionProvider,                \
      (*KernelDefBuilder::Create())                                         \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                           \
          .TypeConstraint("Tparams", DataTypeImpl::GetTensorType<T>())      \
          .TypeConstraint("Tindices", std::vector<MLDataType>{              \
                                      DataTypeImpl::GetTensorType<int32_t>(),\
                                      DataTypeImpl::GetTensorType<int64_t>()})\
          .TypeConstraint("Taxis", std::vector<MLDataType>{                 \
                                      DataTypeImpl::GetTensorType<int32_t>(),\
                                      DataTypeImpl::GetTensorType<int64_t>()}),\
      GatherV2<T>);

#define REGISTER_MUSA_GATHER_TYPED_KERNEL(ver, T)                           \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                               \
      Gather, kOnnxDomain, ver, T, kMusaExecutionProvider,                  \
      (*KernelDefBuilder::Create())                                            \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())              \
          .TypeConstraint("Tind", std::vector<MLDataType>{                    \
                                      DataTypeImpl::GetTensorType<int32_t>(), \
                                      DataTypeImpl::GetTensorType<int64_t>()}), \
      Gather<T>);

// Macro for registering versioned typed kernel
#define REGISTER_MUSA_GATHER_VERSIONED_TYPED_KERNEL(startver, endver, T)     \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                   \
      Gather, kOnnxDomain, startver, endver, T, kMusaExecutionProvider,      \
      (*KernelDefBuilder::Create())                                            \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())              \
          .TypeConstraint("Tind", std::vector<MLDataType>{                    \
                                      DataTypeImpl::GetTensorType<int32_t>(), \
                                      DataTypeImpl::GetTensorType<int64_t>()}), \
      Gather<T>);

// Combined macro for both kernel and compute registration
#define REGISTER_MUSA_GATHER_TYPED(ver, T)                                  \
  REGISTER_MUSA_GATHER_TYPED_KERNEL(ver, T)                                 \
  REGISTER_MUSA_GATHER_TYPED_COMPUTE(T)

#define REGISTER_MUSA_GATHER_V2_TYPED(ver, T)                               \
  REGISTER_MUSA_GATHER_V2_TYPED_KERNEL(ver, T)                              \
  REGISTER_MUSA_GATHER_V2_TYPED_COMPUTE(T)

// Macro for versioned typed registration
#define REGISTER_MUSA_GATHER_VERSIONED_TYPED(startver, endver, T)           \
  REGISTER_MUSA_GATHER_VERSIONED_TYPED_KERNEL(startver, endver, T)

// Register Gather operations for supported data types and versions

// Gather operations (version 1-10)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(1, 10, uint8_t)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(1, 10, uint16_t)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(1, 10, uint32_t)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(1, 10, uint64_t)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(1, 10, int8_t)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(1, 10, int16_t)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(1, 10, int32_t)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(1, 10, int64_t)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(1, 10, MLFloat16)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(1, 10, float)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(1, 10, double)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(1, 10, bool)

// Gather operations (version 11-12)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(11, 12, uint8_t)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(11, 12, uint16_t)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(11, 12, uint32_t)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(11, 12, uint64_t)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(11, 12, int8_t)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(11, 12, int16_t)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(11, 12, int32_t)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(11, 12, int64_t)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(11, 12, MLFloat16)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(11, 12, float)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(11, 12, double)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(11, 12, bool)

// Gather operations (version 13+)
REGISTER_MUSA_GATHER_TYPED(13, uint8_t)
REGISTER_MUSA_GATHER_TYPED(13, uint16_t)
REGISTER_MUSA_GATHER_TYPED(13, uint32_t)
REGISTER_MUSA_GATHER_TYPED(13, uint64_t)
REGISTER_MUSA_GATHER_TYPED(13, int8_t)
REGISTER_MUSA_GATHER_TYPED(13, int16_t)
REGISTER_MUSA_GATHER_TYPED(13, int32_t)
REGISTER_MUSA_GATHER_TYPED(13, int64_t)
REGISTER_MUSA_GATHER_TYPED(13, MLFloat16)
REGISTER_MUSA_GATHER_TYPED(13, float)
REGISTER_MUSA_GATHER_TYPED(13, double)
REGISTER_MUSA_GATHER_TYPED(13, bool)

REGISTER_MUSA_GATHER_V2_TYPED(1, int32_t)
REGISTER_MUSA_GATHER_V2_TYPED(1, int64_t)
REGISTER_MUSA_GATHER_V2_TYPED(1, MLFloat16)
REGISTER_MUSA_GATHER_V2_TYPED(1, float)
REGISTER_MUSA_GATHER_V2_TYPED(1, double)

} // namespace musa
} // namespace onnxruntime
