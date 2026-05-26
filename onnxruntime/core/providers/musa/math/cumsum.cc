// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"

#include "core/providers/common.h"
#include "core/providers/cpu/math/cumsum.h"
#include "core/providers/musa/math/cumsum.h"
#include "core/providers/musa/math/cumsum_impl.h"
#include "core/providers/musa/musa_call.h"
#include "core/providers/musa/musa_fwd.h"

using onnxruntime::common::Status;

namespace onnxruntime {
namespace musa {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    CumSum,
    kOnnxDomain,
    11, 13,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .TypeConstraint("T", std::vector<MLDataType>{
                                 DataTypeImpl::GetTensorType<int32_t>(),
                                 DataTypeImpl::GetTensorType<int64_t>(),
                                 DataTypeImpl::GetTensorType<uint32_t>(),
                                 DataTypeImpl::GetTensorType<uint64_t>(),
                                 DataTypeImpl::GetTensorType<float>(),
                                 DataTypeImpl::GetTensorType<double>()})
        .TypeConstraint("T2", std::vector<MLDataType>{
                                  DataTypeImpl::GetTensorType<int32_t>(),
                                  DataTypeImpl::GetTensorType<int64_t>()}),
    CumSum);

ONNX_OPERATOR_KERNEL_EX(
    CumSum,
    kOnnxDomain,
    14,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .TypeConstraint("T", std::vector<MLDataType>{
                                 DataTypeImpl::GetTensorType<int32_t>(),
                                 DataTypeImpl::GetTensorType<int64_t>(),
                                 DataTypeImpl::GetTensorType<uint32_t>(),
                                 DataTypeImpl::GetTensorType<uint64_t>(),
                                 DataTypeImpl::GetTensorType<float>(),
                                 DataTypeImpl::GetTensorType<double>(),
                                 DataTypeImpl::GetTensorType<MLFloat16>()})
        .TypeConstraint("T2", std::vector<MLDataType>{
                                  DataTypeImpl::GetTensorType<int32_t>(),
                                  DataTypeImpl::GetTensorType<int64_t>()}),
    CumSum);

Status CumSum::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* input = ctx->Input<Tensor>(0);
  const auto rank = static_cast<int64_t>(input->Shape().NumDimensions());
  ORT_RETURN_IF_NOT(rank != 0, "Cannot apply CumSum operator on a scalar");

  int64_t axis = 0;
  ORT_THROW_IF_ERROR(cumsum_op::GetAxis(ctx->Input<Tensor>(1), rank, axis));

  TensorShape output_shape(input->Shape());
  Tensor& output = *ctx->Output(0, output_shape);
  if (output_shape.Size() == 0) {
    return Status::OK();
  }

  const auto& input_dims = input->Shape().GetDims();
  int64_t current_dim = rank - 1;
  int64_t input_stride_along_axis = 1;
  while (current_dim > axis) {
    input_stride_along_axis *= input_dims[current_dim--];
  }

  fast_divmod input_dim_fdm(static_cast<int>(input_dims[axis]));
  fast_divmod input_stride_fdm(static_cast<int>(input_stride_along_axis));
  const int64_t output_size = output_shape.Size();

  if (input->IsDataType<float>()) {
    CumSumImpl(Stream(ctx), input->Data<float>(), input_dim_fdm, input_stride_fdm,
               output.MutableData<float>(), output_size, exclusive_, reverse_);
  } else if (input->IsDataType<double>()) {
    CumSumImpl(Stream(ctx), input->Data<double>(), input_dim_fdm, input_stride_fdm,
               output.MutableData<double>(), output_size, exclusive_, reverse_);
  } else if (input->IsDataType<int32_t>()) {
    CumSumImpl(Stream(ctx), input->Data<int32_t>(), input_dim_fdm, input_stride_fdm,
               output.MutableData<int32_t>(), output_size, exclusive_, reverse_);
  } else if (input->IsDataType<int64_t>()) {
    CumSumImpl(Stream(ctx), input->Data<int64_t>(), input_dim_fdm, input_stride_fdm,
               output.MutableData<int64_t>(), output_size, exclusive_, reverse_);
  } else if (input->IsDataType<uint32_t>()) {
    CumSumImpl(Stream(ctx), input->Data<uint32_t>(), input_dim_fdm, input_stride_fdm,
               output.MutableData<uint32_t>(), output_size, exclusive_, reverse_);
  } else if (input->IsDataType<uint64_t>()) {
    CumSumImpl(Stream(ctx), input->Data<uint64_t>(), input_dim_fdm, input_stride_fdm,
               output.MutableData<uint64_t>(), output_size, exclusive_, reverse_);
  } else if (input->IsDataType<MLFloat16>()) {
    CumSumImplHalf(Stream(ctx), input->DataRaw(), input_dim_fdm, input_stride_fdm,
                   output.MutableDataRaw(), output_size, exclusive_, reverse_);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Unsupported input data type to the CumSum op: ",
                           input->DataType());
  }

  MUSA_RETURN_IF_ERROR(musaGetLastError());
  return Status::OK();
}

}  // namespace musa
}  // namespace onnxruntime
