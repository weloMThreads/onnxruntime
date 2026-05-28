// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/math/topk.h"

#include "core/providers/musa/math/topk_impl.h"
#include "core/providers/musa/musa_call.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/shared_library/provider_api.h"

#include <type_traits>

namespace onnxruntime {
namespace musa {
namespace {

int64_t NormalizeAxis(int64_t axis, size_t rank) {
  if (axis < 0) {
    axis += static_cast<int64_t>(rank);
  }
  return axis;
}

Status ReadK(const Tensor* k_tensor, int64_t& k) {
  ORT_RETURN_IF_NOT(k_tensor != nullptr, "TopK: K input is null");
  ORT_RETURN_IF_NOT(k_tensor->Shape().Size() == 1, "TopK: K input must contain exactly one element");
  k = k_tensor->Data<int64_t>()[0];
  ORT_RETURN_IF_NOT(k >= 0, "TopK: K must be non-negative");
  return Status::OK();
}

}  // namespace

template <typename T>
Status TopK<T>::ComputeInternal(OpKernelContext* ctx) const {
  const auto* input = ctx->Input<Tensor>(0);
  const auto* k_tensor = ctx->Input<Tensor>(1);
  ORT_RETURN_IF_NOT(input != nullptr, "TopK: input tensor is null");

  const TensorShape& input_shape = input->Shape();
  const size_t rank = input_shape.NumDimensions();
  ORT_RETURN_IF_NOT(rank > 0, "TopK: input tensor must have rank >= 1");

  int64_t k = 0;
  ORT_RETURN_IF_ERROR(ReadK(k_tensor, k));

  const int64_t axis = NormalizeAxis(axis_, rank);
  ORT_RETURN_IF_NOT(axis >= 0 && axis < static_cast<int64_t>(rank),
                    "TopK: axis ", axis_, " is out of range for rank ", rank);
  const int64_t axis_dim = input_shape[static_cast<size_t>(axis)];
  ORT_RETURN_IF_NOT(k <= axis_dim, "TopK: K must not exceed selected axis dimension. K=", k,
                    ", axis_dim=", axis_dim);

  TensorShapeVector output_dims = input_shape.AsShapeVector();
  output_dims[static_cast<size_t>(axis)] = k;
  Tensor* values = ctx->Output(0, TensorShape(output_dims));
  Tensor* indices = ctx->Output(1, TensorShape(output_dims));
  ORT_RETURN_IF_NOT(values != nullptr, "TopK: values output is null");
  ORT_RETURN_IF_NOT(indices != nullptr, "TopK: indices output is null");

  if (input_shape.Size() == 0 || k == 0) {
    return Status::OK();
  }

  TopKLaunchParams params;
  params.outer_size = input_shape.SizeToDimension(static_cast<size_t>(axis));
  params.axis_dim = axis_dim;
  params.inner_size = input_shape.SizeFromDimension(static_cast<size_t>(axis) + 1);
  params.k = k;
  params.largest = largest_;

  if constexpr (std::is_same_v<T, MLFloat16>) {
    LaunchTopKKernelHalf(Stream(ctx), input->DataRaw(), values->MutableDataRaw(), indices->MutableData<int64_t>(), params);
  } else {
    LaunchTopKKernel<T>(Stream(ctx), input->Data<T>(), values->MutableData<T>(), indices->MutableData<int64_t>(), params);
  }

  MUSA_RETURN_IF_ERROR(musaGetLastError());
  return Status::OK();
}

template class TopK<float>;
template class TopK<double>;
template class TopK<MLFloat16>;
template class TopK<int32_t>;
template class TopK<int64_t>;

#define MUSA_TOPK_KERNEL_DEF (*KernelDefBuilder::Create()).InputMemoryType(OrtMemTypeCPUInput, 1)

#define REGISTER_MUSA_TOPK_VERSIONED_TYPED(startver, endver, T)                                            \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(TopK, kOnnxDomain, startver, endver, T, kMusaExecutionProvider, \
                                          MUSA_TOPK_KERNEL_DEF.TypeConstraint("T", DataTypeImpl::GetTensorType<T>()) \
                                              .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()), \
                                          TopK<T>);

#define REGISTER_MUSA_TOPK_TYPED(ver, T)                                                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(TopK, kOnnxDomain, ver, T, kMusaExecutionProvider,                       \
                                MUSA_TOPK_KERNEL_DEF.TypeConstraint("T", DataTypeImpl::GetTensorType<T>()) \
                                    .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),        \
                                TopK<T>);

REGISTER_MUSA_TOPK_VERSIONED_TYPED(11, 23, float)
REGISTER_MUSA_TOPK_VERSIONED_TYPED(11, 23, double)
REGISTER_MUSA_TOPK_VERSIONED_TYPED(11, 23, MLFloat16)
REGISTER_MUSA_TOPK_VERSIONED_TYPED(11, 23, int32_t)
REGISTER_MUSA_TOPK_VERSIONED_TYPED(11, 23, int64_t)

REGISTER_MUSA_TOPK_TYPED(24, float)
REGISTER_MUSA_TOPK_TYPED(24, double)
REGISTER_MUSA_TOPK_TYPED(24, MLFloat16)
REGISTER_MUSA_TOPK_TYPED(24, int32_t)
REGISTER_MUSA_TOPK_TYPED(24, int64_t)

#undef REGISTER_MUSA_TOPK_TYPED
#undef REGISTER_MUSA_TOPK_VERSIONED_TYPED
#undef MUSA_TOPK_KERNEL_DEF

}  // namespace musa
}  // namespace onnxruntime
