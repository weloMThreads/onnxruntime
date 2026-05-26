// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "contrib_ops/musa/activation/activations.h"
#include "contrib_ops/musa/activation/activations_impl.h"

namespace onnxruntime {
namespace contrib {
namespace musa {

#define REGISTER_QUICKGELU_KERNEL_TYPED(T)                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                   \
      QuickGelu,                                                   \
      kMSDomain,                                                   \
      1,                                                           \
      T,                                                           \
      kMusaExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())   \
          .MayInplace(0, 0),                                       \
      QuickGelu<T>);

REGISTER_QUICKGELU_KERNEL_TYPED(float)
REGISTER_QUICKGELU_KERNEL_TYPED(MLFloat16)

template <>
Status QuickGelu<float>::ComputeInternal(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  ORT_ENFORCE(X);
  auto* Y = context->Output(0, X->Shape());
  ORT_ENFORCE(Y);

  const auto input_size = X->Shape().Size();
  if (input_size == 0) return Status::OK();

  LaunchQuickGeluKernel<float>(
      Stream(context), input_size,
      X->Data<float>(), Y->MutableData<float>(), alpha_);

  return Status::OK();
}

template <>
Status QuickGelu<MLFloat16>::ComputeInternal(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  ORT_ENFORCE(X);
  auto* Y = context->Output(0, X->Shape());
  ORT_ENFORCE(Y);

  const auto input_size = X->Shape().Size();
  if (input_size == 0) return Status::OK();

  LaunchQuickGeluKernelHalf(
      Stream(context), input_size,
      X->DataRaw(), Y->MutableDataRaw(), alpha_);

  return Status::OK();
}

}  // namespace musa
}  // namespace contrib
}  // namespace onnxruntime
