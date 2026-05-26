// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"

#include "core/providers/common.h"
#include "core/providers/musa/math/element_wise_ops.h"
#include "core/providers/musa/math/not_impl.h"
#include "core/providers/musa/musa_call.h"
#include "core/providers/musa/musa_fwd.h"

using onnxruntime::common::Status;

namespace onnxruntime {
namespace musa {

template <typename T>
Status Not<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* input = ctx->Input<Tensor>(0);
  Tensor* output = ctx->Output(0, input->Shape());
  const int64_t count = input->Shape().Size();
  if (count == 0) {
    return Status::OK();
  }

  NotImpl(Stream(ctx), input->Data<bool>(), output->MutableData<bool>(), count);
  MUSA_RETURN_IF_ERROR(musaGetLastError());
  return Status::OK();
}

template class Not<bool>;

ONNX_OPERATOR_TYPED_KERNEL_EX(
    Not,
    kOnnxDomain,
    1,
    bool,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<bool>()),
    Not<bool>);

}  // namespace musa
}  // namespace onnxruntime
