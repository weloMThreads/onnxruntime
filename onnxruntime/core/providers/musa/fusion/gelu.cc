// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/fusion/gelu.h"
#include "core/providers/musa/fusion/gelu_impl.h"
#include "core/providers/musa/musa_fwd.h"

namespace onnxruntime {
namespace musa {

Status MusaGelu::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* input = ctx->Input<Tensor>(0);
  ORT_RETURN_IF_NOT(input != nullptr, "MusaGelu input is null");
  ORT_RETURN_IF_NOT(input->IsDataType<float>(), "MusaGelu currently supports float only");
  Tensor* output = ctx->Output(0, input->Shape());
  ORT_RETURN_IF_NOT(output != nullptr, "MusaGelu output is null");
  const int64_t count = input->Shape().Size();
  if (count == 0) {
    return Status::OK();
  }
  musaError_t status = LaunchGeluFloat(Stream(ctx), input->Data<float>(), output->MutableData<float>(), count);
  if (status != musaSuccess) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "MusaGelu kernel launch failed, status=", static_cast<int>(status));
  }
  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    MusaGelu,
    kMSDomain,
    1,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create()),
    MusaGelu);

}  // namespace musa
}  // namespace onnxruntime
