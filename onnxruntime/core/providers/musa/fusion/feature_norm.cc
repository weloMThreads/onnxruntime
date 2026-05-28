// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/fusion/feature_norm.h"
#include "core/providers/musa/fusion/feature_norm_impl.h"
#include "core/providers/musa/musa_fwd.h"

namespace onnxruntime {
namespace musa {

Status MusaFeatureNorm::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* input = ctx->Input<Tensor>(0);
  const Tensor* gamma = ctx->Input<Tensor>(1);
  const Tensor* beta = ctx->Input<Tensor>(2);
  ORT_RETURN_IF_NOT(input != nullptr && gamma != nullptr && beta != nullptr,
                    "MusaFeatureNorm inputs must not be null");
  ORT_RETURN_IF_NOT(input->IsDataType<float>() && gamma->IsDataType<float>() && beta->IsDataType<float>(),
                    "MusaFeatureNorm currently supports float tensors only");
  ORT_RETURN_IF_NOT(input->Shape().NumDimensions() >= 1,
                    "MusaFeatureNorm input rank must be >= 1");

  const int64_t cols = input->Shape()[input->Shape().NumDimensions() - 1];
  ORT_RETURN_IF_NOT(cols > 0, "MusaFeatureNorm last dimension must be positive");
  ORT_RETURN_IF_NOT(gamma->Shape().Size() == cols && beta->Shape().Size() == cols,
                    "MusaFeatureNorm gamma/beta length must equal input last dimension");

  Tensor* output = ctx->Output(0, input->Shape());
  ORT_RETURN_IF_NOT(output != nullptr, "MusaFeatureNorm output is null");
  if (output->Shape().Size() == 0) {
    return Status::OK();
  }

  const int64_t rows = input->Shape().Size() / cols;
  musaError_t status = LaunchFeatureNormFloat(Stream(ctx),
                                              input->Data<float>(),
                                              gamma->Data<float>(),
                                              beta->Data<float>(),
                                              output->MutableData<float>(),
                                              rows,
                                              cols,
                                              epsilon_);
  if (status != musaSuccess) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "MusaFeatureNorm kernel launch failed, status=", static_cast<int>(status));
  }
  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    MusaFeatureNorm,
    kMSDomain,
    1,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create()),
    MusaFeatureNorm);

}  // namespace musa
}  // namespace onnxruntime
