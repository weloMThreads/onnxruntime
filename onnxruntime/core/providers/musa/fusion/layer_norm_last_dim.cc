// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/fusion/layer_norm_last_dim.h"
#include "core/providers/musa/fusion/layer_norm_last_dim_impl.h"
#include "core/providers/musa/musa_fwd.h"

#include <limits>

namespace onnxruntime {
namespace musa {

Status MusaLayerNormLastDim::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* input = ctx->Input<Tensor>(0);
  const Tensor* gamma = ctx->Input<Tensor>(1);
  const Tensor* beta = ctx->Input<Tensor>(2);
  ORT_RETURN_IF_NOT(input != nullptr && gamma != nullptr && beta != nullptr,
                    "MusaLayerNormLastDim inputs must not be null");
  ORT_RETURN_IF_NOT(input->IsDataType<float>() && gamma->IsDataType<float>() && beta->IsDataType<float>(),
                    "MusaLayerNormLastDim supports float inputs only");
  ORT_RETURN_IF_NOT(input->Shape().NumDimensions() == 2,
                    "MusaLayerNormLastDim currently supports rank-2 inputs only, got ",
                    input->Shape().ToString());

  const int64_t rows64 = input->Shape()[0];
  const int64_t cols64 = input->Shape()[1];
  ORT_RETURN_IF_NOT(rows64 <= std::numeric_limits<int>::max() &&
                        cols64 <= std::numeric_limits<int>::max(),
                    "MusaLayerNormLastDim input shape is too large: ",
                    input->Shape().ToString());

  const int64_t gamma_count64 = gamma->Shape().Size();
  const int64_t beta_count64 = beta->Shape().Size();
  ORT_RETURN_IF_NOT((gamma_count64 == 1 || gamma_count64 == cols64) &&
                        (beta_count64 == 1 || beta_count64 == cols64),
                    "MusaLayerNormLastDim gamma/beta must be scalar or match last dim. input=",
                    input->Shape().ToString(), ", gamma=", gamma->Shape().ToString(),
                    ", beta=", beta->Shape().ToString());

  Tensor* output = ctx->Output(0, input->Shape());
  ORT_RETURN_IF_NOT(output != nullptr, "MusaLayerNormLastDim output is null");
  if (input->Shape().Size() == 0) {
    return Status::OK();
  }

  const musaError_t status = LaunchLayerNormLastDimFloat(Stream(ctx),
                                                         input->Data<float>(),
                                                         gamma->Data<float>(),
                                                         beta->Data<float>(),
                                                         output->MutableData<float>(),
                                                         static_cast<int>(rows64),
                                                         static_cast<int>(cols64),
                                                         static_cast<int>(gamma_count64),
                                                         static_cast<int>(beta_count64),
                                                         clip_min_,
                                                         clip_max_);
  if (status != musaSuccess) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "MusaLayerNormLastDim kernel launch failed, status=",
                           static_cast<int>(status));
  }

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    MusaLayerNormLastDim,
    kMSDomain,
    1,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create()),
    MusaLayerNormLastDim);

}  // namespace musa
}  // namespace onnxruntime
