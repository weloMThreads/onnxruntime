// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/fusion/token_mixer_residual.h"
#include "core/providers/musa/fusion/token_mixer_residual_impl.h"
#include "core/providers/musa/musa_call.h"
#include "core/providers/musa/musa_fwd.h"

#include <type_traits>

namespace onnxruntime {
namespace musa {

Status MusaTokenMixerResidual::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* input = ctx->Input<Tensor>(0);
  ORT_RETURN_IF_NOT(input != nullptr, "MusaTokenMixerResidual input is null");
  ORT_RETURN_IF_NOT(num_t_ == num_h_,
                    "MusaTokenMixerResidual currently requires num_T == num_H for residual Add shape parity");

  const int64_t per_batch = num_t_ * num_h_ * d_k_;
  ORT_RETURN_IF_NOT(per_batch > 0, "MusaTokenMixerResidual invalid shape attributes");
  const int64_t input_size = input->Shape().Size();
  ORT_RETURN_IF_NOT(input_size % per_batch == 0,
                    "MusaTokenMixerResidual input size ", input_size,
                    " is not divisible by num_T*num_H*d_k=", per_batch);

  const int64_t batch = input_size / per_batch;
  TensorShape output_shape({batch, num_h_, num_t_ * d_k_});
  Tensor* output = ctx->Output(0, output_shape);
  ORT_RETURN_IF_NOT(output != nullptr, "MusaTokenMixerResidual output is null");

  if (input_size == 0) {
    return Status::OK();
  }

  musaError_t status = musaSuccess;
  if (input->IsDataType<float>()) {
    status = LaunchTokenMixerResidualFloat(Stream(ctx),
                                           input->Data<float>(),
                                           output->MutableData<float>(),
                                           batch,
                                           num_t_,
                                           num_h_,
                                           d_k_);
  } else if (input->IsDataType<MLFloat16>()) {
    status = LaunchTokenMixerResidualHalf(Stream(ctx),
                                          input->DataRaw(),
                                          output->MutableDataRaw(),
                                          batch,
                                          num_t_,
                                          num_h_,
                                          d_k_);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "MusaTokenMixerResidual only supports float and float16 inputs");
  }

  if (status != musaSuccess) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "MusaTokenMixerResidual kernel launch failed, status: ",
                           static_cast<int>(status));
  }

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    MusaTokenMixerResidual,
    kMSDomain,
    1,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create()),
    MusaTokenMixerResidual);

}  // namespace musa
}  // namespace onnxruntime
