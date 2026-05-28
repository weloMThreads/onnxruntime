// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/musa_call.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/musa_kernel.h"

namespace onnxruntime {
namespace musa {

template <bool is_dropout>
class IdentityOp final : public MusaKernel {
 public:
  explicit IdentityOp(const OpKernelInfo& info) : MusaKernel(info) {}

  Status ComputeInternal(OpKernelContext* context) const override {
    const Tensor* input = context->Input<Tensor>(0);
    ORT_RETURN_IF_NOT(input != nullptr, "Identity: input tensor is null");

    Tensor* output = context->Output(0, input->Shape());
    ORT_RETURN_IF_NOT(output != nullptr, "Identity: output tensor is null");

    const size_t bytes = input->SizeInBytes();
    const void* source = input->DataRaw();
    void* target = output->MutableDataRaw();
    if (bytes > 0 && source != target) {
      MUSA_RETURN_IF_ERROR(musaMemcpyAsync(target, source, bytes, musaMemcpyDeviceToDevice, Stream(context)));
    }

    if constexpr (is_dropout) {
      Tensor* mask = context->Output(1, input->Shape());
      if (mask != nullptr && mask->SizeInBytes() > 0) {
        MUSA_RETURN_IF_ERROR(musaMemsetAsync(mask->MutableDataRaw(), 0, mask->SizeInBytes(), Stream(context)));
      }
    }

    return Status::OK();
  }
};

class IdentityNOp final : public MusaKernel {
 public:
  explicit IdentityNOp(const OpKernelInfo& info) : MusaKernel(info) {}

  Status ComputeInternal(OpKernelContext* context) const override {
    const size_t input_count = static_cast<size_t>(context->InputCount());
    const size_t output_count = static_cast<size_t>(context->OutputCount());
    ORT_RETURN_IF_NOT(input_count == output_count,
                      "IdentityN: input and output counts must match");

    for (size_t i = 0; i < input_count; ++i) {
      const Tensor* input = context->Input<Tensor>(static_cast<int>(i));
      ORT_RETURN_IF_NOT(input != nullptr, "IdentityN: input tensor is null");

      Tensor* output = context->Output(static_cast<int>(i), input->Shape());
      ORT_RETURN_IF_NOT(output != nullptr, "IdentityN: output tensor is null");

      const size_t bytes = input->SizeInBytes();
      const void* source = input->DataRaw();
      void* target = output->MutableDataRaw();
      if (bytes > 0 && source != target) {
        MUSA_RETURN_IF_ERROR(musaMemcpyAsync(target, source, bytes, musaMemcpyDeviceToDevice, Stream(context)));
      }
    }

    return Status::OK();
  }
};

}  // namespace musa
}  // namespace onnxruntime
