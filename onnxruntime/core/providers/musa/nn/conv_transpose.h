// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/providers/musa/musa_kernel.h"
#include "core/providers/cpu/nn/conv_transpose_attributes.h"

namespace onnxruntime {
namespace musa {

template <typename T, bool NHWC = false>
class ConvTranspose final : public MusaKernel {
 public:
  explicit ConvTranspose(const OpKernelInfo& info)
      : MusaKernel(info), conv_transpose_attrs_(info) {
    is_nhwc_domain_ = info.node().Domain() == kMSInternalNHWCDomain;
  }

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 bool& is_packed, PrePackedWeights* prepacked_weights) override;

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  ConvTransposeAttributes conv_transpose_attrs_;
  bool is_nhwc_domain_ = false;
  std::unique_ptr<Tensor> W_;
  bool W_already_nhwc = false;
};

}  // namespace musa
}  // namespace onnxruntime
