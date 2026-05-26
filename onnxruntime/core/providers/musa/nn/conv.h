// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/musa/musa_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"

namespace onnxruntime {
namespace musa {

template <typename T, bool NHWC = false>
class Conv final : public MusaKernel {
 public:
  explicit Conv(const OpKernelInfo& info) : MusaKernel(info), conv_attrs_(info) {
    auto pads_size = conv_attrs_.pads.size();
    ORT_ENFORCE(pads_size % 2 == 0);
    is_nhwc_domain_ = info.node().Domain() == kMSInternalNHWCDomain;
  }

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 bool& is_packed, PrePackedWeights* prepacked_weights) override;

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  ConvAttributes conv_attrs_;
  
  // NHWC related members
  bool is_nhwc_domain_;         // prepack is only needed for the Conv in kMSInternalNHWCDomain
  std::unique_ptr<Tensor> W_;   // pre-packed NHWC weight
  bool W_already_nhwc = false;  // In case NHWC == true and Conv is not in kMSInternalNHWCDomain

  // Helper function to handle 1D convolution by converting to 2D
  Status ComputeConv1D(OpKernelContext* ctx, const Tensor* X, const Tensor* W, const Tensor* B,
                       bool x_is_nhwc, bool w_is_nhwc) const;
};

}  // namespace musa
}  // namespace onnxruntime
