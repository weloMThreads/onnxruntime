// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/musa_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace musa {

template <typename T>
class QuickGelu final : public onnxruntime::musa::MusaKernel {
 public:
  explicit QuickGelu(const OpKernelInfo& info) : MusaKernel(info) {
    alpha_ = info.GetAttrOrDefault<float>("alpha", 1.702f);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float alpha_;
};

}  // namespace musa
}  // namespace contrib
}  // namespace onnxruntime
