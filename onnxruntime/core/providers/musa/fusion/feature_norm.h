// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/musa/musa_kernel.h"

namespace onnxruntime {
namespace musa {

class MusaFeatureNorm final : public MusaKernel {
 public:
  explicit MusaFeatureNorm(const OpKernelInfo& info)
      : MusaKernel(info),
        epsilon_(info.GetAttrOrDefault<float>("epsilon", 1e-3f)) {}

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  float epsilon_;
};

}  // namespace musa
}  // namespace onnxruntime
