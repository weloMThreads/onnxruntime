// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/musa/musa_kernel.h"

namespace onnxruntime {
namespace musa {

class MusaLayerNormLastDim final : public MusaKernel {
 public:
  explicit MusaLayerNormLastDim(const OpKernelInfo& info)
      : MusaKernel(info),
        clip_min_(info.GetAttrOrDefault<float>("clip_min", 0.0f)),
        clip_max_(info.GetAttrOrDefault<float>("clip_max", 1.0e30f)) {}

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  float clip_min_;
  float clip_max_;
};

}  // namespace musa
}  // namespace onnxruntime
