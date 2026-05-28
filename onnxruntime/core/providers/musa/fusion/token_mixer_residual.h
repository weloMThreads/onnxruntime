// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>

#include "core/providers/musa/musa_kernel.h"

namespace onnxruntime {
namespace musa {

class MusaTokenMixerResidual final : public MusaKernel {
 public:
  explicit MusaTokenMixerResidual(const OpKernelInfo& info)
      : MusaKernel(info),
        num_t_(info.GetAttrOrDefault<int64_t>("num_T", 0)),
        num_h_(info.GetAttrOrDefault<int64_t>("num_H", 0)),
        d_k_(info.GetAttrOrDefault<int64_t>("d_k", 0)) {
    ORT_ENFORCE(num_t_ > 0 && num_h_ > 0 && d_k_ > 0,
                "MusaTokenMixerResidual requires positive num_T, num_H and d_k");
  }

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  int64_t num_t_;
  int64_t num_h_;
  int64_t d_k_;
};

}  // namespace musa
}  // namespace onnxruntime
