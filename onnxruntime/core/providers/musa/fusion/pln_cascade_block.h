// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>

#include "core/providers/musa/musa_kernel.h"

namespace onnxruntime {
namespace musa {

class MusaPlnCascadeBlock final : public MusaKernel {
 public:
  explicit MusaPlnCascadeBlock(const OpKernelInfo& info)
      : MusaKernel(info),
        num_steps_(info.GetAttrOrDefault<int64_t>("num_steps", 0)) {
    ORT_ENFORCE(num_steps_ > 0 && num_steps_ <= 16,
                "MusaPlnCascadeBlock requires num_steps in [1, 16], got ",
                num_steps_);
  }

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  int64_t num_steps_;
};

}  // namespace musa
}  // namespace onnxruntime
