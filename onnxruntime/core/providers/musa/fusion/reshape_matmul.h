// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>

#include "core/providers/musa/musa_kernel.h"

namespace onnxruntime {
namespace musa {

class MusaReshapeMatMul final : public MusaKernel {
 public:
  explicit MusaReshapeMatMul(const OpKernelInfo& info)
      : MusaKernel(info),
        transpose_b_(info.GetAttrOrDefault<int64_t>("transpose_b", 0) != 0) {}

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  bool transpose_b_;
};

}  // namespace musa
}  // namespace onnxruntime
