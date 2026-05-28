// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/musa/musa_kernel.h"

namespace onnxruntime {
namespace musa {

template <typename T>
class TopK final : public MusaKernel {
 public:
  explicit TopK(const OpKernelInfo& info)
      : MusaKernel(info),
        axis_(info.GetAttrOrDefault<int64_t>("axis", -1)),
        largest_(info.GetAttrOrDefault<int64_t>("largest", 1) != 0),
        sorted_(info.GetAttrOrDefault<int64_t>("sorted", 1) != 0) {}

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  int64_t axis_;
  bool largest_;
  bool sorted_;
};

}  // namespace musa
}  // namespace onnxruntime
