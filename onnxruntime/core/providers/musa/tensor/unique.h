// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/musa/musa_kernel.h"

namespace onnxruntime {
namespace musa {

template <typename T>
class Unique final : public MusaKernel {
 public:
  explicit Unique(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  bool sorted_{true};
  bool flatten_{false};
  int64_t axis_{0};
};

}  // namespace musa
}  // namespace onnxruntime
