// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/musa/musa_kernel.h"

namespace onnxruntime {
namespace musa {

class CumSum final : public MusaKernel {
 public:
  explicit CumSum(const OpKernelInfo& info) : MusaKernel(info) {
    int64_t exclusive = 0;
    auto status = info.GetAttr("exclusive", &exclusive);
    if (status.IsOK()) {
      ORT_ENFORCE(exclusive == 0 || exclusive == 1, "attribute exclusive can only be 0 or 1");
      exclusive_ = (exclusive == 1);
    }

    int64_t reverse = 0;
    status = info.GetAttr("reverse", &reverse);
    if (status.IsOK()) {
      ORT_ENFORCE(reverse == 0 || reverse == 1, "attribute reverse can only be 0 or 1");
      reverse_ = (reverse == 1);
    }
  }

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  bool exclusive_ = false;
  bool reverse_ = false;
};

}  // namespace musa
}  // namespace onnxruntime
