// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/musa_kernel.h"

namespace onnxruntime {
namespace musa {

class Flatten final : public MusaKernel {
 public:
  Flatten(const OpKernelInfo& info) : MusaKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("axis", &axis_).IsOK());
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t axis_;
};

}  // namespace musa
}  // namespace onnxruntime
