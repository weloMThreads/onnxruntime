// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/musa_kernel.h"

namespace onnxruntime {
namespace musa {

template <typename T>
class GatherElements final : public MusaKernel {
 public:
  GatherElements(const OpKernelInfo& info) : MusaKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("axis", &axis_).IsOK(), "Missing/Invalid 'axis' attribute value");
  }

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  int64_t axis_;
};

}  // namespace musa
}  // namespace onnxruntime
