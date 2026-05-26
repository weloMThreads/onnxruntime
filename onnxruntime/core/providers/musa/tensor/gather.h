// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cpu/tensor/gatherbase.h"
#include "core/providers/musa/musa_kernel.h"

namespace onnxruntime {
namespace musa {

template <typename T>
class Gather final : public MusaKernel, public GatherBase {
 public:
  Gather(const OpKernelInfo& info) : MusaKernel(info), GatherBase(info) {
  }

  Status ComputeInternal(OpKernelContext* ctx) const override;
};

}  // namespace musa
}  // namespace onnxruntime
