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

template <typename T>
class GatherV2 final : public MusaKernel {
 public:
  explicit GatherV2(const OpKernelInfo& info) : MusaKernel(info) {
    info.GetAttrOrDefault<int64_t>("batch_dims", &batch_dims_, 0);
  }

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  int64_t batch_dims_ = 0;
};

}  // namespace musa
}  // namespace onnxruntime
