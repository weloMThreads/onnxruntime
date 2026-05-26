// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/musa_kernel.h"
#include "core/providers/cpu/tensor/squeeze.h"
#include "core/providers/musa/musa_utils.h"

namespace onnxruntime {
namespace musa {

template <typename T>
class Squeeze final : public MusaKernel, public SqueezeBase {
 public:
  Squeeze(const OpKernelInfo& info) : MusaKernel(info), SqueezeBase(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  Status Prepare(OpKernelContext* ctx, MusaPreparation& prepare) const;
};

}  // namespace musa
}  // namespace onnxruntime
