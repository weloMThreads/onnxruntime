// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/musa_kernel.h"
#include "core/providers/musa/musa_utils.h"

namespace onnxruntime {
namespace musa {

template <typename T>
class Sqrt : public MusaKernel {
 public:
  explicit Sqrt(const OpKernelInfo& info) : MusaKernel(info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;

 protected:
  Status Prepare(OpKernelContext* ctx, MusaPreparation& prepare) const;
};

}  // namespace musa
}  // namespace onnxruntime
