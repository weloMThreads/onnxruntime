// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/musa_kernel.h"
#include "core/providers/musa/musa_utils.h"

namespace onnxruntime {
namespace musa {

template <typename T>
class Reciprocal : public MusaKernel {
 public:
  explicit Reciprocal(const OpKernelInfo& info) : MusaKernel(info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;

 protected:
  Status Prepare(OpKernelContext* ctx, MusaPreparation& prepare) const;
};

}  // namespace musa
}  // namespace onnxruntime
