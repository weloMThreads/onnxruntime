// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/musa/musa_kernel.h"

namespace onnxruntime {
namespace musa {

template <typename T>
class GreaterOrEqual final : public MusaKernel {
public:
  GreaterOrEqual(const OpKernelInfo& info) : MusaKernel(info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;

private:
  template <typename U>
  Status Prepare(OpKernelContext* ctx, MusaPreparation& prepare) const;
};

}  // namespace musa
}  // namespace onnxruntime