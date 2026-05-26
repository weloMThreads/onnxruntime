// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/musa/musa_kernel.h"

namespace onnxruntime {
namespace musa {

template <typename T>
class Sum final : public MusaKernel {
 public:
  Sum(const OpKernelInfo& info) : MusaKernel(info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  Status Prepare(OpKernelContext* ctx, MusaPreparation& prepare) const;
  Status ComputeBinarySum(OpKernelContext* ctx, MusaPreparation& prepare) const;
  Status ComputeVariadicSum(OpKernelContext* ctx, MusaPreparation& prepare) const;
};

}  // namespace musa
}  // namespace onnxruntime