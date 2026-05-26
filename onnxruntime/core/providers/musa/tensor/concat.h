// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cpu/tensor/concatbase.h"
#include "core/providers/musa/musa_kernel.h"
#include "core/providers/musa/musa_utils.h"
#include "gsl/gsl"

namespace onnxruntime {
namespace musa {

template <typename T>
class Concat final : public MusaKernel, public ConcatBase {
 public:
  Concat(const OpKernelInfo& info) : MusaKernel(info), ConcatBase(info) {
  }

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  Status Prepare(OpKernelContext* ctx, MusaPreparation& prepare, struct Prepare& p) const;
};

}  // namespace musa
}  // namespace onnxruntime