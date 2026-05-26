// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/musa_kernel.h"
#include "core/providers/cpu/tensor/upsamplebase.h"
#include "core/providers/musa/tensor/resize_impl.h"

namespace onnxruntime {
namespace musa {

template <typename T>
class Resize : public UpsampleBase, public MusaKernel {
 public:
  explicit Resize(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  Status BaseCompute(OpKernelContext* context, gsl::span<const float> roi, gsl::span<const float> scales,
                     gsl::span<const int64_t> output_dims) const;
};

}  // namespace musa
}  // namespace onnxruntime
