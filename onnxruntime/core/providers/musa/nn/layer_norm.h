// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/musa_kernel.h"
#include "core/providers/musa/musa_utils.h"

namespace onnxruntime {
namespace musa {

// NOTE: This was originally a contrib op with 3 type constraints. The ONNX spec merges 'T' and 'V'.
// the kernel is templatized on all three for backwards compatibility, but in ONNX usage T == V.
template <typename T, typename U, typename V, bool simplified>
class LayerNorm final : public MusaKernel {
 public:
  LayerNorm(const OpKernelInfo& op_kernel_info);

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  int64_t axis_;
  double epsilon_;
};

}  // namespace musa
}  // namespace onnxruntime
