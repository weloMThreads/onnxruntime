// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/common/common.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/musa_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace musa {

template <typename T, bool IsNHWC>
class GridSample final : public onnxruntime::musa::MusaKernel {
 public:
  explicit GridSample(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t mode_i_;
  int64_t padding_mode_i_;
  int64_t align_corners_;
};

}  // namespace musa
}  // namespace contrib
}  // namespace onnxruntime
