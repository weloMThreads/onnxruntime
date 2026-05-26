// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/musa_kernel.h"
#include "core/providers/musa/musa_utils.h"
#include "gsl/gsl"

namespace onnxruntime {
namespace musa {

template <typename in_type, typename out_type, typename depth_type>
class OneHot final : public MusaKernel {
 public:
  OneHot(const OpKernelInfo& info) : MusaKernel(info) {
    int64_t tmp_axis;
    if (info.GetAttr<int64_t>("axis", &tmp_axis).IsOK()) {
      axis_ = tmp_axis;
    }
  }

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  Status Prepare(OpKernelContext* ctx, MusaPreparation& prepare) const;
  
  int64_t axis_ = -1;
};

}  // namespace musa
}  // namespace onnxruntime