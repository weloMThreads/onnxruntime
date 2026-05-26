// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/musa_kernel.h"
#include "core/providers/musa/musa_utils.h"

namespace onnxruntime {
namespace musa {

template <typename T>
class Clip_6 final : public MusaKernel {
 public:
  explicit Clip_6(const OpKernelInfo& info) : MusaKernel{info} {
    // Get min and max attributes with default values
    constexpr auto min_default = std::numeric_limits<T>::lowest();
    constexpr auto max_default = std::numeric_limits<T>::max();
    
    // Handle attribute retrieval safely via float conversion
    float min_float, max_float;
    auto min_status = info.GetAttr("min", &min_float);
    auto max_status = info.GetAttr("max", &max_float);
    
    min_ = min_status.IsOK() ? static_cast<T>(min_float) : min_default;
    max_ = max_status.IsOK() ? static_cast<T>(max_float) : max_default;
    
    ORT_ENFORCE(min_ <= max_, "min value must be less than or equal to max value");
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 protected:
  Status Prepare(OpKernelContext* ctx, MusaPreparation& prepare) const;
  T min_;
  T max_;
};

// Since version 11. Min and Max are inputs
// version 12 adds type support
class Clip final : public MusaKernel {
 public:
  explicit Clip(const OpKernelInfo& info) : MusaKernel{info} {
  }

  Status ComputeInternal(OpKernelContext* context) const override;

  template <typename T>
  Status Prepare(OpKernelContext* ctx, MusaPreparation& prepare) const;

 private:
  template <typename T>
  struct ComputeImpl;
};

}  // namespace musa
}  // namespace onnxruntime