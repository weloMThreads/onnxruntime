// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/musa_kernel.h"
#include "core/providers/musa/musa_utils.h"

namespace onnxruntime {
namespace musa {

template <typename T>
class Pad : public MusaKernel {
 public:
  explicit Pad(const OpKernelInfo& info) : MusaKernel(info) {
    std::string mode = "constant";  // default mode
    if (info.GetAttr("mode", &mode).IsOK()) {
      // mode provided as attribute
    }
    
    if (mode == "constant") {
      mode_ = 0;
    } else if (mode == "reflect") {
      mode_ = 1;
    } else if (mode == "edge") {
      mode_ = 2;
    } else if (mode == "wrap") {
      mode_ = 3;
    } else {
      ORT_THROW("Pad mode '", mode, "' is not supported");
    }

    // Get value attribute (constant mode only)
    float value = 0.0f;
    if (info.GetAttr("value", &value).IsOK()) {
      value_ = static_cast<T>(value);
    } else {
      value_ = T{}; // use default constructor instead of static_cast
    }

    // Get pads attribute if present (for older versions)
    if (info.GetAttrs("pads", pads_).IsOK()) {
      // pads provided as attribute
      is_dynamic_ = false;
    } else {
      // pads will come from input tensor (v11+)
      is_dynamic_ = true;
    }
  }

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  int mode_;            // 0=constant, 1=reflect, 2=edge, 3=wrap
  T value_;             // padding value for constant mode
  std::vector<int64_t> pads_;  // padding values for static case
  bool is_dynamic_;     // whether pads come from input tensor

  Status ExtractDynamicParams(OpKernelContext* ctx, 
                              std::vector<int64_t>& pads,
                              T& pad_value) const;

  Status ComputePadding(OpKernelContext* ctx,
                        const Tensor& input_tensor,
                        Tensor& output_tensor,
                        const std::vector<int64_t>& pads,
                        T pad_value) const;

  Status ComputePadding2DWorkaround(OpKernelContext* ctx,
                                   const Tensor& input_tensor,
                                   Tensor& output_tensor,
                                   const std::vector<int64_t>& temp_input_dims,
                                   const std::vector<int64_t>& temp_output_dims,
                                   const std::vector<int64_t>& temp_pads,
                                   T pad_value) const;

  Status ValidatePadding(const TensorShape& input_shape,
                         const std::vector<int64_t>& pads) const;

  std::vector<int64_t> ComputeOutputShape(const TensorShape& input_shape,
                                          const std::vector<int64_t>& pads) const;
};

}  // namespace musa
}  // namespace onnxruntime