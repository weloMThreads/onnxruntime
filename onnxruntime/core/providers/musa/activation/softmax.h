// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/musa_kernel.h"
#include "core/providers/musa/musa_utils.h"

namespace onnxruntime {
namespace musa {

template <typename T>
class Softmax final : public MusaKernel {
 public:
  explicit Softmax(const OpKernelInfo& info) : MusaKernel(info) {
    // Get axis attribute with proper opset version handling
    int64_t axis;
    Status status = info.GetAttr<int64_t>("axis", &axis);
    if (status.IsOK()) {
      axis_ = gsl::narrow_cast<int>(axis);
    } else {
      // Handle default axis based on opset version
      if (info.node().SinceVersion() < 13) {
        axis_ = 1;  // opset-12 and below default axis
      } else {
        axis_ = -1; // opset-13+ default axis  
      }
    }
    
    // Determine if this is LogSoftmax
    log_softmax_ = info.GetKernelDef().OpName() == "LogSoftmax";
    opset_version_ = info.node().SinceVersion();
  }

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  Status Prepare(OpKernelContext* ctx, MusaPreparation& prepare) const;
  
  int axis_;
  bool log_softmax_;
  int opset_version_;
};

}  // namespace musa
}  // namespace onnxruntime