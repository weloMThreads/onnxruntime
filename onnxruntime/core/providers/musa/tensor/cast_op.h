// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/musa_kernel.h"

namespace onnxruntime {
namespace musa {

template <typename SrcT>
class Cast final : public MusaKernel {
 public:
  Cast(const OpKernelInfo& info) : MusaKernel(info) {
    int64_t to;
    Status status = info.GetAttr("to", &to);
    ORT_ENFORCE(status.IsOK(), "Attribute 'to' is not set.");
    to_ = gsl::narrow_cast<ONNX_NAMESPACE::TensorProto_DataType>(to);

    int64_t saturate = info.GetAttrOrDefault("saturate", int64_t{1});
    saturate_ = saturate == 1;
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  Status Prepare(OpKernelContext* ctx, MusaPreparation& prepare) const;
  
  ONNX_NAMESPACE::TensorProto_DataType to_;
  bool saturate_;
};

}  // namespace musa
}  // namespace onnxruntime