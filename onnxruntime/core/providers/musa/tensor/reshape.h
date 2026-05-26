// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cpu/tensor/reshape_helper.h"
#include "core/providers/musa/musa_kernel.h"
#include "core/providers/musa/musa_utils.h"

namespace onnxruntime {
namespace musa {

class Reshape final : public MusaKernel {
 public:
  Reshape(const OpKernelInfo& info) : MusaKernel(info),
      allow_zero_(info.GetAttrOrDefault<int64_t>("allowzero", 0) == 1) {}

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  bool allow_zero_;
  
  Status InferOutputShape(const Tensor* data, const Tensor* shape, 
                         TensorShape& output_shape) const;
  bool CanReshapeInPlace(const Tensor* input, const TensorShape& target_shape) const;
  Status ExecuteMusaReshape(OpKernelContext* ctx, const Tensor* input, 
                           Tensor* output, const TensorShape& target_shape) const;
};

// Support ONNX v1-4 version Reshape_1
class Reshape_1 final : public MusaKernel {
 public:
  Reshape_1(const OpKernelInfo& info) : MusaKernel(info) {
    Status status = info.GetAttrs("shape", shape_);
    ORT_ENFORCE(status.IsOK(), "Attribute shape is not set.");
  }
  
  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  TensorShapeVector shape_;
};

}  // namespace musa
}  // namespace onnxruntime