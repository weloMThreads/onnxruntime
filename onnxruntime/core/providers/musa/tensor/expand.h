// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/musa_kernel.h"
#include "core/providers/musa/musa_utils.h"

namespace onnxruntime {
namespace musa {

template <typename T>
class Expand final : public MusaKernel {
 public:
  Expand(const OpKernelInfo& info) : MusaKernel(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  Status Prepare(OpKernelContext* ctx, MusaPreparation& prepare) const;
  Status InferOutputShape(const Tensor* input, const Tensor* shape,
                         TensorShape& output_shape) const;
  Status ComputeBroadcastShape(const TensorShape& input_shape,
                              const TensorShapeVector& target_shape_vec,
                              TensorShape& output_shape) const;
  Status ValidateBroadcast(const TensorShape& input_shape, 
                          const TensorShape& target_shape) const;
};

}  // namespace musa
}  // namespace onnxruntime