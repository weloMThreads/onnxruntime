// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/musa/musa_kernel.h"

namespace onnxruntime {
namespace musa {

class GatherNDBase : public MusaKernel {
 public:
  explicit GatherNDBase(const OpKernelInfo& info) : MusaKernel(info) {
    info.GetAttrOrDefault("batch_dims", &batch_dims_, static_cast<int64_t>(0));
    ORT_ENFORCE(batch_dims_ >= 0, "GatherND batch_dims must be non-negative");
  }

 protected:
  Status ValidateInputShapes(const TensorShape& input_shape,
                             const TensorShape& indices_shape) const;

  template <typename TIndex>
  Status ValidateIndicesOnCpu(const TensorShape& input_shape,
                              const TensorShape& indices_shape,
                              const Tensor* indices_tensor) const;

  int64_t batch_dims_ = 0;
};

template <typename TIndex>
class GatherND final : public GatherNDBase {
 public:
  explicit GatherND(const OpKernelInfo& info) : GatherNDBase(info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;
};

}  // namespace musa
}  // namespace onnxruntime
