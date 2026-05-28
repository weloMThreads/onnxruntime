// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/musa/musa_kernel.h"

namespace onnxruntime {
namespace musa {

class ScatterND final : public MusaKernel {
 public:
  explicit ScatterND(const OpKernelInfo& info) : MusaKernel(info) {
    std::string reduction;
    if (info.GetAttr<std::string>("reduction", &reduction).IsOK() && reduction != "none") {
      ORT_THROW("MUSA ScatterND currently supports only reduction='none'. Actual: ", reduction);
    }
  }

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  Status ValidateIndicesOnCpu(const TensorShape& input_shape,
                              const TensorShape& indices_shape,
                              const Tensor* indices_tensor) const;
};

}  // namespace musa
}  // namespace onnxruntime
