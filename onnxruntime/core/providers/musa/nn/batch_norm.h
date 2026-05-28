// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/musa/musa_kernel.h"

namespace onnxruntime {
namespace musa {

template <typename T>
class BatchNormalization final : public MusaKernel {
 public:
  explicit BatchNormalization(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  float epsilon_;
  bool is_train_;
  int output_count_;
};

}  // namespace musa
}  // namespace onnxruntime
