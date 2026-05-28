// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <mutex>
#include <random>

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/musa_kernel.h"

namespace onnxruntime {
namespace musa {

class RandomUniformLike final : public MusaKernel {
 public:
  explicit RandomUniformLike(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  float high_;
  float low_;
  mutable std::default_random_engine generator_;
  mutable std::mutex generator_mutex_;
  ONNX_NAMESPACE::TensorProto::DataType dtype_ = ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
};

}  // namespace musa
}  // namespace onnxruntime
