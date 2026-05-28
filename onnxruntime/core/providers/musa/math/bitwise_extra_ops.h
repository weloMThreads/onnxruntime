// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/musa/musa_kernel.h"

namespace onnxruntime {
namespace musa {

template <typename T>
class BitwiseOr final : public MusaKernel {
 public:
  explicit BitwiseOr(const OpKernelInfo& info) : MusaKernel(info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;
};

template <typename T>
class BitwiseXor final : public MusaKernel {
 public:
  explicit BitwiseXor(const OpKernelInfo& info) : MusaKernel(info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;
};

template <typename T>
class BitwiseNot final : public MusaKernel {
 public:
  explicit BitwiseNot(const OpKernelInfo& info) : MusaKernel(info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;
};

}  // namespace musa
}  // namespace onnxruntime
