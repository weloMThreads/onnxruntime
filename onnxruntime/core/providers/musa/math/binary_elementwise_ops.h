// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/musa/musa_kernel.h"


namespace onnxruntime {
namespace musa {

class BinaryElementwise : public MusaKernel {
protected:
  explicit BinaryElementwise(const OpKernelInfo &info) : MusaKernel(info) {}
  Status ComputeInternal(OpKernelContext *) const override {
    return Status(common::ONNXRUNTIME, common::FAIL); // should not reach here
  }
  template <typename T>
  Status Prepare(OpKernelContext *ctx, MusaPreparation &prepare) const;
};

template <typename T> class Add final : public BinaryElementwise {
public:
  Add(const OpKernelInfo &info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext *ctx) const override;
};

template <typename T> class Sub final : public BinaryElementwise {
public:
  Sub(const OpKernelInfo &info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext *ctx) const override;
};

template <typename T> class Mul final : public BinaryElementwise {
public:
  Mul(const OpKernelInfo &info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext *ctx) const override;
};

template <typename T> class Div final : public BinaryElementwise {
public:
  Div(const OpKernelInfo &info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext *ctx) const override;
};

template <typename T> class Pow final : public BinaryElementwise {
public:
  Pow(const OpKernelInfo &info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext *ctx) const override;
};

template <typename T> class Min final : public BinaryElementwise {
public:
  Min(const OpKernelInfo &info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext *ctx) const override;
};

template <typename T> class Max final : public BinaryElementwise {
public:
  Max(const OpKernelInfo &info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext *ctx) const override;
};

template <typename T> class PRelu final : public BinaryElementwise {
public:
  PRelu(const OpKernelInfo &info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext *ctx) const override;
};

} // namespace musa
} // namespace onnxruntime
