// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/musa_kernel.h"
#include "core/providers/musa/musa_utils.h"

namespace onnxruntime {
namespace musa {

template <typename T>
class Activations : public MusaKernel {
 public:
  explicit Activations(const OpKernelInfo& info) : MusaKernel(info) {}

 protected:
  Status Prepare(OpKernelContext* ctx, MusaPreparation& prepare) const;
};

template <typename T>
class Relu : public Activations<T> {
 public:
  explicit Relu(const OpKernelInfo& info) : Activations<T>(info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;
};

template <typename T>
class Tanh : public Activations<T> {
 public:
  explicit Tanh(const OpKernelInfo& info) : Activations<T>(info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;
};

template <typename T>
class Sigmoid : public Activations<T> {
 public:
  explicit Sigmoid(const OpKernelInfo& info) : Activations<T>(info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;
};

template <typename T>
class LeakyRelu : public Activations<T> {
 public:
  explicit LeakyRelu(const OpKernelInfo& info) : Activations<T>(info), alpha_(0.01f) {
    ORT_ENFORCE(info.GetAttr("alpha", &alpha_).IsOK());
  }
  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  float alpha_;
};

template <typename T>
class Log : public Activations<T> {
 public:
  explicit Log(const OpKernelInfo& info) : Activations<T>(info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;
};

template <typename T>
class Softplus : public Activations<T> {
 public:
  explicit Softplus(const OpKernelInfo& info) : Activations<T>(info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;
};

}  // namespace musa
}  // namespace onnxruntime
