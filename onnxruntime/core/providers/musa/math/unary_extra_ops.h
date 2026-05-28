// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/musa/musa_kernel.h"

namespace onnxruntime {
namespace musa {

template <typename T>
class Reciprocal final : public MusaKernel {
 public:
  explicit Reciprocal(const OpKernelInfo& info) : MusaKernel(info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;
};

template <typename T>
class Log1p final : public MusaKernel {
 public:
  explicit Log1p(const OpKernelInfo& info) : MusaKernel(info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;
};

template <typename T>
class Expm1 final : public MusaKernel {
 public:
  explicit Expm1(const OpKernelInfo& info) : MusaKernel(info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;
};

template <typename T>
class Square final : public MusaKernel {
 public:
  explicit Square(const OpKernelInfo& info) : MusaKernel(info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;
};

template <typename T>
class Rsqrt final : public MusaKernel {
 public:
  explicit Rsqrt(const OpKernelInfo& info) : MusaKernel(info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;
};

template <typename T>
class Floor final : public MusaKernel {
 public:
  explicit Floor(const OpKernelInfo& info) : MusaKernel(info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;
};

template <typename T>
class Ceil final : public MusaKernel {
 public:
  explicit Ceil(const OpKernelInfo& info) : MusaKernel(info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;
};

template <typename T>
class ZerosLike final : public MusaKernel {
 public:
  explicit ZerosLike(const OpKernelInfo& info) : MusaKernel(info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;
};

template <typename T>
class Sign final : public MusaKernel {
 public:
  explicit Sign(const OpKernelInfo& info) : MusaKernel(info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;
};

template <typename T>
class IsNaN final : public MusaKernel {
 public:
  explicit IsNaN(const OpKernelInfo& info) : MusaKernel(info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;
};

}  // namespace musa
}  // namespace onnxruntime
