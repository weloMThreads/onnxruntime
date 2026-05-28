// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/musa/musa_kernel.h"

#include <string>


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

template <typename T> class BiasAdd final : public MusaKernel {
public:
  BiasAdd(const OpKernelInfo &info) : MusaKernel(info) {
    std::string data_format;
    if (info.GetAttr<std::string>("data_format", &data_format).IsOK()) {
      data_format_ = data_format;
    }
    ORT_ENFORCE(data_format_ == "NHWC" || data_format_ == "NCHW",
                "BiasAdd only supports NHWC and NCHW data_format, got ", data_format_);
  }
  Status ComputeInternal(OpKernelContext *ctx) const override;

private:
  std::string data_format_{"NHWC"};
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

template <typename T> class DivNoNan final : public BinaryElementwise {
public:
  DivNoNan(const OpKernelInfo &info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext *ctx) const override;
};

template <typename T> class SquaredDifference final : public BinaryElementwise {
public:
  SquaredDifference(const OpKernelInfo &info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext *ctx) const override;
};

template <typename T> class FloorDiv final : public BinaryElementwise {
public:
  FloorDiv(const OpKernelInfo &info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext *ctx) const override;
};

template <typename T> class FloorMod final : public BinaryElementwise {
public:
  FloorMod(const OpKernelInfo &info) : BinaryElementwise(info) {}
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
