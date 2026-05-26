// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/math/gemm_helper.h"
#include "core/providers/musa/musa_kernel.h"

namespace onnxruntime {
namespace musa {

template <typename T>
class Gemm final : public MusaKernel {
 public:
  Gemm(const OpKernelInfo& info) : MusaKernel(info) {
    int64_t temp;
    ORT_ENFORCE(info.GetAttr<int64_t>("transA", &temp).IsOK());
    trans_A_ = (temp != 0);

    ORT_ENFORCE(info.GetAttr<int64_t>("transB", &temp).IsOK());
    trans_B_ = (temp != 0);

    ORT_ENFORCE(info.GetAttr<float>("alpha", &alpha_).IsOK());
    ORT_ENFORCE(info.GetAttr<float>("beta", &beta_).IsOK());
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  Status Prepare(OpKernelContext* ctx, MusaPreparation& prepare, int M, int N, int K) const;

  bool trans_A_;
  bool trans_B_;
  float alpha_;
  float beta_;
};

}  // namespace musa
}  // namespace onnxruntime
