// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/math/gemm_helper.h"
#include "core/providers/musa/musa_kernel.h"
#include <string>

namespace onnxruntime {
namespace musa {

template <typename T>
class Gemm : public MusaKernel {
 public:
  Gemm(const OpKernelInfo& info) : MusaKernel(info) {
    trans_A_ = info.GetAttrOrDefault<int64_t>("transA", 0) != 0;
    trans_B_ = info.GetAttrOrDefault<int64_t>("transB", 0) != 0;
    alpha_ = info.GetAttrOrDefault<float>("alpha", 1.0f);
    beta_ = info.GetAttrOrDefault<float>("beta", 1.0f);
    activation_ = info.GetAttrOrDefault<std::string>("activation", "");
    activation_alpha_ = info.GetAttrOrDefault<float>("activation_alpha", 0.01f);
    activation_beta_ = info.GetAttrOrDefault<float>("activation_beta", 0.0f);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  Status Prepare(OpKernelContext* ctx, MusaPreparation& prepare, int M, int N, int K) const;
  Status ApplyActivationInPlace(MusaPreparation& prepare) const;

  bool trans_A_;
  bool trans_B_;
  float alpha_;
  float beta_;
  std::string activation_;
  float activation_alpha_;
  float activation_beta_;
};

}  // namespace musa
}  // namespace onnxruntime
