// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/musa/musa_kernel.h"
#include "core/framework/allocator.h"

namespace onnxruntime {
namespace musa {

template <typename T>
class LSTM final : public MusaKernel {
 public:
  explicit LSTM(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  // Helper methods
  Status ValidateInputs(const Tensor& X,
                        const Tensor& W,
                        const Tensor& R,
                        const Tensor* B,
                        const Tensor* sequence_lens,
                        const Tensor* initial_h,
                        const Tensor* initial_c,
                        const Tensor* P) const;

  Status PrepareWeights(OpKernelContext* ctx,
                        const Tensor& W,
                        const Tensor& R,
                        const Tensor* B,
                        IAllocatorUniquePtr<T>& W_reordered,
                        IAllocatorUniquePtr<T>& R_reordered,
                        IAllocatorUniquePtr<T>& B_reordered) const;

  Status ComputeLSTM(OpKernelContext* ctx,
                     const Tensor& X,
                     const T* W_reordered,
                     const T* R_reordered,
                     const T* B_reordered,
                     const Tensor* sequence_lens,
                     const Tensor* initial_h,
                     const Tensor* initial_c) const;

  // Attributes
  std::string direction_;
  int num_directions_;
  int hidden_size_;
  float clip_;
  bool input_forget_;
  int64_t layout_;
  std::vector<std::string> activation_func_names_;
  std::vector<float> activation_func_alphas_;
  std::vector<float> activation_func_betas_;
};

}  // namespace musa
}  // namespace onnxruntime