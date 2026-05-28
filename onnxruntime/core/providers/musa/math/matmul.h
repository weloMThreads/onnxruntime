// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/musa/musa_kernel.h"

namespace onnxruntime {
namespace musa {

template <typename T>
class MatMul final : public MusaKernel {
 public:
  MatMul(const OpKernelInfo& info) : MusaKernel(info) {
    // Get attributes with default values
    trans_A_ = info.GetAttrOrDefault<int64_t>("transA", 0) != 0;
    trans_B_ = info.GetAttrOrDefault<int64_t>("transB", 0) != 0;
    alpha_ = info.GetAttrOrDefault<float>("alpha", 1.0f);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  enum class ComputeStrategy {
    MU_BLAS_GEMM,   // mublas GEMM for compatibility
    MU_DNN_MATMUL,  // mudnn MatMul for performance
    MU_DNN_BATCH,   // mudnn BatchMatMul for batch operations
    MU_DNN_LOOP     // mudnn MatMul in loop for complex broadcasting (FP16)
  };

  Status PrepareMatMul(OpKernelContext* ctx, MusaPreparation& prepare,
                       const MatMulComputeHelper& helper) const;

  bool CanReshapeTo2D(const TensorShape& shape, size_t batch_count) const;

  bool CanReshape4DTo3D(const TensorShape& a_shape, const TensorShape& b_shape) const;

  Status SetupMusaTensorWithReshape(::musa::dnn::Tensor& musa_tensor,
                                    const Tensor* ort_tensor,
                                    ::musa::dnn::Tensor::Type data_type,
                                    MusaPreparation* preparation,
                                    int target_dims) const;
  ;

  ComputeStrategy SelectStrategy(const TensorShape& a_shape, const TensorShape& b_shape,
                                 MLDataType dtype, size_t batch_count,
                                 bool use_bf16_fast_math) const;

  Status ExecuteWithMuDNN(const MusaPreparation& prepare,
                          const MatMulComputeHelper& helper, bool use_batch,
                          onnxruntime::Stream* ort_stream) const;

  Status ExecuteWithMuBLAS(const MusaPreparation& prepare,
                           const MatMulComputeHelper& helper,
                           musaStream_t stream,
                           onnxruntime::Stream* ort_stream) const;

  Status ExecuteWithMuDNNLoop(OpKernelContext* ctx,
                              const MatMulComputeHelper& helper,
                              onnxruntime::Stream* ort_stream) const;

  bool trans_A_;
  bool trans_B_;
  float alpha_;
};

}  // namespace musa
}  // namespace onnxruntime