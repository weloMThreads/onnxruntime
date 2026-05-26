// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cpu/tensor/shape_op.h"
#include "core/providers/musa/musa_fwd.h"
#include <gsl/gsl>
#include <limits>
#include <cstdio>

namespace onnxruntime {
namespace musa {

// MusaShape 实现：采用CPU重定向策略的Shape算子
class MusaShape final : public OpKernel {
 public:
  MusaShape(const OpKernelInfo& info) : OpKernel(info) {
    info.GetAttrOrDefault<int64_t>("start", &start_index_, 0);

    if (start_index_ != 0) {
      // "start" is provided and is non-default (default is 0)
      needs_slicing_ = true;
    }

    if (info.GetAttr<int64_t>("end", &end_index_).IsOK()) {
      needs_slicing_ = true;
    }
  }

  Status Compute(OpKernelContext* context) const override {
    const auto* input = context->Input<Tensor>(0);
    const TensorShape& input_shape = input->Shape();
    
    int64_t rank = gsl::narrow_cast<int64_t>(input_shape.NumDimensions());

    if (!needs_slicing_) {  // vanilla use of Shape (no slicing)
      Tensor* output = context->Output(0, {rank});
      input_shape.CopyDims(output->MutableData<int64_t>(), static_cast<size_t>(rank));
    } else {  // slicing is needed
      int64_t true_start = start_index_;
      int64_t true_end = end_index_;

      // Deal with negative(s) and clamp
      true_start = true_start < 0 ? true_start + rank : true_start;
      true_start = true_start < 0 ? 0 : ((true_start > rank) ? rank : true_start);

      true_end = true_end < 0 ? true_end + rank : true_end;
      true_end = true_end < 0 ? 0 : ((true_end > rank) ? rank : true_end);

      auto slice_length = true_end - true_start;
      Tensor* output = context->Output(0, {slice_length < 0 ? 0 : slice_length});

      if (slice_length > 0) {
        input_shape.CopyDims(output->MutableData<int64_t>(), onnxruntime::narrow<size_t>(true_start), onnxruntime::narrow<size_t>(slice_length));
      }
    }
    
    return Status::OK();
  }

 private:
  bool needs_slicing_ = false;
  int64_t start_index_ = 0;
  int64_t end_index_ = std::numeric_limits<int64_t>::max();
};

// ONNX v1-v12: 基础版本
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Shape,
    kOnnxDomain,
    1, 12,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create())
        .OutputMemoryType(OrtMemTypeCPUInput, 0)  // 强制输出到CPU
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    MusaShape);

// ONNX v13-v14: 增加start/end切片支持
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Shape,
    kOnnxDomain,
    13, 14,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create())
        .OutputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    MusaShape);

// ONNX v15-v18: 扩展数据类型
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Shape,
    kOnnxDomain,
    15, 18,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create())
        .OutputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    MusaShape);

// ONNX v19-v20: IRv9数据类型支持
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Shape,
    kOnnxDomain,
    19, 20,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create())
        .OutputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypesIRv9())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    MusaShape);

// ONNX v21-v22: 更多数据类型支持（当前限制与CPU EP一致）
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Shape,
    kOnnxDomain,
    21, 22,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create())
        .OutputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypesIRv9())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    MusaShape);

// ONNX v23: 最新版本支持
ONNX_OPERATOR_KERNEL_EX(
    Shape,
    kOnnxDomain,
    23,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create())
        .OutputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypesIRv9())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    MusaShape);

}  // namespace musa
}  // namespace onnxruntime