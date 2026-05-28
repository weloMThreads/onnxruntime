// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cpu/tensor/shape_op.h"
#include "core/providers/musa/musa_fwd.h"
#include <algorithm>
#include <gsl/gsl>
#include <limits>
#include <type_traits>
#include <vector>
#include <cstdio>

namespace onnxruntime {
namespace musa {


namespace {

template <typename T>
Status ReadShapeVector(const Tensor& tensor, const char* name, std::vector<int64_t>& shape) {
  ORT_RETURN_IF_NOT(tensor.Shape().NumDimensions() == 1, name, " must be a vector.");
  const int64_t count = tensor.Shape().Size();
  const T* data = tensor.Data<T>();
  shape.resize(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    const int64_t dim = static_cast<int64_t>(data[i]);
    ORT_RETURN_IF_NOT(dim >= 0, name, " contains negative dimension ", dim, ".");
    shape[static_cast<size_t>(i)] = dim;
  }
  return Status::OK();
}

template <typename T>
Status ComputeBroadcastGradientArgsTyped(const Tensor& s0_tensor, const Tensor& s1_tensor,
                                         Tensor& r0_tensor, Tensor& r1_tensor) {
  std::vector<int64_t> s0;
  std::vector<int64_t> s1;
  ORT_RETURN_IF_ERROR(ReadShapeVector<T>(s0_tensor, "BroadcastGradientArgs input 0", s0));
  ORT_RETURN_IF_ERROR(ReadShapeVector<T>(s1_tensor, "BroadcastGradientArgs input 1", s1));

  const size_t rank0 = s0.size();
  const size_t rank1 = s1.size();
  const size_t rank = std::max(rank0, rank1);
  const size_t pad0 = rank - rank0;
  const size_t pad1 = rank - rank1;
  std::vector<int64_t> r0;
  std::vector<int64_t> r1;

  for (size_t axis = 0; axis < rank; ++axis) {
    const bool has0 = axis >= pad0;
    const bool has1 = axis >= pad1;
    const int64_t dim0 = has0 ? s0[axis - pad0] : 1;
    const int64_t dim1 = has1 ? s1[axis - pad1] : 1;

    if (!has0) {
      r0.push_back(static_cast<int64_t>(axis));
    }
    if (!has1) {
      r1.push_back(static_cast<int64_t>(axis));
    }
    if (has0 && has1) {
      if (dim0 == dim1) {
        continue;
      }
      if (dim0 == 1) {
        r0.push_back(static_cast<int64_t>(axis));
      } else if (dim1 == 1) {
        r1.push_back(static_cast<int64_t>(axis));
      } else {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "BroadcastGradientArgs incompatible shapes.");
      }
    }
  }

  T* r0_data = r0_tensor.MutableData<T>();
  T* r1_data = r1_tensor.MutableData<T>();
  for (size_t i = 0; i < r0.size(); ++i) {
    r0_data[i] = static_cast<T>(r0[i]);
  }
  for (size_t i = 0; i < r1.size(); ++i) {
    r1_data[i] = static_cast<T>(r1[i]);
  }
  return Status::OK();
}

class BroadcastGradientArgs final : public OpKernel {
 public:
  explicit BroadcastGradientArgs(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* ctx) const override {
    const Tensor* s0 = ctx->Input<Tensor>(0);
    const Tensor* s1 = ctx->Input<Tensor>(1);
    ORT_RETURN_IF_NOT(s0 != nullptr && s1 != nullptr, "BroadcastGradientArgs inputs must not be null.");

    std::vector<int64_t> s0_shape;
    std::vector<int64_t> s1_shape;
    if (s0->IsDataType<int32_t>()) {
      ORT_RETURN_IF_ERROR(ReadShapeVector<int32_t>(*s0, "BroadcastGradientArgs input 0", s0_shape));
      ORT_RETURN_IF_ERROR(ReadShapeVector<int32_t>(*s1, "BroadcastGradientArgs input 1", s1_shape));
    } else if (s0->IsDataType<int64_t>()) {
      ORT_RETURN_IF_ERROR(ReadShapeVector<int64_t>(*s0, "BroadcastGradientArgs input 0", s0_shape));
      ORT_RETURN_IF_ERROR(ReadShapeVector<int64_t>(*s1, "BroadcastGradientArgs input 1", s1_shape));
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "BroadcastGradientArgs inputs must be int32 or int64.");
    }

    const size_t rank = std::max(s0_shape.size(), s1_shape.size());
    const size_t pad0 = rank - s0_shape.size();
    const size_t pad1 = rank - s1_shape.size();
    size_t r0_count = 0;
    size_t r1_count = 0;
    for (size_t axis = 0; axis < rank; ++axis) {
      const bool has0 = axis >= pad0;
      const bool has1 = axis >= pad1;
      const int64_t dim0 = has0 ? s0_shape[axis - pad0] : 1;
      const int64_t dim1 = has1 ? s1_shape[axis - pad1] : 1;
      if (!has0) {
        ++r0_count;
      }
      if (!has1) {
        ++r1_count;
      }
      if (has0 && has1 && dim0 != dim1) {
        if (dim0 == 1) {
          ++r0_count;
        } else if (dim1 == 1) {
          ++r1_count;
        } else {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "BroadcastGradientArgs incompatible shapes.");
        }
      }
    }

    Tensor* r0 = ctx->Output(0, TensorShape{static_cast<int64_t>(r0_count)});
    Tensor* r1 = ctx->Output(1, TensorShape{static_cast<int64_t>(r1_count)});
    ORT_RETURN_IF_NOT(r0 != nullptr && r1 != nullptr, "BroadcastGradientArgs outputs must not be null.");

    if (s0->IsDataType<int32_t>()) {
      return ComputeBroadcastGradientArgsTyped<int32_t>(*s0, *s1, *r0, *r1);
    }
    return ComputeBroadcastGradientArgsTyped<int64_t>(*s0, *s1, *r0, *r1);
  }
};

template <typename T>
Status ComputeConcatOffsetTyped(OpKernelContext* ctx) {
  const Tensor* concat_dim_tensor = ctx->Input<Tensor>(0);
  ORT_RETURN_IF_NOT(concat_dim_tensor != nullptr, "ConcatOffset concat_dim input is null.");
  ORT_RETURN_IF_NOT(concat_dim_tensor->Shape().NumDimensions() == 0,
                    "ConcatOffset concat_dim must be a scalar.");

  const int input_count = ctx->InputCount();
  const int output_count = ctx->OutputCount();
  ORT_RETURN_IF_NOT(input_count >= 2, "ConcatOffset requires at least one shape input.");
  ORT_RETURN_IF_NOT(output_count == input_count - 1,
                    "ConcatOffset output count must match shape input count.");

  const Tensor* first_shape_tensor = ctx->Input<Tensor>(1);
  ORT_RETURN_IF_NOT(first_shape_tensor != nullptr, "ConcatOffset first shape input is null.");
  std::vector<int64_t> first_shape;
  ORT_RETURN_IF_ERROR(ReadShapeVector<T>(*first_shape_tensor, "ConcatOffset shape input", first_shape));
  const int64_t rank = static_cast<int64_t>(first_shape.size());
  ORT_RETURN_IF_NOT(rank > 0, "ConcatOffset shape rank must be greater than zero.");

  int64_t concat_dim = static_cast<int64_t>(concat_dim_tensor->Data<T>()[0]);
  if (concat_dim < 0) {
    concat_dim += rank;
  }
  ORT_RETURN_IF_NOT(concat_dim >= 0 && concat_dim < rank,
                    "ConcatOffset concat_dim is out of range.");

  std::vector<int64_t> offset(static_cast<size_t>(rank), 0);
  for (int i = 0; i < output_count; ++i) {
    const Tensor* shape_tensor = ctx->Input<Tensor>(i + 1);
    ORT_RETURN_IF_NOT(shape_tensor != nullptr, "ConcatOffset shape input is null.");
    std::vector<int64_t> shape;
    ORT_RETURN_IF_ERROR(ReadShapeVector<T>(*shape_tensor, "ConcatOffset shape input", shape));
    ORT_RETURN_IF_NOT(shape.size() == static_cast<size_t>(rank),
                      "ConcatOffset all shape inputs must have the same rank.");

    Tensor* output = ctx->Output(i, TensorShape{rank});
    ORT_RETURN_IF_NOT(output != nullptr, "ConcatOffset output tensor is null.");
    T* output_data = output->MutableData<T>();
    for (int64_t axis = 0; axis < rank; ++axis) {
      output_data[static_cast<size_t>(axis)] = static_cast<T>(offset[static_cast<size_t>(axis)]);
      if (axis == concat_dim) {
        offset[static_cast<size_t>(axis)] += shape[static_cast<size_t>(axis)];
      } else {
        ORT_RETURN_IF_NOT(shape[static_cast<size_t>(axis)] == first_shape[static_cast<size_t>(axis)],
                          "ConcatOffset input shapes must match except along concat_dim.");
      }
    }
  }

  return Status::OK();
}

class ConcatOffset final : public OpKernel {
 public:
  explicit ConcatOffset(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* ctx) const override {
    const Tensor* concat_dim = ctx->Input<Tensor>(0);
    ORT_RETURN_IF_NOT(concat_dim != nullptr, "ConcatOffset concat_dim input is null.");
    if (concat_dim->IsDataType<int32_t>()) {
      return ComputeConcatOffsetTyped<int32_t>(ctx);
    }
    if (concat_dim->IsDataType<int64_t>()) {
      return ComputeConcatOffsetTyped<int64_t>(ctx);
    }
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "ConcatOffset inputs must be int32 or int64.");
  }
};

}  // namespace

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

class MusaShapeN final : public OpKernel {
 public:
  explicit MusaShapeN(const OpKernelInfo& info) : OpKernel(info) {
    info.GetAttrOrDefault<int64_t>("out_type", &out_type_, ONNX_NAMESPACE::TensorProto_DataType_INT32);
  }

  Status Compute(OpKernelContext* context) const override {
    const size_t input_count = static_cast<size_t>(context->InputCount());
    const size_t output_count = static_cast<size_t>(context->OutputCount());
    ORT_RETURN_IF_NOT(input_count == output_count,
                      "ShapeN: input and output counts must match");

    for (size_t i = 0; i < input_count; ++i) {
      const Tensor* input = context->Input<Tensor>(static_cast<int>(i));
      ORT_RETURN_IF_NOT(input != nullptr, "ShapeN: input tensor is null");

      const TensorShape& input_shape = input->Shape();
      const int64_t rank = gsl::narrow_cast<int64_t>(input_shape.NumDimensions());
      switch (out_type_) {
        case ONNX_NAMESPACE::TensorProto_DataType_INT32:
          FillOutput<int32_t>(context, static_cast<int>(i), input_shape, rank);
          break;
        case ONNX_NAMESPACE::TensorProto_DataType_INT64:
          FillOutput<int64_t>(context, static_cast<int>(i), input_shape, rank);
          break;
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
          FillOutput<float>(context, static_cast<int>(i), input_shape, rank);
          break;
        case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
          FillOutput<double>(context, static_cast<int>(i), input_shape, rank);
          break;
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
          FillOutput<MLFloat16>(context, static_cast<int>(i), input_shape, rank);
          break;
        default:
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "ShapeN: unsupported out_type ", out_type_);
      }
    }

    return Status::OK();
  }

 private:
  template <typename T>
  static void FillOutput(OpKernelContext* context, int output_index,
                         const TensorShape& input_shape, int64_t rank) {
    Tensor* output = context->Output(output_index, TensorShape{rank});
    T* output_data = output->MutableData<T>();
    for (int64_t j = 0; j < rank; ++j) {
      output_data[j] = ConvertDim<T>(input_shape.GetDims()[onnxruntime::narrow<size_t>(j)]);
    }
  }

  template <typename T>
  static T ConvertDim(int64_t dim) {
    if constexpr (std::is_same<T, MLFloat16>::value) {
      return MLFloat16(static_cast<float>(dim));
    } else {
      return static_cast<T>(dim);
    }
  }

  int64_t out_type_;
};


ONNX_OPERATOR_KERNEL_EX(
    BroadcastGradientArgs,
    kOnnxDomain,
    1,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 0)
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .OutputMemoryType(OrtMemTypeCPUInput, 0)
        .OutputMemoryType(OrtMemTypeCPUInput, 1)
        .TypeConstraint("T", BuildKernelDefConstraints<int32_t, int64_t>()),
    BroadcastGradientArgs);

ONNX_OPERATOR_KERNEL_EX(
    ConcatOffset,
    kOnnxDomain,
    1,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 0)
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .OutputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", BuildKernelDefConstraints<int32_t, int64_t>()),
    ConcatOffset);

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

ONNX_OPERATOR_KERNEL_EX(
    ShapeN,
    kOnnxDomain,
    1,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create())
        .OutputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("T1", BuildKernelDefConstraints<int32_t, int64_t, MLFloat16, float, double>()),
    MusaShapeN);

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