// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/shape_op.h"

#include <algorithm>
#include <vector>

namespace onnxruntime {


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


ONNX_CPU_OPERATOR_KERNEL(
    BroadcastGradientArgs,
    1,
    KernelDefBuilder().TypeConstraint("T", BuildKernelDefConstraints<int32_t, int64_t>()),
    BroadcastGradientArgs);

ONNX_CPU_OPERATOR_KERNEL(
    ConcatOffset,
    1,
    KernelDefBuilder().TypeConstraint("T", BuildKernelDefConstraints<int32_t, int64_t>()),
    ConcatOffset);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Shape,
    1,
    12,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()).TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Shape);

ONNX_CPU_OPERATOR_KERNEL(
    ShapeN,
    1,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("T1", BuildKernelDefConstraints<int32_t, int64_t, MLFloat16, float, double>()),
    ShapeN);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Shape,
    13, 14,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()).TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Shape);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Shape,
    15, 18,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()).TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Shape);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Shape,
    19, 20,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()).TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Shape);

// Opset 21 added support for int4 and uint4.
// TODO(adrianlizarraga): Implement int4 and uint4 support.
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Shape,
    21,
    22,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypesIRv9()).TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Shape);

// Opset 23 added support for float4e2m1.
// TODO: Implement float4e2m1 support.
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Shape,
    23,
    23,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypesIRv9()).TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Shape);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Shape,
    24,
    24,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypesIRv9()).TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Shape);

// Opset 25
ONNX_CPU_OPERATOR_KERNEL(
    Shape,
    25,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypesIRv9()).TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Shape);

}  // namespace onnxruntime
