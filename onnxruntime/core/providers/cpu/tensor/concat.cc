// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/concat.h"

#include "core/framework/element_type_lists.h"
#include "core/framework/TensorSeq.h"
#include "core/framework/copy.h"
#include "core/providers/common.h"
#include "core/providers/op_kernel_type_control.h"

#include <cstring>

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Concat,
    4,
    10,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Concat);

// Opset 11 starts to support Neg Axis.
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Concat,
    11,
    12,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Concat);

// Opset 13 .
ONNX_CPU_OPERATOR_KERNEL(
    Concat,
    13,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Concat);

namespace {

Status ReadIntScalar(const Tensor& tensor, int64_t& value) {
  ORT_RETURN_IF_NOT(tensor.Shape().Size() == 1, "Expected a scalar tensor.");
  if (tensor.IsDataType<int32_t>()) {
    value = static_cast<int64_t>(*tensor.Data<int32_t>());
    return Status::OK();
  }
  if (tensor.IsDataType<int64_t>()) {
    value = *tensor.Data<int64_t>();
    return Status::OK();
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Expected int32 or int64 scalar tensor.");
}

Status PrepareConcatV2(OpKernelContext* ctx, int64_t& axis, TensorShapeVector& output_dims,
                       std::vector<const Tensor*>& inputs) {
  const int input_count = ctx->InputCount();
  ORT_RETURN_IF_NOT(input_count >= 2, "ConcatV2 requires at least one value input and one axis input.");

  const Tensor* axis_tensor = ctx->Input<Tensor>(input_count - 1);
  ORT_RETURN_IF_NOT(axis_tensor != nullptr, "ConcatV2 axis input is null.");
  ORT_RETURN_IF_ERROR(ReadIntScalar(*axis_tensor, axis));

  const int value_count = input_count - 1;
  inputs.reserve(static_cast<size_t>(value_count));
  const Tensor* first = ctx->Input<Tensor>(0);
  ORT_RETURN_IF_NOT(first != nullptr, "ConcatV2 first value input is null.");
  ORT_RETURN_IF_NOT(first->Shape().NumDimensions() > 0, "ConcatV2 cannot concatenate scalar tensors.");
  const auto rank = first->Shape().NumDimensions();
  axis = HandleNegativeAxis(axis, gsl::narrow_cast<int64_t>(rank));
  output_dims = first->Shape().AsShapeVector();
  output_dims[onnxruntime::narrow<size_t>(axis)] = 0;

  for (int i = 0; i < value_count; ++i) {
    const Tensor* input = ctx->Input<Tensor>(i);
    ORT_RETURN_IF_NOT(input != nullptr, "ConcatV2 value input is null.");
    ORT_RETURN_IF_NOT(input->DataType() == first->DataType(), "ConcatV2 value input data types must match.");
    ORT_RETURN_IF_NOT(input->Shape().NumDimensions() == rank, "ConcatV2 value input ranks must match.");
    const auto& dims = input->Shape().GetDims();
    for (size_t d = 0; d < rank; ++d) {
      if (d == static_cast<size_t>(axis)) {
        continue;
      }
      ORT_RETURN_IF_NOT(dims[d] == output_dims[d], "ConcatV2 non-concat dimensions must match.");
    }
    output_dims[onnxruntime::narrow<size_t>(axis)] += dims[onnxruntime::narrow<size_t>(axis)];
    inputs.push_back(input);
  }

  return Status::OK();
}

}  // namespace

class ConcatV2 final : public OpKernel {
 public:
  explicit ConcatV2(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* ctx) const override {
    int64_t axis = 0;
    TensorShapeVector output_dims;
    std::vector<const Tensor*> inputs;
    ORT_RETURN_IF_ERROR(PrepareConcatV2(ctx, axis, output_dims, inputs));

    Tensor* output = ctx->Output(0, TensorShape{output_dims});
    ORT_RETURN_IF_NOT(output != nullptr, "ConcatV2 output tensor is null.");
    if (output->Shape().Size() == 0) {
      return Status::OK();
    }

    const size_t element_size = inputs.front()->DataType()->Size();
    const size_t axis_index = onnxruntime::narrow<size_t>(axis);
    const int64_t outer = inputs.front()->Shape().SizeToDimension(axis_index);
    const int64_t inner = (axis_index + 1 == output_dims.size())
                              ? 1
                              : inputs.front()->Shape().SizeFromDimension(axis_index + 1);
    const int64_t output_axis = output_dims[axis_index];
    auto* dst_base = static_cast<uint8_t*>(output->MutableDataRaw());

    for (int64_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
      int64_t dst_axis_offset = 0;
      for (const Tensor* input : inputs) {
        const int64_t input_axis = input->Shape()[axis_index];
        const size_t bytes = onnxruntime::narrow<size_t>(input_axis * inner) * element_size;
        if (bytes > 0) {
          const auto* src = static_cast<const uint8_t*>(input->DataRaw()) +
                            onnxruntime::narrow<size_t>(outer_idx * input_axis * inner) * element_size;
          auto* dst = dst_base +
                      onnxruntime::narrow<size_t>((outer_idx * output_axis + dst_axis_offset) * inner) * element_size;
          std::memcpy(dst, src, bytes);
        }
        dst_axis_offset += input_axis;
      }
    }

    return Status::OK();
  }
};

ONNX_CPU_OPERATOR_KERNEL(
    ConcatV2,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    ConcatV2);

namespace op_kernel_type_control {
// we're using one set of types for all opsets
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Concat, Input, 0,
    element_type_lists::All);

// Concat can be used with dimensions or indices so require int32_t and int64_t to be supported
ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Concat, Input, 0, int32_t, int64_t);
}  // namespace op_kernel_type_control

namespace {
using EnabledDataTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(kCpuExecutionProvider, kOnnxDomain,
                                                                        Concat, Input, 0);
}  // namespace

namespace {
TensorShapeVector StridesForStack(const TensorShapeVector& full_strides, uint64_t axis) {
  // if we are stacking, skip the dimension that will be stacked along in the output strides
  // (the striding for that dimension is handled by the initial_output_offset)
  const auto num_dims = full_strides.size();

  TensorShapeVector strides;
  strides.reserve(num_dims - 1);

  for (size_t i = 0; i < num_dims - 1; i++) {
    auto read_i = (i >= axis) ? i + 1 : i;
    strides.push_back(full_strides[read_i]);
  }
  return strides;
}
}  // namespace

// This method computes the output tensor for Concat/ConcatFromSequence ops
Status ConcatBase::ComputeImpl(Prepare& p, OpKernelContext* ctx) const {
  int input_count = static_cast<int>(p.inputs.size());
  int64_t initial_output_offset = 0;  // initial offset for each input

  auto output_strides_full = StridesForTensor(*p.output_tensor);
  // Note that output_strides_full is only used later when is_stack_ is true, so it's safe to move
  auto output_strides_for_copy = is_stack_ ? StridesForStack(output_strides_full, p.axis) : std::move(output_strides_full);

  for (int input_index = 0; input_index < input_count; input_index++) {
    const auto& prep = p.inputs[input_index];

    // no data in this tensor - so skip it
    if (prep.num_elements == 0)
      continue;

    // parallel copy the data across
    auto status = DispatchStridedCopy<EnabledDataTypes>(ctx->GetOperatorThreadPool(),
                                                        *p.output_tensor,
                                                        onnxruntime::narrow<ptrdiff_t>(initial_output_offset),
                                                        output_strides_for_copy,
                                                        prep.tensor->Shape(),
                                                        *prep.tensor,
                                                        0,  // src_offset
                                                        StridesForTensor(*prep.tensor));
    ORT_RETURN_IF_ERROR(status);

    // advance along the axis that we are concatenating on (by the size of the axis of the tensor that we just copied)
    if (is_stack_) {
      initial_output_offset += output_strides_full[onnxruntime::narrow<size_t>(p.axis)];
    } else {
      initial_output_offset += prep.tensor->Shape()[onnxruntime::narrow<size_t>(p.axis)] * output_strides_for_copy[onnxruntime::narrow<size_t>(p.axis)];
    }
  }

  return Status::OK();
}

// core Compute() method for the 'Concat' kernel
Status Concat::Compute(OpKernelContext* ctx) const {
  // Number of input tensors to concatenate
  auto input_count = Node().InputArgCount().front();

  // Hold pointers to the input tensors to be used in the PrepareForCompute() step
  InlinedTensorsVector input_tensors;
  input_tensors.reserve(input_count);
  for (int i = 0; i < input_count; ++i) {
    input_tensors.push_back(ctx->Input<Tensor>(i));
  }

  // Validate inputs and prepare some metadata used during actual compute
  Prepare p;
  auto status = PrepareForCompute(ctx, input_tensors, p);
  if (!status.IsOK())
    return status;

  // Return at this point if output tensor is going to be empty
  if (p.output_num_elements == 0)
    return Status::OK();

  // Compute values to be placed in the output tensor
  return ComputeImpl(p, ctx);
}

}  // namespace onnxruntime
