// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/split.h"

#include "core/common/narrow.h"
#include <gsl/gsl>
#include "core/common/safeint.h"
#include "core/framework/copy.h"
#include "core/framework/element_type_lists.h"
#include "core/framework/op_kernel_type_control_utils.h"
#include "core/providers/common.h"
#include "core/providers/op_kernel_type_control.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

#include <cstring>

namespace onnxruntime {

namespace op_kernel_type_control {
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Split, Input, 0,
    element_type_lists::All);
ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Split, Input, 0,
    int32_t, int64_t);
}  // namespace op_kernel_type_control

using EnabledSplitDataTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Split, Input, 0);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Split,
    2,
    10,
    KernelDefBuilder().TypeConstraint("T",
                                      BuildKernelDefConstraintsFromTypeList<EnabledSplitDataTypes>()),
    Split_1_13);

// Opset 11 starts to support Neg Axis.
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Split,
    11,
    12,
    KernelDefBuilder().TypeConstraint("T",
                                      BuildKernelDefConstraintsFromTypeList<EnabledSplitDataTypes>()),
    Split_1_13);

// Opset 13 starts to supports 'split' as optional input.
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Split,
    13,
    17,
    KernelDefBuilder().TypeConstraint("T",
                                      BuildKernelDefConstraintsFromTypeList<EnabledSplitDataTypes>()),
    Split_1_13);

// TODO: support unequal split and num_outputs
ONNX_CPU_OPERATOR_KERNEL(
    Split,
    18,
    KernelDefBuilder().TypeConstraint("T",
                                      BuildKernelDefConstraintsFromTypeList<EnabledSplitDataTypes>()),
    Split_18);

namespace {

Status ReadSplitVScalar(const Tensor& tensor, int64_t& value) {
  ORT_RETURN_IF_NOT(tensor.Shape().Size() == 1, "SplitV split_dim must be a scalar tensor.");
  if (tensor.IsDataType<int32_t>()) {
    value = static_cast<int64_t>(*tensor.Data<int32_t>());
    return Status::OK();
  }
  if (tensor.IsDataType<int64_t>()) {
    value = *tensor.Data<int64_t>();
    return Status::OK();
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "SplitV split_dim must be int32 or int64.");
}

Status ReadSplitSizes(const Tensor& tensor, std::vector<int64_t>& split_sizes) {
  ORT_RETURN_IF_NOT(tensor.Shape().NumDimensions() == 1, "SplitV size_splits must be a 1D tensor.");
  const size_t count = onnxruntime::narrow<size_t>(tensor.Shape()[0]);
  split_sizes.resize(count);
  if (tensor.IsDataType<int32_t>()) {
    const int32_t* data = tensor.Data<int32_t>();
    for (size_t i = 0; i < count; ++i) split_sizes[i] = static_cast<int64_t>(data[i]);
    return Status::OK();
  }
  if (tensor.IsDataType<int64_t>()) {
    const int64_t* data = tensor.Data<int64_t>();
    std::copy(data, data + count, split_sizes.begin());
    return Status::OK();
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "SplitV size_splits must be int32 or int64.");
}

Status PrepareSplitV(OpKernelContext* ctx, const Tensor*& input, int64_t& axis,
                     std::vector<int64_t>& split_sizes) {
  input = ctx->Input<Tensor>(0);
  const Tensor* split_tensor = ctx->Input<Tensor>(1);
  const Tensor* axis_tensor = ctx->Input<Tensor>(2);
  ORT_RETURN_IF_NOT(input != nullptr && split_tensor != nullptr && axis_tensor != nullptr,
                    "SplitV inputs must not be null.");
  ORT_RETURN_IF_NOT(input->Shape().NumDimensions() > 0, "SplitV cannot split scalar tensors.");

  ORT_RETURN_IF_ERROR(ReadSplitSizes(*split_tensor, split_sizes));
  ORT_RETURN_IF_NOT(split_sizes.size() == static_cast<size_t>(ctx->OutputCount()),
                    "SplitV size_splits length must match output count.");
  ORT_RETURN_IF_ERROR(ReadSplitVScalar(*axis_tensor, axis));
  axis = HandleNegativeAxis(axis, gsl::narrow_cast<int64_t>(input->Shape().NumDimensions()));

  const int64_t input_axis = input->Shape()[onnxruntime::narrow<size_t>(axis)];
  int neg_one_index = -1;
  int64_t known_sum = 0;
  for (size_t i = 0; i < split_sizes.size(); ++i) {
    if (split_sizes[i] == -1) {
      ORT_RETURN_IF_NOT(neg_one_index == -1, "SplitV allows at most one -1 split size.");
      neg_one_index = onnxruntime::narrow<int>(i);
    } else {
      ORT_RETURN_IF_NOT(split_sizes[i] >= 0, "SplitV split sizes must be non-negative or -1.");
      known_sum += split_sizes[i];
    }
  }
  if (neg_one_index >= 0) {
    split_sizes[onnxruntime::narrow<size_t>(neg_one_index)] = input_axis - known_sum;
    ORT_RETURN_IF_NOT(split_sizes[onnxruntime::narrow<size_t>(neg_one_index)] >= 0,
                      "SplitV inferred split size must be non-negative.");
  }
  const int64_t total = std::accumulate(split_sizes.begin(), split_sizes.end(), int64_t{0});
  ORT_RETURN_IF_NOT(total == input_axis, "SplitV split sizes must sum to the selected input dimension.");
  return Status::OK();
}

}  // namespace

class SplitV final : public OpKernel {
 public:
  explicit SplitV(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* ctx) const override {
    const Tensor* input = nullptr;
    int64_t axis = 0;
    std::vector<int64_t> split_sizes;
    ORT_RETURN_IF_ERROR(PrepareSplitV(ctx, input, axis, split_sizes));

    TensorShapeVector output_dims = input->Shape().AsShapeVector();
    const size_t axis_index = onnxruntime::narrow<size_t>(axis);
    std::vector<Tensor*> outputs(split_sizes.size());
    for (size_t i = 0; i < split_sizes.size(); ++i) {
      output_dims[axis_index] = split_sizes[i];
      outputs[i] = ctx->Output(static_cast<int>(i), TensorShape{output_dims});
    }

    if (input->Shape().Size() == 0) {
      return Status::OK();
    }

    const size_t element_size = input->DataType()->Size();
    const int64_t outer = input->Shape().SizeToDimension(axis_index);
    const int64_t inner = (axis_index + 1 == input->Shape().NumDimensions())
                              ? 1
                              : input->Shape().SizeFromDimension(axis_index + 1);
    const int64_t input_axis = input->Shape()[axis_index];
    const auto* src_base = static_cast<const uint8_t*>(input->DataRaw());

    for (int64_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
      int64_t src_axis_offset = 0;
      for (size_t i = 0; i < split_sizes.size(); ++i) {
        const int64_t split = split_sizes[i];
        const size_t bytes = onnxruntime::narrow<size_t>(split * inner) * element_size;
        if (bytes > 0) {
          const auto* src = src_base +
                            onnxruntime::narrow<size_t>((outer_idx * input_axis + src_axis_offset) * inner) * element_size;
          auto* dst = static_cast<uint8_t*>(outputs[i]->MutableDataRaw()) +
                      onnxruntime::narrow<size_t>(outer_idx * split * inner) * element_size;
          std::memcpy(dst, src, bytes);
        }
        src_axis_offset += split;
      }
    }

    return Status::OK();
  }
};

ONNX_CPU_OPERATOR_KERNEL(
    SplitV,
    1,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("Tlen", BuildKernelDefConstraints<int32_t, int64_t>())
        .TypeConstraint("Taxis", BuildKernelDefConstraints<int32_t, int64_t>()),
    SplitV);

Status SplitImpl::Compute(OpKernelContext* context) const {
  const Tensor& input = *context->Input<Tensor>(0);
  auto& input_shape = input.Shape();
  auto num_outputs = context->OutputCount();
  int64_t axis = axis_;
  int before_dims = 0;
  int after_dims_including_split_axis = 0;
  int after_dims_excluding_split = 0;
  std::vector<int64_t> split_sizes;

  const Tensor* split_tensor = context->Input<Tensor>(1);
  if (split_tensor != nullptr) {
    // override the attribute value with the input value for split
    ORT_ENFORCE(split_tensor->Shape().NumDimensions() == 1, "The split tensor must be a vector tensor.");
    auto nDims = static_cast<size_t>(split_tensor->Shape()[0]);
    const auto* data = split_tensor->Data<int64_t>();
    split_sizes.assign(data, data + nDims);
  } else {
    split_sizes.assign(split_sizes_.begin(), split_sizes_.end());
  }

  ORT_RETURN_IF_ERROR(PrepareForCompute(input_shape,
                                        num_outputs,
                                        axis,
                                        before_dims,
                                        after_dims_including_split_axis,
                                        after_dims_excluding_split,
                                        split_sizes));

  const auto input_strides = StridesForTensor(input);

  // copy dimensions so we can update the selected axis in place
  auto output_dimensions = input_shape.AsShapeVector();

  SafeInt<ptrdiff_t> input_offset = 0;

  for (int i = 0; i < num_outputs; ++i) {
    // update size of dimension for axis we're splitting on
    auto split_size = narrow<int>(split_sizes[i]);
    output_dimensions[narrow<size_t>(axis)] = split_size;

    Tensor* output = context->Output(i, TensorShape{output_dimensions});
    const auto output_strides = StridesForTensor(*output);

    ORT_RETURN_IF_ERROR(DispatchStridedCopy<EnabledSplitDataTypes>(context->GetOperatorThreadPool(),
                                                                   *output, /* dst_offset */ 0, output_strides,
                                                                   output->Shape(),
                                                                   input, input_offset, input_strides));

    input_offset += SafeInt<ptrdiff_t>(split_size) * after_dims_excluding_split;  // offset by the data we used in this iteration
  }

  return Status::OK();
}

}  // namespace onnxruntime
