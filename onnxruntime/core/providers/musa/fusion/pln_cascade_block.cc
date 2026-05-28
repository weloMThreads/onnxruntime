// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/fusion/pln_cascade_block.h"
#include "core/providers/musa/fusion/pln_cascade_block_impl.h"
#include "core/providers/musa/musa_fwd.h"

#include <limits>
#include <vector>

namespace onnxruntime {
namespace musa {
namespace {

PlnCascadeBlockShape BuildShape(const TensorShape& shape) {
  PlnCascadeBlockShape result{};
  result.rank = static_cast<int>(shape.NumDimensions());
  for (int i = 0; i < kPlnCascadeBlockMaxDims; ++i) {
    result.dims[i] = 1;
  }
  for (int i = 0; i < result.rank; ++i) {
    result.dims[i] = static_cast<int>(shape[i]);
  }
  return result;
}

Status BuildBroadcastStrides(const TensorShape& input_shape,
                             const TensorShape& output_shape,
                             PlnCascadeBlockStrides& strides) {
  for (int i = 0; i < kPlnCascadeBlockMaxDims; ++i) {
    strides.values[i] = 0;
  }

  const int input_rank = static_cast<int>(input_shape.NumDimensions());
  const int output_rank = static_cast<int>(output_shape.NumDimensions());
  if (input_rank > output_rank || output_rank > kPlnCascadeBlockMaxDims) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "MusaPlnCascadeBlock unsupported broadcast ranks: input=",
                           input_rank, ", output=", output_rank);
  }

  std::vector<int64_t> dense_strides(input_rank, 1);
  int64_t acc = 1;
  for (int i = input_rank - 1; i >= 0; --i) {
    dense_strides[i] = acc;
    acc *= input_shape[i];
  }

  const int rank_delta = output_rank - input_rank;
  for (int out_axis = 0; out_axis < output_rank; ++out_axis) {
    const int in_axis = out_axis - rank_delta;
    if (in_axis < 0) {
      strides.values[out_axis] = 0;
      continue;
    }

    const int64_t in_dim = input_shape[in_axis];
    const int64_t out_dim = output_shape[out_axis];
    if (in_dim == out_dim) {
      strides.values[out_axis] = static_cast<int>(dense_strides[in_axis]);
    } else if (in_dim == 1) {
      strides.values[out_axis] = 0;
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "MusaPlnCascadeBlock cannot broadcast shape ",
                             input_shape.ToString(), " to ", output_shape.ToString());
    }
  }

  return Status::OK();
}

Status BuildMaskStrides(const TensorShape& mask_shape,
                        const TensorShape& output_shape,
                        PlnCascadeBlockStrides& strides) {
  const Status right_aligned = BuildBroadcastStrides(mask_shape, output_shape, strides);
  if (right_aligned.IsOK()) {
    return right_aligned;
  }

  const int mask_rank = static_cast<int>(mask_shape.NumDimensions());
  const int output_rank = static_cast<int>(output_shape.NumDimensions());
  if (mask_rank == 1 && output_rank >= 2 && mask_shape[0] == output_shape[0]) {
    for (int i = 0; i < kPlnCascadeBlockMaxDims; ++i) {
      strides.values[i] = 0;
    }
    strides.values[0] = 1;
    return Status::OK();
  }

  return right_aligned;
}

Status ValidateScaleOrBias(const Tensor& tensor,
                           const TensorShape& output_shape,
                           int& value_count) {
  const int64_t count = tensor.Shape().Size();
  ORT_RETURN_IF_NOT(count <= std::numeric_limits<int>::max(),
                    "MusaPlnCascadeBlock scale/bias tensor is too large");
  const int64_t last_dim = output_shape[output_shape.NumDimensions() - 1];
  ORT_RETURN_IF_NOT(count == 1 || count == last_dim,
                    "MusaPlnCascadeBlock scale/bias size must be scalar or match output last dim. Got ",
                    tensor.Shape().ToString(), " for output ", output_shape.ToString());
  value_count = static_cast<int>(count);
  return Status::OK();
}

}  // namespace

Status MusaPlnCascadeBlock::ComputeInternal(OpKernelContext* ctx) const {
  const int expected_inputs = 1 + static_cast<int>(num_steps_) * 4;
  ORT_RETURN_IF_NOT(ctx->InputCount() == expected_inputs,
                    "MusaPlnCascadeBlock expected ", expected_inputs,
                    " inputs, got ", ctx->InputCount());

  const Tensor* norm_out = ctx->Input<Tensor>(0);
  ORT_RETURN_IF_NOT(norm_out != nullptr, "MusaPlnCascadeBlock norm_out is null");
  ORT_RETURN_IF_NOT(norm_out->IsDataType<float>(), "MusaPlnCascadeBlock supports float only");
  ORT_RETURN_IF_NOT(norm_out->Shape().NumDimensions() >= 1 &&
                        norm_out->Shape().NumDimensions() <= kPlnCascadeBlockMaxDims,
                    "MusaPlnCascadeBlock unsupported output rank: ",
                    norm_out->Shape().ToString());

  const TensorShape& output_shape = norm_out->Shape();
  Tensor* output = ctx->Output(0, output_shape);
  ORT_RETURN_IF_NOT(output != nullptr, "MusaPlnCascadeBlock output is null");
  if (output_shape.Size() == 0) {
    return Status::OK();
  }
  ORT_RETURN_IF_NOT(output_shape.Size() <= std::numeric_limits<int>::max(),
                    "MusaPlnCascadeBlock output is too large: ", output_shape.Size());

  PlnCascadeBlockFloatPtrs ptrs{};
  PlnCascadeBlockMeta meta{};
  meta.num_steps = static_cast<int>(num_steps_);
  meta.output_last_dim = static_cast<int>(output_shape[output_shape.NumDimensions() - 1]);
  for (int step = 0; step < kPlnCascadeBlockMaxSteps; ++step) {
    ptrs.candidate_masks[step] = nullptr;
    ptrs.passthrough_masks[step] = nullptr;
    ptrs.add_values[step] = nullptr;
    ptrs.bias_values[step] = nullptr;
    meta.add_value_counts[step] = 0;
    meta.bias_value_counts[step] = 0;
    for (int dim = 0; dim < kPlnCascadeBlockMaxDims; ++dim) {
      meta.candidate_mask_strides[step].values[dim] = 0;
      meta.passthrough_mask_strides[step].values[dim] = 0;
    }
  }

  for (int step = 0; step < meta.num_steps; ++step) {
    const int base = 1 + step * 4;
    const Tensor* candidate_mask = ctx->Input<Tensor>(base);
    const Tensor* passthrough_mask = ctx->Input<Tensor>(base + 1);
    const Tensor* add_values = ctx->Input<Tensor>(base + 2);
    const Tensor* bias_values = ctx->Input<Tensor>(base + 3);
    ORT_RETURN_IF_NOT(candidate_mask != nullptr && passthrough_mask != nullptr &&
                          add_values != nullptr && bias_values != nullptr,
                      "MusaPlnCascadeBlock input is null at step ", step);
    ORT_RETURN_IF_NOT(candidate_mask->IsDataType<float>() &&
                          passthrough_mask->IsDataType<float>() &&
                          add_values->IsDataType<float>() &&
                          bias_values->IsDataType<float>(),
                      "MusaPlnCascadeBlock all inputs must be float at step ", step);

    ORT_RETURN_IF_ERROR(BuildMaskStrides(candidate_mask->Shape(), output_shape,
                                         meta.candidate_mask_strides[step]));
    ORT_RETURN_IF_ERROR(BuildMaskStrides(passthrough_mask->Shape(), output_shape,
                                         meta.passthrough_mask_strides[step]));
    ORT_RETURN_IF_ERROR(ValidateScaleOrBias(*add_values, output_shape,
                                            meta.add_value_counts[step]));
    ORT_RETURN_IF_ERROR(ValidateScaleOrBias(*bias_values, output_shape,
                                            meta.bias_value_counts[step]));

    ptrs.candidate_masks[step] = candidate_mask->Data<float>();
    ptrs.passthrough_masks[step] = passthrough_mask->Data<float>();
    ptrs.add_values[step] = add_values->Data<float>();
    ptrs.bias_values[step] = bias_values->Data<float>();
  }

  const musaError_t status = LaunchPlnCascadeBlockFloat(Stream(ctx),
                                                        norm_out->Data<float>(),
                                                        ptrs,
                                                        meta,
                                                        output->MutableData<float>(),
                                                        BuildShape(output_shape),
                                                        static_cast<int>(output_shape.Size()));
  if (status != musaSuccess) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "MusaPlnCascadeBlock kernel launch failed, status=",
                           static_cast<int>(status));
  }

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    MusaPlnCascadeBlock,
    kMSDomain,
    1,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create()),
    MusaPlnCascadeBlock);

}  // namespace musa
}  // namespace onnxruntime
