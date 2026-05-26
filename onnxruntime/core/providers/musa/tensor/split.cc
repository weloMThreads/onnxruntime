// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/tensor/split.h"

#include <algorithm>
#include <vector>

#include "core/providers/common.h"
#include "core/providers/musa/musa_call.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/tensor/split_impl.h"

namespace onnxruntime {
namespace musa {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Split,
                                  kOnnxDomain,
                                  2, 10,
                                  kMusaExecutionProvider,
                                  (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
                                  Split_2_13);

// explicitly supports negative axis
ONNX_OPERATOR_VERSIONED_KERNEL_EX(Split,
                                  kOnnxDomain,
                                  11, 12,
                                  kMusaExecutionProvider,
                                  (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
                                  Split_2_13);

// explicitly supports 'split' as optional input
ONNX_OPERATOR_VERSIONED_KERNEL_EX(Split,
                                  kOnnxDomain,
                                  13, 17,
                                  kMusaExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .InputMemoryType(OrtMemTypeCPUInput, 1)
                                      .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
                                  Split_2_13);

ONNX_OPERATOR_KERNEL_EX(Split,
                        kOnnxDomain,
                        18,
                        kMusaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .InputMemoryType(OrtMemTypeCPUInput, 1)
                            .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
                        Split_18);

namespace {

Status CopyHostVectorToDevice(const std::vector<void*>& src, void** dst, musaStream_t stream) {
  if (src.empty()) {
    return Status::OK();
  }

  MUSA_RETURN_IF_ERROR(musaMemcpyAsync(dst, src.data(), src.size() * sizeof(void*), musaMemcpyHostToDevice, stream));
  return Status::OK();
}

Status CopyHostVectorToDevice(const std::vector<int64_t>& src, int64_t* dst, musaStream_t stream) {
  if (src.empty()) {
    return Status::OK();
  }

  MUSA_RETURN_IF_ERROR(
      musaMemcpyAsync(dst, src.data(), src.size() * sizeof(int64_t), musaMemcpyHostToDevice, stream));
  return Status::OK();
}

Status CopyHostVectorToDevice(const TensorShapeVector& src, int64_t* dst, musaStream_t stream) {
  if (src.empty()) {
    return Status::OK();
  }

  MUSA_RETURN_IF_ERROR(
      musaMemcpyAsync(dst, src.data(), src.size() * sizeof(int64_t), musaMemcpyHostToDevice, stream));
  return Status::OK();
}

}  // namespace

Status SplitKernel::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* input_tensor = ctx->Input<Tensor>(0);
  ORT_ENFORCE(input_tensor);

  const auto& input_shape = input_tensor->Shape();
  const int num_outputs = ctx->OutputCount();
  int64_t axis = HandleNegativeAxis(axis_, input_shape.NumDimensions());
  int before_dims = 0;
  int block_size_including_axis_dim = 0;
  int block_size_inside_axis_dim = 0;
  std::vector<int64_t> split_sizes(static_cast<size_t>(num_outputs));

  const Tensor* split_tensor = ctx->Input<Tensor>(1);
  if (split_tensor) {
    ORT_ENFORCE(split_tensor->Shape().NumDimensions() == 1, "A split tensor must be a vector tensor.");
    size_t n_dims = static_cast<size_t>(split_tensor->Shape()[0]);
    const int64_t* data = split_tensor->Data<int64_t>();
    split_sizes.assign(data, data + n_dims);
  } else {
    split_sizes.assign(split_sizes_.begin(), split_sizes_.end());
  }

  ORT_RETURN_IF_ERROR(PrepareForCompute(input_shape,
                                        num_outputs,
                                        axis,
                                        before_dims,
                                        block_size_including_axis_dim,
                                        block_size_inside_axis_dim,
                                        split_sizes));

  auto output_dimensions = input_shape.AsShapeVector();
  std::vector<void*> output_ptr_cpu(static_cast<size_t>(num_outputs));
  TensorShapeVector axis_dimension_input_output_mapping(
      static_cast<size_t>(input_shape.GetDims()[static_cast<size_t>(axis)]));
  int mapping_index = 0;

  for (int i = 0; i < num_outputs; ++i) {
    int split_size = static_cast<int>(split_sizes[static_cast<size_t>(i)]);
    output_dimensions[static_cast<size_t>(axis)] = split_size;

    Tensor* output = ctx->Output(i, TensorShape{output_dimensions});
    output_ptr_cpu[static_cast<size_t>(i)] = output->MutableDataRaw();
    for (int j = 0; j < split_size; ++j) {
      axis_dimension_input_output_mapping.at(static_cast<size_t>(mapping_index++)) = i;
    }
  }

  if (input_shape.Size() <= 0) {
    return Status::OK();
  }

  auto output_ptr_gpu = GetScratchBuffer<void*>(static_cast<size_t>(num_outputs), ctx->GetComputeStream());
  ORT_RETURN_IF_ERROR(CopyHostVectorToDevice(output_ptr_cpu, output_ptr_gpu.get(), Stream(ctx)));

  size_t element_size = input_tensor->DataType()->Size();
  if (std::all_of(split_sizes.begin(), split_sizes.end(),
                  [&](int64_t size) { return size == split_sizes.front(); })) {
    return SplitSameSplitDimImpl(Stream(ctx), element_size, block_size_including_axis_dim,
                                 block_size_inside_axis_dim, split_sizes.front(), num_outputs,
                                 input_tensor->DataRaw(), output_ptr_gpu.get(),
                                 static_cast<size_t>(input_shape.Size()));
  }

  auto split_sizes_gpu = GetScratchBuffer<int64_t>(split_sizes.size(), ctx->GetComputeStream());
  ORT_RETURN_IF_ERROR(CopyHostVectorToDevice(split_sizes, split_sizes_gpu.get(), Stream(ctx)));

  std::vector<int64_t> split_sizes_range(split_sizes);
  for (size_t i = 1; i < split_sizes_range.size(); ++i) {
    split_sizes_range[i] += split_sizes_range[i - 1];
  }

  auto split_sizes_range_gpu = GetScratchBuffer<int64_t>(split_sizes_range.size(), ctx->GetComputeStream());
  ORT_RETURN_IF_ERROR(CopyHostVectorToDevice(split_sizes_range, split_sizes_range_gpu.get(), Stream(ctx)));

  auto axis_mapping_gpu =
      GetScratchBuffer<int64_t>(axis_dimension_input_output_mapping.size(), ctx->GetComputeStream());
  ORT_RETURN_IF_ERROR(
      CopyHostVectorToDevice(axis_dimension_input_output_mapping, axis_mapping_gpu.get(), Stream(ctx)));

  return SplitImpl(Stream(ctx), element_size, block_size_including_axis_dim, block_size_inside_axis_dim,
                   split_sizes_gpu.get(), split_sizes_range_gpu.get(), axis_mapping_gpu.get(), num_outputs,
                   input_tensor->DataRaw(), output_ptr_gpu.get(), static_cast<size_t>(input_shape.Size()));
}

}  // namespace musa
}  // namespace onnxruntime
