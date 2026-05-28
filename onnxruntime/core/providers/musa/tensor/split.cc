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

class SplitV final : public MusaKernel {
 public:
  explicit SplitV(const OpKernelInfo& info) : MusaKernel(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
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
          MUSA_RETURN_IF_ERROR(musaMemcpyAsync(dst, src, bytes, musaMemcpyDeviceToDevice, Stream(ctx)));
        }
        src_axis_offset += split;
      }
    }

    return Status::OK();
  }
};

}  // namespace

ONNX_OPERATOR_KERNEL_EX(
    SplitV,
    kOnnxDomain,
    1,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .InputMemoryType(OrtMemTypeCPUInput, 2)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("Tlen", BuildKernelDefConstraints<int32_t, int64_t>())
        .TypeConstraint("Taxis", BuildKernelDefConstraints<int32_t, int64_t>()),
    SplitV);

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
