// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/tensor/gather_nd.h"

#include "core/providers/musa/musa_call.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/tensor/gather_nd_impl.h"

#include <cstddef>
#include <vector>

namespace onnxruntime {
namespace musa {
namespace {

Status CheckBatchDimensionsMatch(size_t num_batch_dimensions,
                                 const TensorShape& input_shape,
                                 const TensorShape& indices_shape) {
  ORT_RETURN_IF_NOT(num_batch_dimensions <= input_shape.NumDimensions(),
                    "GatherND: batch_dims exceeds input rank");
  ORT_RETURN_IF_NOT(num_batch_dimensions <= indices_shape.NumDimensions(),
                    "GatherND: batch_dims exceeds indices rank");

  for (size_t i = 0; i < num_batch_dimensions; ++i) {
    ORT_RETURN_IF_NOT(input_shape[i] == indices_shape[i],
                      "GatherND: input and indices batch dimensions differ at ", i,
                      ": ", input_shape[i], " != ", indices_shape[i]);
  }

  return Status::OK();
}

template <typename T, int32_t capacity, typename Dims>
Status CopyDimsToFixedArray(GatherNDFixedArray<T, capacity>& dst,
                            const Dims& src,
                            const char* name) {
  ORT_RETURN_IF_NOT(src.size() <= static_cast<size_t>(dst.Capacity()),
                    "GatherND ", name, " rank ", src.size(),
                    " exceeds kernel capacity ", dst.Capacity());
  dst.SetSize(static_cast<int32_t>(src.size()));
  for (int32_t i = 0; i < dst.Size(); ++i) {
    dst[i] = static_cast<T>(src[static_cast<size_t>(i)]);
  }
  return Status::OK();
}

bool IsSupportedElementSize(size_t element_size) {
  return element_size == 1 || element_size == 2 || element_size == 4 || element_size == 8;
}

}  // namespace

Status GatherNDBase::ValidateInputShapes(const TensorShape& input_shape,
                                         const TensorShape& indices_shape) const {
  ORT_RETURN_IF_NOT(indices_shape.NumDimensions() > 0,
                    "GatherND: indices tensor must have rank larger than 0");
  ORT_RETURN_IF_NOT(batch_dims_ < static_cast<int64_t>(indices_shape.NumDimensions()),
                    "GatherND: batch_dims must be less than indices rank");
  ORT_RETURN_IF_NOT(batch_dims_ <= static_cast<int64_t>(input_shape.NumDimensions()),
                    "GatherND: batch_dims must not exceed input rank");

  const int64_t num_slice_dims = indices_shape[indices_shape.NumDimensions() - 1];
  ORT_RETURN_IF_NOT(num_slice_dims >= 0, "GatherND: last indices dimension must be non-negative");

  const int64_t last_indices_dimension = batch_dims_ + num_slice_dims;
  ORT_RETURN_IF_NOT(last_indices_dimension <= static_cast<int64_t>(input_shape.NumDimensions()),
                    "GatherND: last dimension of indices must not be larger than rank of input tensor");

  ORT_RETURN_IF_ERROR(CheckBatchDimensionsMatch(static_cast<size_t>(batch_dims_), input_shape, indices_shape));
  return Status::OK();
}

template <typename TIndex>
Status GatherNDBase::ValidateIndicesOnCpu(const TensorShape& input_shape,
                                          const TensorShape& indices_shape,
                                          const Tensor* indices_tensor) const {
  const int64_t num_slice_dims = indices_shape[indices_shape.NumDimensions() - 1];
  const int64_t num_slices = indices_shape.SizeToDimension(indices_shape.NumDimensions() - 1);
  const int64_t num_batches = input_shape.SizeToDimension(static_cast<size_t>(batch_dims_));

  ORT_RETURN_IF_NOT(num_batches != 0,
                    "GatherND: input tensor batch dimensions cannot be zero");
  ORT_RETURN_IF_NOT(num_slices % num_batches == 0,
                    "GatherND: indices batch size ", num_slices,
                    " is not divisible by input batch size ", num_batches);

  const TIndex* indices_data = indices_tensor->Data<TIndex>();
  for (int64_t slice_idx = 0; slice_idx < num_slices; ++slice_idx) {
    const TIndex* slice_indices = indices_data + slice_idx * num_slice_dims;
    for (int64_t dim_idx = 0; dim_idx < num_slice_dims; ++dim_idx) {
      const int64_t input_dim_idx = batch_dims_ + dim_idx;
      const int64_t dim_size = input_shape[static_cast<size_t>(input_dim_idx)];
      const int64_t index = static_cast<int64_t>(slice_indices[dim_idx]);
      if (index < -dim_size || index >= dim_size) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "GatherND: invalid index found, index = ", index);
      }
    }
  }

  return Status::OK();
}

template <typename TIndex>
Status GatherND<TIndex>::ComputeInternal(OpKernelContext* ctx) const {
  const auto* input_tensor = ctx->Input<Tensor>(0);
  const auto* indices_tensor = ctx->Input<Tensor>(1);
  ORT_RETURN_IF_NOT(input_tensor != nullptr, "GatherND: input tensor is null");
  ORT_RETURN_IF_NOT(indices_tensor != nullptr, "GatherND: indices tensor is null");

  const TensorShape& input_shape = input_tensor->Shape();
  const TensorShape& indices_shape = indices_tensor->Shape();
  ORT_RETURN_IF_ERROR(ValidateInputShapes(input_shape, indices_shape));

  const int64_t num_slice_dims = indices_shape[indices_shape.NumDimensions() - 1];
  const int64_t last_indices_dimension = batch_dims_ + num_slice_dims;

  std::vector<int64_t> output_shape(indices_shape.GetDims().begin(), indices_shape.GetDims().end() - 1);
  const auto input_suffix_begin = input_shape.GetDims().begin() + static_cast<std::ptrdiff_t>(last_indices_dimension);
  output_shape.insert(output_shape.end(), input_suffix_begin, input_shape.GetDims().end());

  Tensor* output_tensor = ctx->Output(0, TensorShape(output_shape));
  if (output_tensor->Shape().Size() == 0) {
    return Status::OK();
  }

  ORT_RETURN_IF_ERROR(ValidateIndicesOnCpu<TIndex>(input_shape, indices_shape, indices_tensor));

  const size_t element_size = input_tensor->DataType()->Size();
  ORT_RETURN_IF_NOT(IsSupportedElementSize(element_size),
                    "GatherND MUSA kernel only supports POD tensor element sizes 1, 2, 4, and 8. Actual: ",
                    element_size);

  GatherNDKernelArgs args;
  args.batch_dims = batch_dims_;
  args.num_slice_dims = num_slice_dims;
  args.num_slices = indices_shape.SizeToDimension(indices_shape.NumDimensions() - 1);
  args.slice_size = input_shape.SizeFromDimension(static_cast<size_t>(last_indices_dimension));
  args.num_batches = input_shape.SizeToDimension(static_cast<size_t>(batch_dims_));
  args.num_slices_per_batch = args.num_slices / args.num_batches;
  args.input_batch_stride = input_shape.SizeFromDimension(static_cast<size_t>(batch_dims_));

  ORT_RETURN_IF_ERROR(CopyDimsToFixedArray(args.input_dims, input_shape.GetDims(), "input"));

  std::vector<int64_t> slice_strides(static_cast<size_t>(num_slice_dims));
  int64_t running_product = args.slice_size;
  for (int64_t i = num_slice_dims - 1; i >= 0; --i) {
    slice_strides[static_cast<size_t>(i)] = running_product;
    running_product *= input_shape[static_cast<size_t>(batch_dims_ + i)];
  }
  ORT_RETURN_IF_ERROR(CopyDimsToFixedArray(args.slice_strides, slice_strides, "slice stride"));

  void* device_indices = nullptr;
  IAllocatorUniquePtr<unsigned char> indices_buffer;
  const size_t indices_bytes = static_cast<size_t>(indices_shape.Size()) * sizeof(TIndex);
  if (indices_bytes > 0) {
    indices_buffer = GetScratchBuffer<unsigned char>(indices_bytes, ctx->GetComputeStream());
    MUSA_RETURN_IF_ERROR(musaMemcpyAsync(indices_buffer.get(), indices_tensor->DataRaw(), indices_bytes,
                                         musaMemcpyHostToDevice, Stream(ctx)));
    device_indices = indices_buffer.get();
  }

  GatherNDImpl<TIndex>(Stream(ctx),
                       args,
                       device_indices,
                       input_tensor->DataRaw(),
                       output_tensor->MutableDataRaw(),
                       element_size,
                       static_cast<size_t>(output_tensor->Shape().Size()));

  return Status::OK();
}

#ifdef ENABLE_STRIDED_TENSORS
#define CREATE_GATHER_ND_KERNEL_DEF (*KernelDefBuilder::Create()).MayStridedInput(1).InputMemoryType(OrtMemTypeCPUInput, 1)
#else
#define CREATE_GATHER_ND_KERNEL_DEF (*KernelDefBuilder::Create()).InputMemoryType(OrtMemTypeCPUInput, 1)
#endif

#define REGISTER_MUSA_GATHER_ND_VERSIONED_TYPED_KERNEL(startver, endver, TIndex) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                       \
      GatherND, kOnnxDomain, startver, endver, TIndex, kMusaExecutionProvider,   \
      CREATE_GATHER_ND_KERNEL_DEF                                                \
          .TypeConstraint("T", std::vector<MLDataType>{                          \
                                   DataTypeImpl::GetTensorType<float>(),         \
                                   DataTypeImpl::GetTensorType<double>(),        \
                                   DataTypeImpl::GetTensorType<MLFloat16>(),     \
                                   DataTypeImpl::GetTensorType<int32_t>(),       \
                                   DataTypeImpl::GetTensorType<int64_t>(),       \
                                   DataTypeImpl::GetTensorType<bool>()})         \
          .TypeConstraint("indices", DataTypeImpl::GetTensorType<TIndex>()),     \
      GatherND<TIndex>);

#define REGISTER_MUSA_GATHER_ND_TYPED_KERNEL(ver, TIndex)                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                             \
      GatherND, kOnnxDomain, ver, TIndex, kMusaExecutionProvider,            \
      CREATE_GATHER_ND_KERNEL_DEF                                            \
          .TypeConstraint("T", std::vector<MLDataType>{                      \
                                   DataTypeImpl::GetTensorType<float>(),     \
                                   DataTypeImpl::GetTensorType<double>(),    \
                                   DataTypeImpl::GetTensorType<MLFloat16>(), \
                                   DataTypeImpl::GetTensorType<int32_t>(),   \
                                   DataTypeImpl::GetTensorType<int64_t>(),   \
                                   DataTypeImpl::GetTensorType<bool>()})     \
          .TypeConstraint("indices", DataTypeImpl::GetTensorType<TIndex>()), \
      GatherND<TIndex>);

REGISTER_MUSA_GATHER_ND_VERSIONED_TYPED_KERNEL(11, 11, int64_t)
REGISTER_MUSA_GATHER_ND_VERSIONED_TYPED_KERNEL(12, 12, int64_t)
REGISTER_MUSA_GATHER_ND_TYPED_KERNEL(13, int64_t)

#define REGISTER_MUSA_GATHER_ND_COMPAT_TYPED_KERNEL(ver, TIndex)           \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                           \
      GatherNd, kOnnxDomain, ver, TIndex, kMusaExecutionProvider,          \
      CREATE_GATHER_ND_KERNEL_DEF                                          \
          .TypeConstraint("T", std::vector<MLDataType>{                 \
                                   DataTypeImpl::GetTensorType<float>(),   \
                                   DataTypeImpl::GetTensorType<double>(),  \
                                   DataTypeImpl::GetTensorType<MLFloat16>(), \
                                   DataTypeImpl::GetTensorType<int32_t>(), \
                                   DataTypeImpl::GetTensorType<int64_t>(), \
                                   DataTypeImpl::GetTensorType<bool>()})   \
          .TypeConstraint("Tindices", DataTypeImpl::GetTensorType<TIndex>()), \
      GatherND<TIndex>);

REGISTER_MUSA_GATHER_ND_COMPAT_TYPED_KERNEL(1, int64_t)

#undef REGISTER_MUSA_GATHER_ND_COMPAT_TYPED_KERNEL
#undef REGISTER_MUSA_GATHER_ND_VERSIONED_TYPED_KERNEL
#undef REGISTER_MUSA_GATHER_ND_TYPED_KERNEL
#undef CREATE_GATHER_ND_KERNEL_DEF

template Status GatherNDBase::ValidateIndicesOnCpu<int64_t>(const TensorShape&, const TensorShape&, const Tensor*) const;
template Status GatherND<int64_t>::ComputeInternal(OpKernelContext*) const;

}  // namespace musa
}  // namespace onnxruntime
