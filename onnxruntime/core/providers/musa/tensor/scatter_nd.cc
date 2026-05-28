// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/tensor/scatter_nd.h"

#include "core/providers/cpu/tensor/scatter_nd.h"
#include "core/providers/musa/musa_call.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/tensor/scatter_nd_impl.h"

#include <cstddef>
#include <vector>

namespace onnxruntime {
namespace musa {
namespace {

template <typename T, int32_t capacity, typename Dims>
Status CopyDimsToFixedArray(ScatterNDFixedArray<T, capacity>& dst,
                            const Dims& src,
                            const char* name) {
  ORT_RETURN_IF_NOT(src.size() <= static_cast<size_t>(dst.Capacity()),
                    "ScatterND ", name, " rank ", src.size(),
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

Status ScatterND::ValidateIndicesOnCpu(const TensorShape& input_shape,
                                       const TensorShape& indices_shape,
                                       const Tensor* indices_tensor) const {
  const int64_t last_index_dimension = indices_shape[indices_shape.NumDimensions() - 1];
  ORT_RETURN_IF_NOT(last_index_dimension > 0,
                    "ScatterND: last dimension of indices must be greater than zero");

  const int64_t num_indices = indices_shape.Size() / last_index_dimension;
  const int64_t* indices_data = indices_tensor->Data<int64_t>();
  for (int64_t index_idx = 0; index_idx < num_indices; ++index_idx) {
    const int64_t* update_indices = indices_data + index_idx * last_index_dimension;
    for (int64_t dim_idx = 0; dim_idx < last_index_dimension; ++dim_idx) {
      const int64_t dim_size = input_shape[static_cast<size_t>(dim_idx)];
      const int64_t index = update_indices[dim_idx];
      if (index < -dim_size || index >= dim_size) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "ScatterND: invalid index found, index = ", index);
      }
    }
  }

  return Status::OK();
}

Status ScatterND::ComputeInternal(OpKernelContext* ctx) const {
  const auto* input_tensor = ctx->Input<Tensor>(0);
  const auto* indices_tensor = ctx->Input<Tensor>(1);
  const auto* updates_tensor = ctx->Input<Tensor>(2);
  ORT_RETURN_IF_NOT(input_tensor != nullptr, "ScatterND: input tensor is null");
  ORT_RETURN_IF_NOT(indices_tensor != nullptr, "ScatterND: indices tensor is null");
  ORT_RETURN_IF_NOT(updates_tensor != nullptr, "ScatterND: updates tensor is null");

  const TensorShape& input_shape = input_tensor->Shape();
  const TensorShape& indices_shape = indices_tensor->Shape();
  const TensorShape& updates_shape = updates_tensor->Shape();
  ORT_RETURN_IF_ERROR(scatter_nd_internal::ValidateShapes(input_shape, indices_shape, updates_shape));
  ORT_RETURN_IF_ERROR(ValidateIndicesOnCpu(input_shape, indices_shape, indices_tensor));

  const size_t element_size = input_tensor->DataType()->Size();
  ORT_RETURN_IF_NOT(IsSupportedElementSize(element_size),
                    "ScatterND MUSA kernel only supports POD tensor element sizes 1, 2, 4, and 8. Actual: ",
                    element_size);

  Tensor* output_tensor = ctx->Output(0, input_shape);
  if (input_tensor->DataRaw() != output_tensor->MutableDataRaw() && input_tensor->SizeInBytes() > 0) {
    MUSA_RETURN_IF_ERROR(musaMemcpyAsync(output_tensor->MutableDataRaw(), input_tensor->DataRaw(), input_tensor->SizeInBytes(),
                                         musaMemcpyDeviceToDevice, Stream(ctx)));
  }

  if (indices_shape.Size() == 0 || updates_shape.Size() == 0) {
    return Status::OK();
  }

  const int64_t last_index_dimension = indices_shape[indices_shape.NumDimensions() - 1];
  const int64_t num_indices = indices_shape.Size() / last_index_dimension;
  const int64_t slice_size = input_shape.SizeFromDimension(static_cast<size_t>(last_index_dimension));

  ScatterNDKernelArgs args;
  args.num_indices = num_indices;
  args.last_index_dimension = last_index_dimension;
  args.slice_size = slice_size;
  ORT_RETURN_IF_ERROR(CopyDimsToFixedArray(args.input_dims, input_shape.GetDims(), "input"));

  std::vector<int64_t> input_strides(static_cast<size_t>(last_index_dimension));
  int64_t running_product = slice_size;
  for (int64_t i = last_index_dimension - 1; i >= 0; --i) {
    input_strides[static_cast<size_t>(i)] = running_product;
    running_product *= input_shape[static_cast<size_t>(i)];
  }
  ORT_RETURN_IF_ERROR(CopyDimsToFixedArray(args.input_strides, input_strides, "input stride"));

  IAllocatorUniquePtr<int64_t> indices_buffer;
  const size_t indices_bytes = static_cast<size_t>(indices_shape.Size()) * sizeof(int64_t);
  indices_buffer = GetScratchBuffer<int64_t>(static_cast<size_t>(indices_shape.Size()), ctx->GetComputeStream());
  MUSA_RETURN_IF_ERROR(musaMemcpyAsync(indices_buffer.get(), indices_tensor->DataRaw(), indices_bytes,
                                       musaMemcpyHostToDevice, Stream(ctx)));

  ScatterNDImpl(Stream(ctx),
                args,
                indices_buffer.get(),
                updates_tensor->DataRaw(),
                output_tensor->MutableDataRaw(),
                element_size,
                static_cast<size_t>(updates_shape.Size()));

  return Status::OK();
}

#ifdef ENABLE_STRIDED_TENSORS
#define CREATE_SCATTER_ND_KERNEL_DEF (*KernelDefBuilder::Create()).MayStridedInput(1).InputMemoryType(OrtMemTypeCPUInput, 1).MayInplace(0, 0)
#else
#define CREATE_SCATTER_ND_KERNEL_DEF (*KernelDefBuilder::Create()).InputMemoryType(OrtMemTypeCPUInput, 1).MayInplace(0, 0)
#endif

#define MUSA_SCATTER_ND_TYPE_CONSTRAINTS          \
  std::vector<MLDataType> {                       \
    DataTypeImpl::GetTensorType<float>(),         \
        DataTypeImpl::GetTensorType<double>(),    \
        DataTypeImpl::GetTensorType<MLFloat16>(), \
        DataTypeImpl::GetTensorType<int32_t>(),   \
        DataTypeImpl::GetTensorType<int64_t>(),   \
        DataTypeImpl::GetTensorType<bool>()       \
  }

ONNX_OPERATOR_VERSIONED_KERNEL_EX(ScatterND,
                                  kOnnxDomain,
                                  11, 12,
                                  kMusaExecutionProvider,
                                  CREATE_SCATTER_ND_KERNEL_DEF.TypeConstraint("T", MUSA_SCATTER_ND_TYPE_CONSTRAINTS),
                                  ScatterND);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(ScatterND,
                                  kOnnxDomain,
                                  13, 15,
                                  kMusaExecutionProvider,
                                  CREATE_SCATTER_ND_KERNEL_DEF.TypeConstraint("T", MUSA_SCATTER_ND_TYPE_CONSTRAINTS),
                                  ScatterND);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(ScatterND,
                                  kOnnxDomain,
                                  16, 17,
                                  kMusaExecutionProvider,
                                  CREATE_SCATTER_ND_KERNEL_DEF.TypeConstraint("T", MUSA_SCATTER_ND_TYPE_CONSTRAINTS),
                                  ScatterND);

ONNX_OPERATOR_KERNEL_EX(ScatterND,
                        kOnnxDomain,
                        18,
                        kMusaExecutionProvider,
                        CREATE_SCATTER_ND_KERNEL_DEF.TypeConstraint("T", MUSA_SCATTER_ND_TYPE_CONSTRAINTS),
                        ScatterND);

#undef MUSA_SCATTER_ND_TYPE_CONSTRAINTS
#undef CREATE_SCATTER_ND_KERNEL_DEF

}  // namespace musa
}  // namespace onnxruntime
