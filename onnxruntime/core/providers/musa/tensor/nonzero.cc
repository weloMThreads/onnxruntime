// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/tensor/nonzero.h"

#include "core/providers/musa/musa_call.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/tensor/nonzero_impl.h"
#include "core/providers/shared_library/provider_api.h"

#include <type_traits>

namespace onnxruntime {
namespace musa {
namespace {

Status FillLaunchParams(const TensorShape& shape, NonZeroLaunchParams& params) {
  const size_t rank = shape.NumDimensions();
  ORT_RETURN_IF_NOT(rank <= static_cast<size_t>(kNonZeroMaxRank),
                    "NonZero: input rank ", rank, " exceeds MUSA kernel capacity ", kNonZeroMaxRank);

  params.total_elements = shape.Size();
  params.rank = static_cast<int32_t>(rank);
  params.coordinate_size = rank == 0 ? 1 : static_cast<int32_t>(rank);

  int64_t running_stride = 1;
  for (int64_t dim = static_cast<int64_t>(rank) - 1; dim >= 0; --dim) {
    params.dims[dim] = shape[static_cast<size_t>(dim)];
    params.strides[dim] = running_stride;
    running_stride *= shape[static_cast<size_t>(dim)];
  }

  return Status::OK();
}

}  // namespace

template <typename T>
Status NonZero<T>::ComputeInternal(OpKernelContext* ctx) const {
  const auto* input = ctx->Input<Tensor>(0);
  ORT_RETURN_IF_NOT(input != nullptr, "NonZero: input tensor is null");

  NonZeroLaunchParams params;
  ORT_RETURN_IF_ERROR(FillLaunchParams(input->Shape(), params));

  int64_t nonzero_count = 0;
  if (params.total_elements > 0) {
    IAllocatorUniquePtr<int64_t> count_buffer = GetScratchBuffer<int64_t>(1, ctx->GetComputeStream());
    if constexpr (std::is_same_v<T, MLFloat16>) {
      LaunchNonZeroCountKernelHalf(Stream(ctx), input->DataRaw(), params.total_elements, count_buffer.get());
    } else {
      LaunchNonZeroCountKernel<T>(Stream(ctx), input->Data<T>(), params.total_elements, count_buffer.get());
    }
    MUSA_RETURN_IF_ERROR(musaMemcpyAsync(&nonzero_count, count_buffer.get(), sizeof(int64_t),
                                         musaMemcpyDeviceToHost, Stream(ctx)));
    MUSA_RETURN_IF_ERROR(musaStreamSynchronize(Stream(ctx)));
  }

  params.nonzero_count = nonzero_count;
  TensorShapeVector output_dims{params.coordinate_size, nonzero_count};
  Tensor* output = ctx->Output(0, TensorShape(output_dims));
  ORT_RETURN_IF_NOT(output != nullptr, "NonZero: output tensor is null");

  if (nonzero_count == 0) {
    return Status::OK();
  }

  if constexpr (std::is_same_v<T, MLFloat16>) {
    LaunchNonZeroFillKernelHalf(Stream(ctx), input->DataRaw(), output->MutableData<int64_t>(), params);
  } else {
    LaunchNonZeroFillKernel<T>(Stream(ctx), input->Data<T>(), output->MutableData<int64_t>(), params);
  }

  MUSA_RETURN_IF_ERROR(musaGetLastError());
  return Status::OK();
}

template class NonZero<bool>;
template class NonZero<uint8_t>;
template class NonZero<int32_t>;
template class NonZero<int64_t>;
template class NonZero<float>;
template class NonZero<double>;
template class NonZero<MLFloat16>;

#define REGISTER_MUSA_NONZERO_VERSIONED_TYPED_KERNEL(startver, endver, T)        \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                       \
      NonZero, kOnnxDomain, startver, endver, T, kMusaExecutionProvider,         \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      NonZero<T>);

#define REGISTER_MUSA_NONZERO_TYPED_KERNEL(ver, T)                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                 \
      NonZero, kOnnxDomain, ver, T, kMusaExecutionProvider,                      \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      NonZero<T>);

REGISTER_MUSA_NONZERO_VERSIONED_TYPED_KERNEL(9, 12, bool)
REGISTER_MUSA_NONZERO_VERSIONED_TYPED_KERNEL(9, 12, uint8_t)
REGISTER_MUSA_NONZERO_VERSIONED_TYPED_KERNEL(9, 12, int32_t)
REGISTER_MUSA_NONZERO_VERSIONED_TYPED_KERNEL(9, 12, int64_t)
REGISTER_MUSA_NONZERO_VERSIONED_TYPED_KERNEL(9, 12, float)

REGISTER_MUSA_NONZERO_TYPED_KERNEL(13, bool)
REGISTER_MUSA_NONZERO_TYPED_KERNEL(13, uint8_t)
REGISTER_MUSA_NONZERO_TYPED_KERNEL(13, int32_t)
REGISTER_MUSA_NONZERO_TYPED_KERNEL(13, int64_t)
REGISTER_MUSA_NONZERO_TYPED_KERNEL(13, float)
REGISTER_MUSA_NONZERO_TYPED_KERNEL(13, double)
REGISTER_MUSA_NONZERO_TYPED_KERNEL(13, MLFloat16)

#undef REGISTER_MUSA_NONZERO_TYPED_KERNEL
#undef REGISTER_MUSA_NONZERO_VERSIONED_TYPED_KERNEL

}  // namespace musa
}  // namespace onnxruntime
