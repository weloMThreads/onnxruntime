// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/math/bitwise_extra_ops.h"

#include "core/providers/common.h"
#include "core/providers/musa/math/bitwise_extra_ops_impl.h"
#include "core/providers/musa/musa_call.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/shared_library/provider_api.h"

#include <musa_runtime.h>

#include <algorithm>
#include <vector>

using onnxruntime::common::Status;

namespace onnxruntime {
namespace musa {
namespace {

void FillOutputStrides(const std::vector<int64_t>& dims, int64_t* strides) {
  int64_t running_stride = 1;
  for (int64_t dim = static_cast<int64_t>(dims.size()) - 1; dim >= 0; --dim) {
    strides[dim] = running_stride;
    running_stride *= dims[dim];
  }
}

void FillInputStrides(const std::vector<int64_t>& dims, int64_t* strides) {
  int64_t running_stride = 1;
  for (int64_t dim = static_cast<int64_t>(dims.size()) - 1; dim >= 0; --dim) {
    strides[dim] = (dims[dim] == 1) ? 0 : running_stride;
    running_stride *= dims[dim];
  }
}

Status BuildBinaryParams(const TensorShape& lhs_shape,
                         const TensorShape& rhs_shape,
                         const TensorShape& output_shape,
                         BitwiseExtraBinaryParams* params) {
  ORT_RETURN_IF_NOT(params != nullptr, "BitwiseExtraBinaryParams must not be null");

  const int32_t rank = static_cast<int32_t>(output_shape.NumDimensions());
  ORT_RETURN_IF_NOT(rank <= kBitwiseExtraMaxDims,
                    "MUSA bitwise binary ops support rank up to ", kBitwiseExtraMaxDims,
                    ", got ", rank);

  params->rank = rank;
  params->total_elements = output_shape.Size();
  std::fill_n(params->output_strides, kBitwiseExtraMaxDims, 0);
  std::fill_n(params->lhs_strides, kBitwiseExtraMaxDims, 0);
  std::fill_n(params->rhs_strides, kBitwiseExtraMaxDims, 0);

  if (rank == 0 || params->total_elements == 0) {
    return Status::OK();
  }

  std::vector<int64_t> padded_output(rank, 1);
  std::vector<int64_t> padded_lhs(rank, 1);
  std::vector<int64_t> padded_rhs(rank, 1);

  const auto output_dims = output_shape.GetDims();
  std::copy(output_dims.begin(), output_dims.end(), padded_output.begin());

  const auto lhs_dims = lhs_shape.GetDims();
  const auto rhs_dims = rhs_shape.GetDims();
  const size_t lhs_offset = static_cast<size_t>(rank - static_cast<int32_t>(lhs_shape.NumDimensions()));
  const size_t rhs_offset = static_cast<size_t>(rank - static_cast<int32_t>(rhs_shape.NumDimensions()));
  std::copy(lhs_dims.begin(), lhs_dims.end(), padded_lhs.begin() + lhs_offset);
  std::copy(rhs_dims.begin(), rhs_dims.end(), padded_rhs.begin() + rhs_offset);

  FillOutputStrides(padded_output, params->output_strides);
  FillInputStrides(padded_lhs, params->lhs_strides);
  FillInputStrides(padded_rhs, params->rhs_strides);
  return Status::OK();
}

bool IsHostPointer(const void* ptr) {
  if (ptr == nullptr) {
    return false;
  }

  musaPointerAttributes attr;
  auto err = musaPointerGetAttributes(&attr, ptr);
  if (err != musaSuccess) {
    musaGetLastError();
    return true;
  }

  return attr.type == musaMemoryTypeHost || attr.type == musaMemoryTypeUnregistered;
}

template <typename T>
Status CopyHostInputIfNeeded(const void* input,
                             int64_t count,
                             musaStream_t stream,
                             const void** device_input,
                             void** temp_buffer) {
  *device_input = input;
  *temp_buffer = nullptr;
  if (!IsHostPointer(input)) {
    return Status::OK();
  }

  const size_t bytes = static_cast<size_t>(count) * sizeof(T);
  if (bytes == 0) {
    return Status::OK();
  }

  MUSA_RETURN_IF_ERROR(musaMalloc(temp_buffer, bytes));
  MUSA_RETURN_IF_ERROR(musaMemcpyAsync(*temp_buffer, input, bytes, musaMemcpyHostToDevice, stream));
  *device_input = *temp_buffer;
  return Status::OK();
}

void FreeTempBuffer(void* ptr) {
  if (ptr != nullptr) {
    musaFree(ptr);
  }
}

template <typename T, typename LaunchFn>
Status RunBitwiseBinary(OpKernelContext* ctx, const char* op_name, musaStream_t stream, LaunchFn launch_fn) {
  const Tensor* A = ctx->Input<Tensor>(0);
  const Tensor* B = ctx->Input<Tensor>(1);
  if (!A || !B) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input tensors");
  }

  TensorShape output_shape;
  ORT_RETURN_IF_ERROR(ComputeBroadcastOutputShape(op_name, A->Shape(), B->Shape(), output_shape));

  Tensor* C = ctx->Output(0, output_shape);
  if (!C) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor");
  }

  BitwiseExtraBinaryParams params{};
  ORT_RETURN_IF_ERROR(BuildBinaryParams(A->Shape(), B->Shape(), output_shape, &params));
  if (params.total_elements == 0) {
    return Status::OK();
  }

  if (!A->DataRaw() || !B->DataRaw() || !C->MutableDataRaw()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid tensor data pointers");
  }

  const void* device_a = nullptr;
  const void* device_b = nullptr;
  void* temp_a = nullptr;
  void* temp_b = nullptr;

  ORT_RETURN_IF_ERROR(CopyHostInputIfNeeded<T>(A->DataRaw(), A->Shape().Size(), stream, &device_a, &temp_a));
  ORT_RETURN_IF_ERROR(CopyHostInputIfNeeded<T>(B->DataRaw(), B->Shape().Size(), stream, &device_b, &temp_b));

  launch_fn(stream,
            reinterpret_cast<const T*>(device_a),
            reinterpret_cast<const T*>(device_b),
            reinterpret_cast<T*>(C->MutableDataRaw()),
            params);
  MUSA_RETURN_IF_ERROR(musaGetLastError());

  if (temp_a || temp_b) {
    MUSA_RETURN_IF_ERROR(musaStreamSynchronize(stream));
  }
  FreeTempBuffer(temp_a);
  FreeTempBuffer(temp_b);
  return Status::OK();
}

template <typename T>
Status RunBitwiseNot(OpKernelContext* ctx, musaStream_t stream) {
  const Tensor* X = ctx->Input<Tensor>(0);
  if (!X) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input tensor");
  }

  Tensor* Y = ctx->Output(0, X->Shape());
  if (!Y) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor");
  }

  const int64_t count = X->Shape().Size();
  if (count == 0) {
    return Status::OK();
  }

  if (!X->DataRaw() || !Y->MutableDataRaw()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid tensor data pointers");
  }

  const void* device_input = nullptr;
  void* temp_input = nullptr;
  ORT_RETURN_IF_ERROR(CopyHostInputIfNeeded<T>(X->DataRaw(), count, stream, &device_input, &temp_input));

  LaunchBitwiseNotKernel<T>(stream, reinterpret_cast<const T*>(device_input), reinterpret_cast<T*>(Y->MutableDataRaw()), count);
  MUSA_RETURN_IF_ERROR(musaGetLastError());

  if (temp_input) {
    MUSA_RETURN_IF_ERROR(musaStreamSynchronize(stream));
  }
  FreeTempBuffer(temp_input);
  return Status::OK();
}

}  // namespace

template <typename T>
Status BitwiseOr<T>::ComputeInternal(OpKernelContext* ctx) const {
  return RunBitwiseBinary<T>(ctx, "BitwiseOr", this->Stream(ctx), [](musaStream_t stream, const T* lhs, const T* rhs, T* output, const BitwiseExtraBinaryParams& params) {
    LaunchBitwiseOrKernel<T>(stream, lhs, rhs, output, params);
  });
}

template <typename T>
Status BitwiseXor<T>::ComputeInternal(OpKernelContext* ctx) const {
  return RunBitwiseBinary<T>(ctx, "BitwiseXor", this->Stream(ctx), [](musaStream_t stream, const T* lhs, const T* rhs, T* output, const BitwiseExtraBinaryParams& params) {
    LaunchBitwiseXorKernel<T>(stream, lhs, rhs, output, params);
  });
}

template <typename T>
Status BitwiseNot<T>::ComputeInternal(OpKernelContext* ctx) const {
  return RunBitwiseNot<T>(ctx, this->Stream(ctx));
}

#define INSTANTIATE_BITWISE_EXTRA_OPS(T) \
  template class BitwiseOr<T>;           \
  template class BitwiseXor<T>;          \
  template class BitwiseNot<T>;

INSTANTIATE_BITWISE_EXTRA_OPS(int8_t)
INSTANTIATE_BITWISE_EXTRA_OPS(uint8_t)
INSTANTIATE_BITWISE_EXTRA_OPS(int16_t)
INSTANTIATE_BITWISE_EXTRA_OPS(uint16_t)
INSTANTIATE_BITWISE_EXTRA_OPS(int32_t)
INSTANTIATE_BITWISE_EXTRA_OPS(uint32_t)
INSTANTIATE_BITWISE_EXTRA_OPS(int64_t)
INSTANTIATE_BITWISE_EXTRA_OPS(uint64_t)

#undef INSTANTIATE_BITWISE_EXTRA_OPS

#define REGISTER_MUSA_VERSIONED_BITWISE_EXTRA_BINARY_KERNEL(OpName, KernelClass, T)         \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                  \
      OpName, kOnnxDomain, 1, 17, T, kMusaExecutionProvider,                                \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      KernelClass<T>);

#define REGISTER_MUSA_VERSIONED_BITWISE_EXTRA_NOT_KERNEL(T)                                 \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                  \
      BitwiseNot, kOnnxDomain, 1, 17, T, kMusaExecutionProvider,                            \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      BitwiseNot<T>);

#define REGISTER_MUSA_VERSIONED_BITWISE_EXTRA_FOR_TYPE(T)                         \
  REGISTER_MUSA_VERSIONED_BITWISE_EXTRA_BINARY_KERNEL(BitwiseOr, BitwiseOr, T)    \
  REGISTER_MUSA_VERSIONED_BITWISE_EXTRA_BINARY_KERNEL(BitwiseXor, BitwiseXor, T)  \
  REGISTER_MUSA_VERSIONED_BITWISE_EXTRA_NOT_KERNEL(T)

#define REGISTER_MUSA_BITWISE_EXTRA_BINARY_KERNEL(OpName, KernelClass, T)                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      OpName, kOnnxDomain, 18, T, kMusaExecutionProvider,                                  \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      KernelClass<T>);

#define REGISTER_MUSA_BITWISE_EXTRA_NOT_KERNEL(T)                                          \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      BitwiseNot, kOnnxDomain, 18, T, kMusaExecutionProvider,                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      BitwiseNot<T>);

#define REGISTER_MUSA_BITWISE_EXTRA_FOR_TYPE(T)                        \
  REGISTER_MUSA_BITWISE_EXTRA_BINARY_KERNEL(BitwiseOr, BitwiseOr, T)   \
  REGISTER_MUSA_BITWISE_EXTRA_BINARY_KERNEL(BitwiseXor, BitwiseXor, T) \
  REGISTER_MUSA_BITWISE_EXTRA_NOT_KERNEL(T)

REGISTER_MUSA_VERSIONED_BITWISE_EXTRA_FOR_TYPE(int8_t)
REGISTER_MUSA_VERSIONED_BITWISE_EXTRA_FOR_TYPE(uint8_t)
REGISTER_MUSA_VERSIONED_BITWISE_EXTRA_FOR_TYPE(int16_t)
REGISTER_MUSA_VERSIONED_BITWISE_EXTRA_FOR_TYPE(uint16_t)
REGISTER_MUSA_VERSIONED_BITWISE_EXTRA_FOR_TYPE(int32_t)
REGISTER_MUSA_VERSIONED_BITWISE_EXTRA_FOR_TYPE(uint32_t)
REGISTER_MUSA_VERSIONED_BITWISE_EXTRA_FOR_TYPE(int64_t)
REGISTER_MUSA_VERSIONED_BITWISE_EXTRA_FOR_TYPE(uint64_t)

REGISTER_MUSA_BITWISE_EXTRA_FOR_TYPE(int8_t)
REGISTER_MUSA_BITWISE_EXTRA_FOR_TYPE(uint8_t)
REGISTER_MUSA_BITWISE_EXTRA_FOR_TYPE(int16_t)
REGISTER_MUSA_BITWISE_EXTRA_FOR_TYPE(uint16_t)
REGISTER_MUSA_BITWISE_EXTRA_FOR_TYPE(int32_t)
REGISTER_MUSA_BITWISE_EXTRA_FOR_TYPE(uint32_t)
REGISTER_MUSA_BITWISE_EXTRA_FOR_TYPE(int64_t)
REGISTER_MUSA_BITWISE_EXTRA_FOR_TYPE(uint64_t)

#undef REGISTER_MUSA_VERSIONED_BITWISE_EXTRA_BINARY_KERNEL
#undef REGISTER_MUSA_VERSIONED_BITWISE_EXTRA_NOT_KERNEL
#undef REGISTER_MUSA_VERSIONED_BITWISE_EXTRA_FOR_TYPE
#undef REGISTER_MUSA_BITWISE_EXTRA_BINARY_KERNEL
#undef REGISTER_MUSA_BITWISE_EXTRA_NOT_KERNEL
#undef REGISTER_MUSA_BITWISE_EXTRA_FOR_TYPE

}  // namespace musa
}  // namespace onnxruntime
