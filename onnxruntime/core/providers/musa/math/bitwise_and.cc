// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/common.h"
#include "core/providers/musa/math/bitwise_and.h"
#include "core/providers/musa/math/bitwise_and_impl.h"
#include "core/providers/musa/musa_call.h"
#include "core/providers/musa/musa_fwd.h"

#include <algorithm>
#include <musa_runtime.h>
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

Status BuildBitwiseAndParams(const TensorShape& lhs_shape,
                             const TensorShape& rhs_shape,
                             const TensorShape& output_shape,
                             BitwiseAndParams* params) {
  ORT_RETURN_IF_NOT(params != nullptr, "BitwiseAndParams must not be null");

  const int32_t rank = static_cast<int32_t>(output_shape.NumDimensions());
  ORT_RETURN_IF_NOT(rank <= kBitwiseAndMaxDims,
                    "BitwiseAnd supports rank up to ", kBitwiseAndMaxDims,
                    ", got ", rank);

  params->rank = rank;
  params->total_elements = output_shape.Size();
  std::fill_n(params->output_strides, kBitwiseAndMaxDims, 0);
  std::fill_n(params->lhs_strides, kBitwiseAndMaxDims, 0);
  std::fill_n(params->rhs_strides, kBitwiseAndMaxDims, 0);

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
                             const TensorShape& shape,
                             musaStream_t stream,
                             const void** device_input,
                             void** temp_buffer) {
  *device_input = input;
  *temp_buffer = nullptr;
  if (!IsHostPointer(input)) {
    return Status::OK();
  }

  const size_t bytes = static_cast<size_t>(shape.Size()) * sizeof(T);
  if (bytes == 0) {
    return Status::OK();
  }

  MUSA_RETURN_IF_ERROR(musaMalloc(temp_buffer, bytes));
  MUSA_RETURN_IF_ERROR(musaMemcpyAsync(*temp_buffer, input, bytes, musaMemcpyHostToDevice, stream));
  *device_input = *temp_buffer;
  return Status::OK();
}

}  // namespace

template <typename T>
Status BitwiseAnd<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* A = ctx->Input<Tensor>(0);
  const Tensor* B = ctx->Input<Tensor>(1);
  if (!A || !B) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input tensors");
  }

  TensorShape output_shape;
  ORT_RETURN_IF_ERROR(ComputeBroadcastOutputShape(Node().Name(), A->Shape(), B->Shape(), output_shape));

  Tensor* C = ctx->Output(0, output_shape);
  if (!C) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor");
  }

  BitwiseAndParams params{};
  ORT_RETURN_IF_ERROR(BuildBitwiseAndParams(A->Shape(), B->Shape(), output_shape, &params));
  if (params.total_elements == 0) {
    return Status::OK();
  }

  if (!A->DataRaw() || !B->DataRaw() || !C->MutableDataRaw()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid tensor data pointers");
  }

  musaStream_t stream = Stream(ctx);
  const void* device_a = nullptr;
  const void* device_b = nullptr;
  void* temp_a = nullptr;
  void* temp_b = nullptr;

  auto cleanup = [&]() {
    if (temp_a) {
      musaFree(temp_a);
    }
    if (temp_b) {
      musaFree(temp_b);
    }
  };

  ORT_RETURN_IF_ERROR(CopyHostInputIfNeeded<T>(A->DataRaw(), A->Shape(), stream, &device_a, &temp_a));
  ORT_RETURN_IF_ERROR(CopyHostInputIfNeeded<T>(B->DataRaw(), B->Shape(), stream, &device_b, &temp_b));

  LaunchBitwiseAndKernel<T>(stream,
                            reinterpret_cast<const T*>(device_a),
                            reinterpret_cast<const T*>(device_b),
                            reinterpret_cast<T*>(C->MutableDataRaw()),
                            params);
  MUSA_RETURN_IF_ERROR(musaGetLastError());

  if (temp_a || temp_b) {
    MUSA_RETURN_IF_ERROR(musaStreamSynchronize(stream));
  }
  cleanup();
  return Status::OK();
}

template class BitwiseAnd<int8_t>;
template class BitwiseAnd<int16_t>;
template class BitwiseAnd<int32_t>;
template class BitwiseAnd<int64_t>;
template class BitwiseAnd<uint8_t>;
template class BitwiseAnd<uint16_t>;
template class BitwiseAnd<uint32_t>;
template class BitwiseAnd<uint64_t>;

#define REGISTER_MUSA_VERSIONED_BITWISE_AND_TYPED_KERNEL(T)                  \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                   \
      BitwiseAnd, kOnnxDomain, 1, 17, T, kMusaExecutionProvider,             \
      (*KernelDefBuilder::Create())                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),            \
      BitwiseAnd<T>);

#define REGISTER_MUSA_BITWISE_AND_TYPED_KERNEL(T)                            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                             \
      BitwiseAnd, kOnnxDomain, 18, T, kMusaExecutionProvider,                \
      (*KernelDefBuilder::Create())                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),            \
      BitwiseAnd<T>);

REGISTER_MUSA_VERSIONED_BITWISE_AND_TYPED_KERNEL(int8_t)
REGISTER_MUSA_VERSIONED_BITWISE_AND_TYPED_KERNEL(int16_t)
REGISTER_MUSA_VERSIONED_BITWISE_AND_TYPED_KERNEL(int32_t)
REGISTER_MUSA_VERSIONED_BITWISE_AND_TYPED_KERNEL(int64_t)
REGISTER_MUSA_VERSIONED_BITWISE_AND_TYPED_KERNEL(uint8_t)
REGISTER_MUSA_VERSIONED_BITWISE_AND_TYPED_KERNEL(uint16_t)
REGISTER_MUSA_VERSIONED_BITWISE_AND_TYPED_KERNEL(uint32_t)
REGISTER_MUSA_VERSIONED_BITWISE_AND_TYPED_KERNEL(uint64_t)

REGISTER_MUSA_BITWISE_AND_TYPED_KERNEL(int8_t)
REGISTER_MUSA_BITWISE_AND_TYPED_KERNEL(int16_t)
REGISTER_MUSA_BITWISE_AND_TYPED_KERNEL(int32_t)
REGISTER_MUSA_BITWISE_AND_TYPED_KERNEL(int64_t)
REGISTER_MUSA_BITWISE_AND_TYPED_KERNEL(uint8_t)
REGISTER_MUSA_BITWISE_AND_TYPED_KERNEL(uint16_t)
REGISTER_MUSA_BITWISE_AND_TYPED_KERNEL(uint32_t)
REGISTER_MUSA_BITWISE_AND_TYPED_KERNEL(uint64_t)

}  // namespace musa
}  // namespace onnxruntime
