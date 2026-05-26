// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/math/sqrt.h"
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/common.h"
#include "core/providers/musa/math/unary_elementwise_safe_impl.h"
#include "core/providers/musa/musa_call.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/musa_execution_provider.h"
#include <algorithm>
#include <musa_runtime.h>
#include <type_traits>
#include <vector>

using onnxruntime::common::Status;
namespace onnxruntime {
namespace musa {

namespace {

void FillUnaryStrides(const std::vector<int64_t>& dims,
                      int64_t* strides) {
  int64_t running_stride = 1;
  for (int64_t dim = static_cast<int64_t>(dims.size()) - 1; dim >= 0; --dim) {
    strides[dim] = running_stride;
    running_stride *= dims[dim];
  }
}

Status BuildSqrtSameTypeParams(const TensorShape& input_shape,
                               const TensorShape& output_shape,
                               UnarySameTypeParams* params) {
  ORT_RETURN_IF_NOT(params != nullptr, "UnarySameTypeParams must not be null");

  const int32_t rank = static_cast<int32_t>(output_shape.NumDimensions());
  ORT_RETURN_IF_NOT(rank <= kUnarySameTypeMaxDims,
                    "Sqrt safe kernel supports rank up to ", kUnarySameTypeMaxDims,
                    ", got ", rank);

  params->rank = rank;
  params->total_elements = output_shape.Size();
  std::fill_n(params->input_strides, kUnarySameTypeMaxDims, 0);
  std::fill_n(params->output_strides, kUnarySameTypeMaxDims, 0);

  if (rank == 0 || params->total_elements == 0) {
    return Status::OK();
  }

  const auto input_dims = input_shape.GetDims();
  const auto output_dims = output_shape.GetDims();
  std::vector<int64_t> padded_input(rank, 1);
  std::vector<int64_t> padded_output(rank, 1);
  std::copy(input_dims.begin(), input_dims.end(), padded_input.begin());
  std::copy(output_dims.begin(), output_dims.end(), padded_output.begin());

  FillUnaryStrides(padded_input, params->input_strides);
  FillUnaryStrides(padded_output, params->output_strides);
  return Status::OK();
}

}  // namespace

// MUSA device-based sqrt implementation using self-written same-type kernel.
template <typename T>
Status SimpleMusaSqrtOp(const MusaPreparation& prepare, musaStream_t stream) {
  if (prepare.output_size == 0) {
    return Status::OK();
  }

  if (!prepare.input_a_ptr || !prepare.output_ptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid prepared tensor pointers");
  }

  UnarySameTypeParams params{};
  ORT_RETURN_IF_ERROR(BuildSqrtSameTypeParams(prepare.input_a_shape,
                                              prepare.output_shape,
                                              &params));

  if constexpr (std::is_same_v<T, MLFloat16>) {
    LaunchSqrtSameTypeKernelHalf(stream, prepare.input_a_ptr, prepare.output_ptr, params);
  } else if constexpr (std::is_same_v<T, BFloat16>) {
    LaunchSqrtSameTypeKernelBFloat16(stream, prepare.input_a_ptr, prepare.output_ptr, params);
  } else {
    LaunchSqrtSameTypeKernel<T>(stream,
                                reinterpret_cast<const T*>(prepare.input_a_ptr),
                                reinterpret_cast<T*>(prepare.output_ptr),
                                params);
  }

  MUSA_RETURN_IF_ERROR(musaGetLastError());
  return Status::OK();
}

// Prepare method for self-written Sqrt kernel.
template <typename T>
Status Sqrt<T>::Prepare(OpKernelContext* ctx, MusaPreparation& prepare) const {
  // 1. Get input tensor
  const Tensor* X = ctx->Input<Tensor>(0);
  if (X == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid input tensor");
  }

  // 2. Create output tensor with same shape as input
  Tensor* Y = ctx->Output(0, X->Shape());
  if (Y == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor");
  }

  // 3. Store tensor pointers and shapes in preparation
  prepare.input_a_ptr = X->DataRaw();
  prepare.output_ptr = Y->MutableDataRaw();
  prepare.output_size = Y->Shape().Size();
  prepare.input_a_shape = X->Shape();
  prepare.output_shape = Y->Shape();

  if (prepare.output_size > 0 && (prepare.input_a_ptr == nullptr || prepare.output_ptr == nullptr)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid tensor data pointers");
  }

  return Status::OK();
}

// ComputeInternal method
template <typename T>
Status Sqrt<T>::ComputeInternal(OpKernelContext* ctx) const {
  const auto* ep = static_cast<const MusaExecutionProvider*>(
      Info().GetExecutionProvider());
  MusaPreparation prepare(ep);
  ORT_RETURN_IF_ERROR(this->Prepare(ctx, prepare));
  ORT_RETURN_IF_ERROR(SimpleMusaSqrtOp<T>(prepare, Stream(ctx)));
  return Status::OK();
}

// Explicit template instantiations for supported types
template class Sqrt<float>;
template class Sqrt<MLFloat16>;
template class Sqrt<double>;
template class Sqrt<BFloat16>;

// Register kernels for different ONNX versions
#define REGISTER_MUSA_SQRT_TYPED_KERNEL(ver, T)                          \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      Sqrt, kOnnxDomain, ver, T, kMusaExecutionProvider,                 \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),       \
      Sqrt<T>);

#define REGISTER_MUSA_SQRT_VERSIONED_TYPED_KERNEL(startver, endver, T)   \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                              \
      Sqrt, kOnnxDomain, startver, endver, T, kMusaExecutionProvider,    \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),       \
      Sqrt<T>);

// Register versions 6-12
REGISTER_MUSA_SQRT_VERSIONED_TYPED_KERNEL(6, 12, float)
REGISTER_MUSA_SQRT_VERSIONED_TYPED_KERNEL(6, 12, MLFloat16)
REGISTER_MUSA_SQRT_VERSIONED_TYPED_KERNEL(6, 12, double)

// Register version 13+ (current) - adds BFloat16 support
REGISTER_MUSA_SQRT_TYPED_KERNEL(13, float)
REGISTER_MUSA_SQRT_TYPED_KERNEL(13, MLFloat16)
REGISTER_MUSA_SQRT_TYPED_KERNEL(13, double)
REGISTER_MUSA_SQRT_TYPED_KERNEL(13, BFloat16)

}  // namespace musa
}  // namespace onnxruntime
