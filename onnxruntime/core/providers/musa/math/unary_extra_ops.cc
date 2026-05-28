// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/math/unary_extra_ops.h"

#include "core/providers/common.h"
#include "core/providers/musa/math/unary_extra_ops_impl.h"
#include "core/providers/musa/musa_call.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/shared_library/provider_api.h"

#include <musa_runtime.h>

#include <type_traits>

using onnxruntime::common::Status;

namespace onnxruntime {
namespace musa {
namespace {

bool IsHostPointer(const void* ptr) {
  if (ptr == nullptr) return false;
  musaPointerAttributes attr;
  auto err = musaPointerGetAttributes(&attr, ptr);
  if (err != musaSuccess) {
    musaGetLastError();
    return true;
  }
  return attr.type == musaMemoryTypeHost || attr.type == musaMemoryTypeUnregistered;
}

template <typename T>
Status CopyHostInputIfNeeded(const void* input, int64_t count, musaStream_t stream, const void** device_input, void** temp_buffer) {
  *device_input = input;
  *temp_buffer = nullptr;
  if (!IsHostPointer(input)) return Status::OK();
  const size_t bytes = static_cast<size_t>(count) * sizeof(T);
  if (bytes == 0) return Status::OK();
  MUSA_RETURN_IF_ERROR(musaMalloc(temp_buffer, bytes));
  MUSA_RETURN_IF_ERROR(musaMemcpyAsync(*temp_buffer, input, bytes, musaMemcpyHostToDevice, stream));
  *device_input = *temp_buffer;
  return Status::OK();
}

void FreeTempBuffer(void* ptr) {
  if (ptr) musaFree(ptr);
}

template <typename T, typename LaunchFn>
Status RunSameTypeUnary(OpKernelContext* ctx, musaStream_t stream, LaunchFn launch_fn) {
  const Tensor* X = ctx->Input<Tensor>(0);
  if (!X) return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input tensor");
  Tensor* Y = ctx->Output(0, X->Shape());
  if (!Y) return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor");
  const int64_t count = X->Shape().Size();
  if (count == 0) return Status::OK();
  if (!X->DataRaw() || !Y->MutableDataRaw()) return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid tensor data pointers");

  const void* device_input = nullptr;
  void* temp_input = nullptr;
  ORT_RETURN_IF_ERROR(CopyHostInputIfNeeded<T>(X->DataRaw(), count, stream, &device_input, &temp_input));
  launch_fn(stream, device_input, Y->MutableDataRaw(), count);
  MUSA_RETURN_IF_ERROR(musaGetLastError());
  if (temp_input) MUSA_RETURN_IF_ERROR(musaStreamSynchronize(stream));
  FreeTempBuffer(temp_input);
  return Status::OK();
}

template <typename T, typename LaunchFn>
Status RunBoolOutputUnary(OpKernelContext* ctx, musaStream_t stream, LaunchFn launch_fn) {
  const Tensor* X = ctx->Input<Tensor>(0);
  if (!X) return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input tensor");
  Tensor* Y = ctx->Output(0, X->Shape());
  if (!Y) return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor");
  const int64_t count = X->Shape().Size();
  if (count == 0) return Status::OK();
  if (!X->DataRaw() || !Y->MutableDataRaw()) return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid tensor data pointers");

  const void* device_input = nullptr;
  void* temp_input = nullptr;
  ORT_RETURN_IF_ERROR(CopyHostInputIfNeeded<T>(X->DataRaw(), count, stream, &device_input, &temp_input));
  launch_fn(stream, device_input, reinterpret_cast<bool*>(Y->MutableDataRaw()), count);
  MUSA_RETURN_IF_ERROR(musaGetLastError());
  if (temp_input) MUSA_RETURN_IF_ERROR(musaStreamSynchronize(stream));
  FreeTempBuffer(temp_input);
  return Status::OK();
}

}  // namespace

template <typename T>
Status Reciprocal<T>::ComputeInternal(OpKernelContext* ctx) const {
  return RunSameTypeUnary<T>(ctx, this->Stream(ctx), [](musaStream_t stream, const void* input, void* output, int64_t count) {
    if constexpr (std::is_same_v<T, MLFloat16>) {
      LaunchReciprocalKernelHalf(stream, input, output, count);
    } else {
      LaunchReciprocalKernel<T>(stream, reinterpret_cast<const T*>(input), reinterpret_cast<T*>(output), count);
    }
  });
}

template <typename T>
Status Log1p<T>::ComputeInternal(OpKernelContext* ctx) const {
  return RunSameTypeUnary<T>(ctx, this->Stream(ctx), [](musaStream_t stream, const void* input, void* output, int64_t count) {
    if constexpr (std::is_same_v<T, MLFloat16>) {
      LaunchLog1pKernelHalf(stream, input, output, count);
    } else {
      LaunchLog1pKernel<T>(stream, reinterpret_cast<const T*>(input), reinterpret_cast<T*>(output), count);
    }
  });
}

template <typename T>
Status Expm1<T>::ComputeInternal(OpKernelContext* ctx) const {
  return RunSameTypeUnary<T>(ctx, this->Stream(ctx), [](musaStream_t stream, const void* input, void* output, int64_t count) {
    if constexpr (std::is_same_v<T, MLFloat16>) {
      LaunchExpm1KernelHalf(stream, input, output, count);
    } else {
      LaunchExpm1Kernel<T>(stream, reinterpret_cast<const T*>(input), reinterpret_cast<T*>(output), count);
    }
  });
}

template <typename T>
Status Square<T>::ComputeInternal(OpKernelContext* ctx) const {
  return RunSameTypeUnary<T>(ctx, this->Stream(ctx), [](musaStream_t stream, const void* input, void* output, int64_t count) {
    if constexpr (std::is_same_v<T, MLFloat16>) {
      LaunchSquareKernelHalf(stream, input, output, count);
    } else {
      LaunchSquareKernel<T>(stream, reinterpret_cast<const T*>(input), reinterpret_cast<T*>(output), count);
    }
  });
}

template <typename T>
Status Rsqrt<T>::ComputeInternal(OpKernelContext* ctx) const {
  return RunSameTypeUnary<T>(ctx, this->Stream(ctx), [](musaStream_t stream, const void* input, void* output, int64_t count) {
    if constexpr (std::is_same_v<T, MLFloat16>) {
      LaunchRsqrtKernelHalf(stream, input, output, count);
    } else {
      LaunchRsqrtKernel<T>(stream, reinterpret_cast<const T*>(input), reinterpret_cast<T*>(output), count);
    }
  });
}

template <typename T>
Status Floor<T>::ComputeInternal(OpKernelContext* ctx) const {
  return RunSameTypeUnary<T>(ctx, this->Stream(ctx), [](musaStream_t stream, const void* input, void* output, int64_t count) {
    if constexpr (std::is_same_v<T, MLFloat16>) {
      LaunchFloorKernelHalf(stream, input, output, count);
    } else {
      LaunchFloorKernel<T>(stream, reinterpret_cast<const T*>(input), reinterpret_cast<T*>(output), count);
    }
  });
}

template <typename T>
Status Ceil<T>::ComputeInternal(OpKernelContext* ctx) const {
  return RunSameTypeUnary<T>(ctx, this->Stream(ctx), [](musaStream_t stream, const void* input, void* output, int64_t count) {
    if constexpr (std::is_same_v<T, MLFloat16>) {
      LaunchCeilKernelHalf(stream, input, output, count);
    } else {
      LaunchCeilKernel<T>(stream, reinterpret_cast<const T*>(input), reinterpret_cast<T*>(output), count);
    }
  });
}

template <typename T>
Status ZerosLike<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* X = ctx->Input<Tensor>(0);
  if (!X) return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input tensor");
  Tensor* Y = ctx->Output(0, X->Shape());
  if (!Y) return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor");
  const int64_t count = Y->Shape().Size();
  if (count == 0) return Status::OK();
  MUSA_RETURN_IF_ERROR(musaMemsetAsync(Y->MutableDataRaw(), 0, static_cast<size_t>(count) * sizeof(T), this->Stream(ctx)));
  return Status::OK();
}

template <typename T>
Status Sign<T>::ComputeInternal(OpKernelContext* ctx) const {
  return RunSameTypeUnary<T>(ctx, this->Stream(ctx), [](musaStream_t stream, const void* input, void* output, int64_t count) {
    if constexpr (std::is_same_v<T, MLFloat16>) {
      LaunchSignKernelHalf(stream, input, output, count);
    } else {
      LaunchSignKernel<T>(stream, reinterpret_cast<const T*>(input), reinterpret_cast<T*>(output), count);
    }
  });
}

template <typename T>
Status IsNaN<T>::ComputeInternal(OpKernelContext* ctx) const {
  return RunBoolOutputUnary<T>(ctx, this->Stream(ctx), [](musaStream_t stream, const void* input, bool* output, int64_t count) {
    if constexpr (std::is_same_v<T, MLFloat16>) {
      LaunchIsNaNKernelHalf(stream, input, output, count);
    } else {
      LaunchIsNaNKernel<T>(stream, reinterpret_cast<const T*>(input), output, count);
    }
  });
}

template class Reciprocal<float>;
template class Reciprocal<double>;
template class Reciprocal<MLFloat16>;
template class Log1p<float>;
template class Log1p<double>;
template class Log1p<MLFloat16>;
template class Expm1<float>;
template class Expm1<double>;
template class Expm1<MLFloat16>;
template class Square<float>;
template class Square<double>;
template class Square<int32_t>;
template class Square<int64_t>;
template class Square<MLFloat16>;
template class Rsqrt<float>;
template class Rsqrt<double>;
template class Rsqrt<MLFloat16>;
template class Floor<float>;
template class Floor<double>;
template class Floor<MLFloat16>;
template class Ceil<float>;
template class Ceil<double>;
template class Ceil<MLFloat16>;
template class ZerosLike<bool>;
template class ZerosLike<float>;
template class ZerosLike<double>;
template class ZerosLike<int32_t>;
template class ZerosLike<int64_t>;
template class ZerosLike<MLFloat16>;
template class Sign<float>;
template class Sign<double>;
template class Sign<MLFloat16>;
template class Sign<int32_t>;
template class Sign<int64_t>;
template class IsNaN<float>;
template class IsNaN<double>;
template class IsNaN<MLFloat16>;

#define REGISTER_MUSA_RECIPROCAL_VERSIONED_TYPED_KERNEL(startver, endver, T)                                    \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(Reciprocal, kOnnxDomain, startver, endver, T, kMusaExecutionProvider, \
                                          (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), Reciprocal<T>);
#define REGISTER_MUSA_RECIPROCAL_TYPED_KERNEL(ver, T)                                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(Reciprocal, kOnnxDomain, ver, T, kMusaExecutionProvider, \
                                (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), Reciprocal<T>);
#define REGISTER_MUSA_LOG1P_TYPED_KERNEL(ver, T)                                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(Log1p, kOnnxDomain, ver, T, kMusaExecutionProvider, \
                                (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), Log1p<T>);
#define REGISTER_MUSA_EXPM1_TYPED_KERNEL(ver, T)                                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(Expm1, kOnnxDomain, ver, T, kMusaExecutionProvider, \
                                (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), Expm1<T>);
#define REGISTER_MUSA_SQUARE_TYPED_KERNEL(ver, T)                                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(Square, kOnnxDomain, ver, T, kMusaExecutionProvider, \
                                (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), Square<T>);
#define REGISTER_MUSA_RSQRT_TYPED_KERNEL(ver, T)                                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(Rsqrt, kOnnxDomain, ver, T, kMusaExecutionProvider, \
                                (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), Rsqrt<T>);
#define REGISTER_MUSA_FLOOR_VERSIONED_TYPED_KERNEL(startver, endver, T)                                    \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(Floor, kOnnxDomain, startver, endver, T, kMusaExecutionProvider, \
                                          (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), Floor<T>);
#define REGISTER_MUSA_FLOOR_TYPED_KERNEL(ver, T)                                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(Floor, kOnnxDomain, ver, T, kMusaExecutionProvider, \
                                (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), Floor<T>);
#define REGISTER_MUSA_CEIL_VERSIONED_TYPED_KERNEL(startver, endver, T)                                    \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(Ceil, kOnnxDomain, startver, endver, T, kMusaExecutionProvider, \
                                          (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), Ceil<T>);
#define REGISTER_MUSA_CEIL_TYPED_KERNEL(ver, T)                                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(Ceil, kOnnxDomain, ver, T, kMusaExecutionProvider, \
                                (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), Ceil<T>);
#define REGISTER_MUSA_ZEROSLIKE_TYPED_KERNEL(ver, T)                                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(ZerosLike, kOnnxDomain, ver, T, kMusaExecutionProvider, \
                                (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), ZerosLike<T>);
#define REGISTER_MUSA_SIGN_TYPED_KERNEL(ver, T)                                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(Sign, kOnnxDomain, ver, T, kMusaExecutionProvider, \
                                (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), Sign<T>);
#define REGISTER_MUSA_ISNAN_VERSIONED_TYPED_KERNEL(startver, endver, T)                                    \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(IsNaN, kOnnxDomain, startver, endver, T, kMusaExecutionProvider, \
                                          (*KernelDefBuilder::Create()).TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()), IsNaN<T>);
#define REGISTER_MUSA_ISNAN_TYPED_KERNEL(ver, T)                                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(IsNaN, kOnnxDomain, ver, T, kMusaExecutionProvider, \
                                (*KernelDefBuilder::Create()).TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()), IsNaN<T>);
#define REGISTER_MUSA_ISNAN_COMPAT_TYPED_KERNEL(ver, T)                            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(IsNan, kOnnxDomain, ver, T, kMusaExecutionProvider, \
                                (*KernelDefBuilder::Create()).TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()), IsNaN<T>);

REGISTER_MUSA_RECIPROCAL_VERSIONED_TYPED_KERNEL(6, 12, float)
REGISTER_MUSA_RECIPROCAL_VERSIONED_TYPED_KERNEL(6, 12, double)
REGISTER_MUSA_RECIPROCAL_VERSIONED_TYPED_KERNEL(6, 12, MLFloat16)
REGISTER_MUSA_RECIPROCAL_TYPED_KERNEL(13, float)
REGISTER_MUSA_RECIPROCAL_TYPED_KERNEL(13, double)
REGISTER_MUSA_RECIPROCAL_TYPED_KERNEL(13, MLFloat16)
REGISTER_MUSA_LOG1P_TYPED_KERNEL(1, float)
REGISTER_MUSA_LOG1P_TYPED_KERNEL(1, double)
REGISTER_MUSA_LOG1P_TYPED_KERNEL(1, MLFloat16)
REGISTER_MUSA_EXPM1_TYPED_KERNEL(1, float)
REGISTER_MUSA_EXPM1_TYPED_KERNEL(1, double)
REGISTER_MUSA_EXPM1_TYPED_KERNEL(1, MLFloat16)
REGISTER_MUSA_SQUARE_TYPED_KERNEL(1, float)
REGISTER_MUSA_SQUARE_TYPED_KERNEL(1, double)
REGISTER_MUSA_SQUARE_TYPED_KERNEL(1, int32_t)
REGISTER_MUSA_SQUARE_TYPED_KERNEL(1, int64_t)
REGISTER_MUSA_SQUARE_TYPED_KERNEL(1, MLFloat16)
REGISTER_MUSA_RSQRT_TYPED_KERNEL(1, float)
REGISTER_MUSA_RSQRT_TYPED_KERNEL(1, double)
REGISTER_MUSA_RSQRT_TYPED_KERNEL(1, MLFloat16)
REGISTER_MUSA_FLOOR_VERSIONED_TYPED_KERNEL(6, 12, float)
REGISTER_MUSA_FLOOR_VERSIONED_TYPED_KERNEL(6, 12, double)
REGISTER_MUSA_FLOOR_VERSIONED_TYPED_KERNEL(6, 12, MLFloat16)
REGISTER_MUSA_FLOOR_TYPED_KERNEL(13, float)
REGISTER_MUSA_FLOOR_TYPED_KERNEL(13, double)
REGISTER_MUSA_FLOOR_TYPED_KERNEL(13, MLFloat16)
REGISTER_MUSA_CEIL_VERSIONED_TYPED_KERNEL(6, 12, float)
REGISTER_MUSA_CEIL_VERSIONED_TYPED_KERNEL(6, 12, double)
REGISTER_MUSA_CEIL_VERSIONED_TYPED_KERNEL(6, 12, MLFloat16)
REGISTER_MUSA_CEIL_TYPED_KERNEL(13, float)
REGISTER_MUSA_CEIL_TYPED_KERNEL(13, double)
REGISTER_MUSA_CEIL_TYPED_KERNEL(13, MLFloat16)
REGISTER_MUSA_ZEROSLIKE_TYPED_KERNEL(1, bool)
REGISTER_MUSA_ZEROSLIKE_TYPED_KERNEL(1, float)
REGISTER_MUSA_ZEROSLIKE_TYPED_KERNEL(1, double)
REGISTER_MUSA_ZEROSLIKE_TYPED_KERNEL(1, int32_t)
REGISTER_MUSA_ZEROSLIKE_TYPED_KERNEL(1, int64_t)
REGISTER_MUSA_ZEROSLIKE_TYPED_KERNEL(1, MLFloat16)
REGISTER_MUSA_SIGN_TYPED_KERNEL(13, float)
REGISTER_MUSA_SIGN_TYPED_KERNEL(13, double)
REGISTER_MUSA_SIGN_TYPED_KERNEL(13, MLFloat16)
REGISTER_MUSA_SIGN_TYPED_KERNEL(13, int32_t)
REGISTER_MUSA_SIGN_TYPED_KERNEL(13, int64_t)
REGISTER_MUSA_ISNAN_VERSIONED_TYPED_KERNEL(9, 12, float)
REGISTER_MUSA_ISNAN_VERSIONED_TYPED_KERNEL(9, 12, double)
REGISTER_MUSA_ISNAN_VERSIONED_TYPED_KERNEL(9, 12, MLFloat16)
REGISTER_MUSA_ISNAN_VERSIONED_TYPED_KERNEL(13, 19, float)
REGISTER_MUSA_ISNAN_VERSIONED_TYPED_KERNEL(13, 19, double)
REGISTER_MUSA_ISNAN_VERSIONED_TYPED_KERNEL(13, 19, MLFloat16)
REGISTER_MUSA_ISNAN_TYPED_KERNEL(20, float)
REGISTER_MUSA_ISNAN_TYPED_KERNEL(20, double)
REGISTER_MUSA_ISNAN_TYPED_KERNEL(20, MLFloat16)
REGISTER_MUSA_ISNAN_COMPAT_TYPED_KERNEL(1, float)
REGISTER_MUSA_ISNAN_COMPAT_TYPED_KERNEL(1, double)
REGISTER_MUSA_ISNAN_COMPAT_TYPED_KERNEL(1, MLFloat16)

}  // namespace musa
}  // namespace onnxruntime
