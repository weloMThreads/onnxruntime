// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/math/unary_extra_ops_impl.h"

#include <math.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <stdint.h>

namespace onnxruntime {
namespace musa {
namespace {

constexpr int kUnaryExtraThreadsPerBlock = 256;
constexpr int kUnaryExtraMaxBlocks = 4096;

inline int BlocksForCount(int64_t count) {
  int64_t blocks = (count + kUnaryExtraThreadsPerBlock - 1) / kUnaryExtraThreadsPerBlock;
  if (blocks > kUnaryExtraMaxBlocks) {
    blocks = kUnaryExtraMaxBlocks;
  }
  return static_cast<int>(blocks);
}

template <typename T>
__device__ __forceinline__ T SignValue(T value) {
  if (value > static_cast<T>(0)) return static_cast<T>(1);
  if (value < static_cast<T>(0)) return static_cast<T>(-1);
  return static_cast<T>(0);
}

__device__ __forceinline__ half SignHalfValue(half value) {
  const float f = __half2float(value);
  float result = 0.0f;
  if (f > 0.0f) {
    result = 1.0f;
  } else if (f < 0.0f) {
    result = -1.0f;
  }
  return __float2half(result);
}

template <typename T>
__global__ void ReciprocalKernel(const T* input, T* output, int64_t count) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    output[index] = static_cast<T>(1) / input[index];
  }
}

__global__ void ReciprocalHalfKernel(const half* input, half* output, int64_t count) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    output[index] = __float2half(1.0f / __half2float(input[index]));
  }
}

template <typename T>
__global__ void Log1pKernel(const T* input, T* output, int64_t count) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    output[index] = log1p(input[index]);
  }
}

__global__ void Log1pHalfKernel(const half* input, half* output, int64_t count) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    output[index] = __float2half(log1pf(__half2float(input[index])));
  }
}

template <typename T>
__global__ void Expm1Kernel(const T* input, T* output, int64_t count) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    output[index] = expm1(input[index]);
  }
}

__global__ void Expm1HalfKernel(const half* input, half* output, int64_t count) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    output[index] = __float2half(expm1f(__half2float(input[index])));
  }
}

template <typename T>
__global__ void SquareKernel(const T* input, T* output, int64_t count) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    output[index] = input[index] * input[index];
  }
}

__global__ void SquareHalfKernel(const half* input, half* output, int64_t count) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    const float value = __half2float(input[index]);
    output[index] = __float2half(value * value);
  }
}

template <typename T>
__global__ void RsqrtKernel(const T* input, T* output, int64_t count) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    output[index] = static_cast<T>(1) / sqrt(input[index]);
  }
}

__global__ void RsqrtHalfKernel(const half* input, half* output, int64_t count) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    output[index] = __float2half(rsqrtf(__half2float(input[index])));
  }
}

template <typename T>
__global__ void FloorKernel(const T* input, T* output, int64_t count) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    output[index] = floor(input[index]);
  }
}

__global__ void FloorHalfKernel(const half* input, half* output, int64_t count) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    output[index] = __float2half(floorf(__half2float(input[index])));
  }
}

template <typename T>
__global__ void CeilKernel(const T* input, T* output, int64_t count) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    output[index] = ceil(input[index]);
  }
}

__global__ void CeilHalfKernel(const half* input, half* output, int64_t count) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    output[index] = __float2half(ceilf(__half2float(input[index])));
  }
}

template <typename T>
__global__ void SignKernel(const T* input, T* output, int64_t count) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    output[index] = SignValue(input[index]);
  }
}

__global__ void SignHalfKernel(const half* input, half* output, int64_t count) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    output[index] = SignHalfValue(input[index]);
  }
}

template <typename T>
__global__ void IsNaNKernel(const T* input, bool* output, int64_t count) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    output[index] = isnan(input[index]);
  }
}

__global__ void IsNaNHalfKernel(const half* input, bool* output, int64_t count) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    output[index] = isnan(__half2float(input[index]));
  }
}

}  // namespace

template <typename T>
void LaunchReciprocalKernel(musaStream_t stream, const T* input, T* output, int64_t count) {
  if (count > 0) ReciprocalKernel<T><<<BlocksForCount(count), kUnaryExtraThreadsPerBlock, 0, stream>>>(input, output, count);
}

void LaunchReciprocalKernelHalf(musaStream_t stream, const void* input, void* output, int64_t count) {
  if (count > 0) ReciprocalHalfKernel<<<BlocksForCount(count), kUnaryExtraThreadsPerBlock, 0, stream>>>(static_cast<const half*>(input), static_cast<half*>(output), count);
}

template <typename T>
void LaunchLog1pKernel(musaStream_t stream, const T* input, T* output, int64_t count) {
  if (count > 0) Log1pKernel<T><<<BlocksForCount(count), kUnaryExtraThreadsPerBlock, 0, stream>>>(input, output, count);
}

void LaunchLog1pKernelHalf(musaStream_t stream, const void* input, void* output, int64_t count) {
  if (count > 0) Log1pHalfKernel<<<BlocksForCount(count), kUnaryExtraThreadsPerBlock, 0, stream>>>(static_cast<const half*>(input), static_cast<half*>(output), count);
}

template <typename T>
void LaunchExpm1Kernel(musaStream_t stream, const T* input, T* output, int64_t count) {
  if (count > 0) Expm1Kernel<T><<<BlocksForCount(count), kUnaryExtraThreadsPerBlock, 0, stream>>>(input, output, count);
}

void LaunchExpm1KernelHalf(musaStream_t stream, const void* input, void* output, int64_t count) {
  if (count > 0) Expm1HalfKernel<<<BlocksForCount(count), kUnaryExtraThreadsPerBlock, 0, stream>>>(static_cast<const half*>(input), static_cast<half*>(output), count);
}

template <typename T>
void LaunchSquareKernel(musaStream_t stream, const T* input, T* output, int64_t count) {
  if (count > 0) SquareKernel<T><<<BlocksForCount(count), kUnaryExtraThreadsPerBlock, 0, stream>>>(input, output, count);
}

void LaunchSquareKernelHalf(musaStream_t stream, const void* input, void* output, int64_t count) {
  if (count > 0) SquareHalfKernel<<<BlocksForCount(count), kUnaryExtraThreadsPerBlock, 0, stream>>>(static_cast<const half*>(input), static_cast<half*>(output), count);
}

template <typename T>
void LaunchRsqrtKernel(musaStream_t stream, const T* input, T* output, int64_t count) {
  if (count > 0) RsqrtKernel<T><<<BlocksForCount(count), kUnaryExtraThreadsPerBlock, 0, stream>>>(input, output, count);
}

void LaunchRsqrtKernelHalf(musaStream_t stream, const void* input, void* output, int64_t count) {
  if (count > 0) RsqrtHalfKernel<<<BlocksForCount(count), kUnaryExtraThreadsPerBlock, 0, stream>>>(static_cast<const half*>(input), static_cast<half*>(output), count);
}

template <typename T>
void LaunchFloorKernel(musaStream_t stream, const T* input, T* output, int64_t count) {
  if (count > 0) FloorKernel<T><<<BlocksForCount(count), kUnaryExtraThreadsPerBlock, 0, stream>>>(input, output, count);
}

void LaunchFloorKernelHalf(musaStream_t stream, const void* input, void* output, int64_t count) {
  if (count > 0) FloorHalfKernel<<<BlocksForCount(count), kUnaryExtraThreadsPerBlock, 0, stream>>>(static_cast<const half*>(input), static_cast<half*>(output), count);
}

template <typename T>
void LaunchCeilKernel(musaStream_t stream, const T* input, T* output, int64_t count) {
  if (count > 0) CeilKernel<T><<<BlocksForCount(count), kUnaryExtraThreadsPerBlock, 0, stream>>>(input, output, count);
}

void LaunchCeilKernelHalf(musaStream_t stream, const void* input, void* output, int64_t count) {
  if (count > 0) CeilHalfKernel<<<BlocksForCount(count), kUnaryExtraThreadsPerBlock, 0, stream>>>(static_cast<const half*>(input), static_cast<half*>(output), count);
}

template <typename T>
void LaunchSignKernel(musaStream_t stream, const T* input, T* output, int64_t count) {
  if (count > 0) SignKernel<T><<<BlocksForCount(count), kUnaryExtraThreadsPerBlock, 0, stream>>>(input, output, count);
}

void LaunchSignKernelHalf(musaStream_t stream, const void* input, void* output, int64_t count) {
  if (count > 0) SignHalfKernel<<<BlocksForCount(count), kUnaryExtraThreadsPerBlock, 0, stream>>>(static_cast<const half*>(input), static_cast<half*>(output), count);
}

template <typename T>
void LaunchIsNaNKernel(musaStream_t stream, const T* input, bool* output, int64_t count) {
  if (count > 0) IsNaNKernel<T><<<BlocksForCount(count), kUnaryExtraThreadsPerBlock, 0, stream>>>(input, output, count);
}

void LaunchIsNaNKernelHalf(musaStream_t stream, const void* input, bool* output, int64_t count) {
  if (count > 0) IsNaNHalfKernel<<<BlocksForCount(count), kUnaryExtraThreadsPerBlock, 0, stream>>>(static_cast<const half*>(input), output, count);
}

template void LaunchReciprocalKernel<float>(musaStream_t, const float*, float*, int64_t);
template void LaunchReciprocalKernel<double>(musaStream_t, const double*, double*, int64_t);
template void LaunchLog1pKernel<float>(musaStream_t, const float*, float*, int64_t);
template void LaunchLog1pKernel<double>(musaStream_t, const double*, double*, int64_t);
template void LaunchExpm1Kernel<float>(musaStream_t, const float*, float*, int64_t);
template void LaunchExpm1Kernel<double>(musaStream_t, const double*, double*, int64_t);
template void LaunchSquareKernel<float>(musaStream_t, const float*, float*, int64_t);
template void LaunchSquareKernel<double>(musaStream_t, const double*, double*, int64_t);
template void LaunchSquareKernel<int32_t>(musaStream_t, const int32_t*, int32_t*, int64_t);
template void LaunchSquareKernel<int64_t>(musaStream_t, const int64_t*, int64_t*, int64_t);
template void LaunchRsqrtKernel<float>(musaStream_t, const float*, float*, int64_t);
template void LaunchRsqrtKernel<double>(musaStream_t, const double*, double*, int64_t);
template void LaunchFloorKernel<float>(musaStream_t, const float*, float*, int64_t);
template void LaunchFloorKernel<double>(musaStream_t, const double*, double*, int64_t);
template void LaunchCeilKernel<float>(musaStream_t, const float*, float*, int64_t);
template void LaunchCeilKernel<double>(musaStream_t, const double*, double*, int64_t);
template void LaunchSignKernel<float>(musaStream_t, const float*, float*, int64_t);
template void LaunchSignKernel<double>(musaStream_t, const double*, double*, int64_t);
template void LaunchSignKernel<int32_t>(musaStream_t, const int32_t*, int32_t*, int64_t);
template void LaunchSignKernel<int64_t>(musaStream_t, const int64_t*, int64_t*, int64_t);
template void LaunchIsNaNKernel<float>(musaStream_t, const float*, bool*, int64_t);
template void LaunchIsNaNKernel<double>(musaStream_t, const double*, bool*, int64_t);

}  // namespace musa
}  // namespace onnxruntime
