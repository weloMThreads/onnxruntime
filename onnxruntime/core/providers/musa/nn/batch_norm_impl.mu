// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include <math.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <stdint.h>

#include "core/providers/musa/nn/batch_norm_impl.h"

namespace onnxruntime {
namespace musa {

namespace {

constexpr int kBatchNormThreadsPerBlock = 256;
constexpr int kBatchNormMaxBlocks = 4096;

template <typename T>
__device__ __forceinline__ float BatchNormToFloat(T value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float BatchNormToFloat<half>(half value) {
  return __half2float(value);
}

template <typename T>
__device__ __forceinline__ T BatchNormFromFloat(float value) {
  return static_cast<T>(value);
}

template <>
__device__ __forceinline__ half BatchNormFromFloat<half>(float value) {
  return __float2half(value);
}

template <typename T>
__global__ void BatchNormalizationKernel(const T* input,
                                         const T* scale,
                                         const T* bias,
                                         const T* mean,
                                         const T* variance,
                                         T* output,
                                         BatchNormalizationParams params) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t index = thread_id; index < params.total_elements; index += total_threads) {
    const int64_t channel = params.channels == 1 ? 0 : (index / params.spatial_size) % params.channels;
    const float x = BatchNormToFloat<T>(input[index]);
    const float s = BatchNormToFloat<T>(scale[channel]);
    const float b = BatchNormToFloat<T>(bias[channel]);
    const float m = BatchNormToFloat<T>(mean[channel]);
    const float v = BatchNormToFloat<T>(variance[channel]);
    const float y = (x - m) * rsqrtf(v + params.epsilon) * s + b;
    output[index] = BatchNormFromFloat<T>(y);
  }
}

template <typename T>
musaError_t LaunchBatchNormalizationTyped(musaStream_t stream,
                                          const T* input,
                                          const T* scale,
                                          const T* bias,
                                          const T* mean,
                                          const T* variance,
                                          T* output,
                                          const BatchNormalizationParams& params) {
  if (params.total_elements == 0) {
    return musaSuccess;
  }

  int64_t blocks = (params.total_elements + kBatchNormThreadsPerBlock - 1) / kBatchNormThreadsPerBlock;
  if (blocks > kBatchNormMaxBlocks) {
    blocks = kBatchNormMaxBlocks;
  }

  BatchNormalizationKernel<T><<<static_cast<int>(blocks), kBatchNormThreadsPerBlock, 0, stream>>>(
      input, scale, bias, mean, variance, output, params);
  return musaGetLastError();
}

}  // namespace

musaError_t LaunchBatchNormalizationFloat(musaStream_t stream,
                                          const float* input,
                                          const float* scale,
                                          const float* bias,
                                          const float* mean,
                                          const float* variance,
                                          float* output,
                                          const BatchNormalizationParams& params) {
  return LaunchBatchNormalizationTyped<float>(stream, input, scale, bias, mean, variance, output, params);
}

musaError_t LaunchBatchNormalizationHalf(musaStream_t stream,
                                         const void* input,
                                         const void* scale,
                                         const void* bias,
                                         const void* mean,
                                         const void* variance,
                                         void* output,
                                         const BatchNormalizationParams& params) {
  return LaunchBatchNormalizationTyped<half>(stream,
                                             static_cast<const half*>(input),
                                             static_cast<const half*>(scale),
                                             static_cast<const half*>(bias),
                                             static_cast<const half*>(mean),
                                             static_cast<const half*>(variance),
                                             static_cast<half*>(output),
                                             params);
}

}  // namespace musa
}  // namespace onnxruntime
