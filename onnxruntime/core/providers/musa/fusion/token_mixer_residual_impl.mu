// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include <stdint.h>

#include <musa_fp16.h>
#include <musa_runtime.h>

#include "core/providers/musa/fusion/token_mixer_residual_impl.h"

namespace onnxruntime {
namespace musa {

namespace {

constexpr int kTokenMixerResidualThreadsPerBlock = 256;
constexpr int kTokenMixerResidualMaxBlocks = 4096;

inline int BlocksForTokenMixerResidual(int64_t count) {
  const int64_t blocks = (count + kTokenMixerResidualThreadsPerBlock - 1) / kTokenMixerResidualThreadsPerBlock;
  return static_cast<int>(blocks > kTokenMixerResidualMaxBlocks ? kTokenMixerResidualMaxBlocks : blocks);
}

template <typename T>
__device__ __forceinline__ T AddValue(T lhs, T rhs) {
  return lhs + rhs;
}

template <>
__device__ __forceinline__ half AddValue(half lhs, half rhs) {
  return __float2half(__half2float(lhs) + __half2float(rhs));
}

template <typename T>
__global__ void TokenMixerResidualKernel(const T* input_data,
                                         T* output_data,
                                         int64_t batch,
                                         int64_t num_t,
                                         int64_t num_h,
                                         int64_t d_k) {
  const int64_t total = batch * num_h * num_t * d_k;
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t input_stride_t = num_h * d_k;
  const int64_t batch_stride = num_t * num_h * d_k;

  for (int64_t index = thread_id; index < total; index += total_threads) {
    const int64_t k = index % d_k;
    int64_t remaining = index / d_k;
    const int64_t t = remaining % num_t;
    remaining /= num_t;
    const int64_t h = remaining % num_h;
    const int64_t b = remaining / num_h;

    const int64_t batch_base = b * batch_stride;
    const int64_t permuted_index = batch_base + t * input_stride_t + h * d_k + k;
    const int64_t residual_index = batch_base + h * input_stride_t + t * d_k + k;
    output_data[index] = AddValue(input_data[permuted_index], input_data[residual_index]);
  }
}

}  // namespace

musaError_t LaunchTokenMixerResidualFloat(musaStream_t stream,
                                          const float* input_data,
                                          float* output_data,
                                          int64_t batch,
                                          int64_t num_t,
                                          int64_t num_h,
                                          int64_t d_k) {
  const int64_t total = batch * num_h * num_t * d_k;
  if (total > 0) {
    TokenMixerResidualKernel<float><<<BlocksForTokenMixerResidual(total),
                                       kTokenMixerResidualThreadsPerBlock,
                                       0,
                                       stream>>>(input_data, output_data, batch, num_t, num_h, d_k);
  }
  return musaGetLastError();
}

musaError_t LaunchTokenMixerResidualHalf(musaStream_t stream,
                                         const void* input_data,
                                         void* output_data,
                                         int64_t batch,
                                         int64_t num_t,
                                         int64_t num_h,
                                         int64_t d_k) {
  const int64_t total = batch * num_h * num_t * d_k;
  if (total > 0) {
    TokenMixerResidualKernel<half><<<BlocksForTokenMixerResidual(total),
                                      kTokenMixerResidualThreadsPerBlock,
                                      0,
                                      stream>>>(static_cast<const half*>(input_data),
                                                static_cast<half*>(output_data),
                                                batch,
                                                num_t,
                                                num_h,
                                                d_k);
  }
  return musaGetLastError();
}

}  // namespace musa
}  // namespace onnxruntime
