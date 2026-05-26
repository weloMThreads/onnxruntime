// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include <musa_runtime.h>
#include <musa_fp16.h>

#include "contrib_ops/musa/activation/activations_impl.h"

namespace {

constexpr int kElementsPerThread = 4;
constexpr int kThreadsPerBlock = 256;

template <typename T>
__device__ __forceinline__ T quick_gelu_op(T x, T alpha) {
  T v = x * alpha;
  T one = static_cast<T>(1.0f);
  T zero = static_cast<T>(0.0f);
  // Numerically stable sigmoid
  T sigmoid = v >= zero ? one / (one + exp(-v)) : one - one / (one + exp(v));
  return x * sigmoid;
}

__device__ __forceinline__ __half quick_gelu_op(__half x, __half alpha) {
  float xf = __half2float(x);
  float af = __half2float(alpha);
  float v = xf * af;
  float sigmoid = v >= 0.0f ? 1.0f / (1.0f + expf(-v)) : 1.0f - 1.0f / (1.0f + expf(v));
  return __float2half(xf * sigmoid);
}

template <typename T>
__global__ void QuickGeluKernel(int64_t n, const T* input, T* output, T alpha) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = quick_gelu_op(input[idx], alpha);
  }
}

// Vectorized version for aligned data
template <typename T, int vec_size>
struct alignas(sizeof(T) * vec_size) AlignedVector {
  T val[vec_size];
};

template <typename T>
__global__ void QuickGeluVectorizedKernel(int64_t n, const T* input, T* output, T alpha) {
  int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * kElementsPerThread;
  if (idx + kElementsPerThread <= n) {
    // Vectorized load
    AlignedVector<T, kElementsPerThread> in_vec;
    AlignedVector<T, kElementsPerThread> out_vec;
    in_vec = *reinterpret_cast<const AlignedVector<T, kElementsPerThread>*>(&input[idx]);
#pragma unroll
    for (int i = 0; i < kElementsPerThread; ++i) {
      out_vec.val[i] = quick_gelu_op(in_vec.val[i], alpha);
    }
    *reinterpret_cast<AlignedVector<T, kElementsPerThread>*>(&output[idx]) = out_vec;
  } else {
    // Scalar tail
    for (int64_t i = idx; i < n && i < idx + kElementsPerThread; ++i) {
      output[i] = quick_gelu_op(input[i], alpha);
    }
  }
}

}  // namespace

namespace onnxruntime {
namespace contrib {
namespace musa {

template <typename T>
void LaunchQuickGeluKernel(musaStream_t stream, int64_t input_size,
                           const T* input, T* output, float alpha) {
  if (input_size == 0) return;

  constexpr int vec_alignment = alignof(AlignedVector<T, kElementsPerThread>);
  bool can_vectorize = (input_size % kElementsPerThread == 0) &&
                       (reinterpret_cast<uint64_t>(input) % vec_alignment == 0) &&
                       (reinterpret_cast<uint64_t>(output) % vec_alignment == 0);

  if (can_vectorize) {
    int64_t num_vec = input_size / kElementsPerThread;
    int num_blocks = static_cast<int>((num_vec + kThreadsPerBlock - 1) / kThreadsPerBlock);
    QuickGeluVectorizedKernel<T><<<num_blocks, kThreadsPerBlock, 0, stream>>>(
        input_size, input, output, static_cast<T>(alpha));
  } else {
    int num_blocks = static_cast<int>((input_size + kThreadsPerBlock - 1) / kThreadsPerBlock);
    QuickGeluKernel<T><<<num_blocks, kThreadsPerBlock, 0, stream>>>(
        input_size, input, output, static_cast<T>(alpha));
  }
}

void LaunchQuickGeluKernelHalf(musaStream_t stream, int64_t input_size,
                               const void* input, void* output, float alpha) {
  LaunchQuickGeluKernel<__half>(stream, input_size,
                                reinterpret_cast<const __half*>(input),
                                reinterpret_cast<__half*>(output),
                                alpha);
}

// Explicit instantiation
template void LaunchQuickGeluKernel<float>(musaStream_t stream, int64_t input_size,
                                           const float* input, float* output, float alpha);

template void LaunchQuickGeluKernel<__half>(musaStream_t stream, int64_t input_size,
                                            const __half* input, __half* output, float alpha);

}  // namespace musa
}  // namespace contrib
}  // namespace onnxruntime
