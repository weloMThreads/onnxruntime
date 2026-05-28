// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/fusion/layer_norm_last_dim_impl.h"

#include <math.h>
#include <musa_runtime.h>

namespace onnxruntime {
namespace musa {
namespace {

__global__ void LayerNormLastDimKernel(const float* input,
                                       const float* gamma,
                                       const float* beta,
                                       float* output,
                                       int rows,
                                       int cols,
                                       int gamma_count,
                                       int beta_count,
                                       float clip_min,
                                       float clip_max) {
  extern __shared__ float shared[];
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const float* row_input = input + static_cast<int64_t>(row) * cols;
  float* row_output = output + static_cast<int64_t>(row) * cols;

  float thread_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    thread_sum += row_input[col];
  }
  shared[threadIdx.x] = thread_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float mean = shared[0] / static_cast<float>(cols);

  float thread_var = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered = row_input[col] - mean;
    thread_var += centered * centered;
  }
  shared[threadIdx.x] = thread_var;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }

  float denom = sqrtf(shared[0] / static_cast<float>(cols));
  denom = fminf(denom, clip_max);
  denom = fmaxf(denom, clip_min);
  const float inv_denom = 1.0f / denom;

  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int gamma_index = gamma_count == 1 ? 0 : col;
    const int beta_index = beta_count == 1 ? 0 : col;
    row_output[col] = (row_input[col] - mean) * inv_denom * gamma[gamma_index] + beta[beta_index];
  }
}

}  // namespace

musaError_t LaunchLayerNormLastDimFloat(musaStream_t stream,
                                        const float* input,
                                        const float* gamma,
                                        const float* beta,
                                        float* output,
                                        int rows,
                                        int cols,
                                        int gamma_count,
                                        int beta_count,
                                        float clip_min,
                                        float clip_max) {
  if (rows <= 0 || cols <= 0) {
    return musaSuccess;
  }

  constexpr int kThreads = 256;
  LayerNormLastDimKernel<<<rows, kThreads, kThreads * sizeof(float), stream>>>(
      input, gamma, beta, output, rows, cols, gamma_count, beta_count, clip_min, clip_max);
  return musaGetLastError();
}

}  // namespace musa
}  // namespace onnxruntime
