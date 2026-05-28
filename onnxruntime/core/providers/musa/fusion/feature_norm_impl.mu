// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/fusion/feature_norm_impl.h"

#include <math.h>
#include <musa_runtime.h>

namespace onnxruntime {
namespace musa {
namespace {

constexpr int kThreads = 256;

__global__ void FeatureNormKernel(const float* input,
                                  const float* gamma,
                                  const float* beta,
                                  float* output,
                                  int64_t rows,
                                  int64_t cols,
                                  float epsilon) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const float* row_input = input + row * cols;
  float* row_output = output + row * cols;
  const int tid = threadIdx.x;

  __shared__ float shared[kThreads];

  float sum = 0.0f;
  for (int64_t col = tid; col < cols; col += blockDim.x) {
    sum += row_input[col];
  }
  shared[tid] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  const float mean = shared[0] / static_cast<float>(cols);

  float var_sum = 0.0f;
  for (int64_t col = tid; col < cols; col += blockDim.x) {
    const float diff = row_input[col] - mean;
    var_sum += diff * diff;
  }
  shared[tid] = var_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  const float variance = shared[0] / static_cast<float>(cols);
  const float inv_std = rsqrtf(variance + epsilon);

  for (int64_t col = tid; col < cols; col += blockDim.x) {
    row_output[col] = (row_input[col] - mean) * inv_std * gamma[col] + beta[col];
  }
}

}  // namespace

musaError_t LaunchFeatureNormFloat(musaStream_t stream,
                                   const float* input,
                                   const float* gamma,
                                   const float* beta,
                                   float* output,
                                   int64_t rows,
                                   int64_t cols,
                                   float epsilon) {
  if (rows <= 0 || cols <= 0) {
    return musaSuccess;
  }
  FeatureNormKernel<<<static_cast<unsigned int>(rows), kThreads, 0, stream>>>(
      input, gamma, beta, output, rows, cols, epsilon);
  return musaGetLastError();
}

}  // namespace musa
}  // namespace onnxruntime
