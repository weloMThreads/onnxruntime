// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/fusion/reshape_matmul_impl.h"

#include <musa_runtime.h>

namespace onnxruntime {
namespace musa {
namespace {

constexpr int kThreads = 256;

__global__ void Transpose021MatMulFloatKernel(const float* input,
                                              const float* weight,
                                              float* output,
                                              int64_t k,
                                              int64_t tokens,
                                              int64_t out_channels,
                                              int64_t total,
                                              bool transpose_b) {
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t out_c = idx % out_channels;
    const int64_t token = (idx / out_channels) % tokens;
    const int64_t batch = idx / (tokens * out_channels);

    float acc = 0.0f;
    for (int64_t c = 0; c < k; ++c) {
      const float input_value = input[(batch * k + c) * tokens + token];
      const float weight_value = transpose_b ? weight[out_c * k + c] : weight[c * out_channels + out_c];
      acc += input_value * weight_value;
    }
    output[idx] = acc;
  }
}

}  // namespace

musaError_t LaunchTranspose021MatMulFloat(musaStream_t stream,
                                          const float* input,
                                          const float* weight,
                                          float* output,
                                          int64_t batch,
                                          int64_t k,
                                          int64_t tokens,
                                          int64_t out_channels,
                                          bool transpose_b) {
  const int64_t total = batch * tokens * out_channels;
  if (total <= 0) {
    return musaSuccess;
  }

  const int64_t blocks64 = (total + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(blocks64 > 4096 ? 4096 : blocks64);
  Transpose021MatMulFloatKernel<<<blocks, kThreads, 0, stream>>>(
      input, weight, output, k, tokens, out_channels, total, transpose_b);
  return musaGetLastError();
}

}  // namespace musa
}  // namespace onnxruntime
