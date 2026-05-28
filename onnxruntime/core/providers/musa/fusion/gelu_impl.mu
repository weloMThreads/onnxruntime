// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/fusion/gelu_impl.h"

#include <math.h>
#include <musa_runtime.h>

namespace onnxruntime {
namespace musa {
namespace {

__global__ void GeluKernel(const float* input, float* output, int64_t count) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }
  const float x = input[idx];
  output[idx] = 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
}

}  // namespace

musaError_t LaunchGeluFloat(musaStream_t stream, const float* input, float* output, int64_t count) {
  if (count <= 0) {
    return musaSuccess;
  }
  constexpr int kThreads = 256;
  const int64_t blocks = (count + kThreads - 1) / kThreads;
  GeluKernel<<<static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(input, output, count);
  return musaGetLastError();
}

}  // namespace musa
}  // namespace onnxruntime
