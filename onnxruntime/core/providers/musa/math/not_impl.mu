// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <musa_runtime.h>
#include <stdint.h>

#include "core/providers/musa/math/not_impl.h"

namespace onnxruntime {
namespace musa {
namespace {

constexpr int kNotThreadsPerBlock = 256;
constexpr int kNotMaxBlocks = 4096;

__global__ void NotKernel(const bool* input_data, bool* output_data, int64_t count) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    output_data[index] = !input_data[index];
  }
}

}  // namespace

void NotImpl(musaStream_t stream,
             const bool* input_data,
             bool* output_data,
             int64_t count) {
  if (count <= 0) {
    return;
  }

  int64_t blocks = (count + kNotThreadsPerBlock - 1) / kNotThreadsPerBlock;
  if (blocks > kNotMaxBlocks) {
    blocks = kNotMaxBlocks;
  }
  NotKernel<<<static_cast<int>(blocks), kNotThreadsPerBlock, 0, stream>>>(input_data, output_data, count);
}

}  // namespace musa
}  // namespace onnxruntime
