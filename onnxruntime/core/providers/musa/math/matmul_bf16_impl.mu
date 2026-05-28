// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include <musa_bf16.h>
#include <musa_runtime.h>
#include <stdint.h>

#include "core/providers/musa/math/matmul_bf16_impl.h"

namespace onnxruntime {
namespace musa {

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kMaxBlocks = 4096;

__global__ void FloatToBFloat16Kernel(const float* input_data,
                                      __mt_bfloat16* output_data,
                                      int64_t element_count) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t i = thread_id; i < element_count; i += total_threads) {
    output_data[i] = __float2bfloat16_rn(input_data[i]);
  }
}

__global__ void BFloat16ToFloatKernel(const __mt_bfloat16* input_data,
                                      float* output_data,
                                      int64_t element_count) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t i = thread_id; i < element_count; i += total_threads) {
    output_data[i] = __bfloat162float(input_data[i]);
  }
}

int GetBlockCount(int64_t element_count) {
  int64_t blocks = (element_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
  if (blocks > kMaxBlocks) {
    blocks = kMaxBlocks;
  }
  return static_cast<int>(blocks);
}

}  // namespace

void LaunchFloatToBFloat16Kernel(musaStream_t stream,
                                 const float* input_data,
                                 void* output_data,
                                 int64_t element_count) {
  if (element_count <= 0) {
    return;
  }

  FloatToBFloat16Kernel<<<GetBlockCount(element_count), kThreadsPerBlock, 0, stream>>>(
      input_data, static_cast<__mt_bfloat16*>(output_data), element_count);
}

void LaunchBFloat16ToFloatKernel(musaStream_t stream,
                                 const void* input_data,
                                 float* output_data,
                                 int64_t element_count) {
  if (element_count <= 0) {
    return;
  }

  BFloat16ToFloatKernel<<<GetBlockCount(element_count), kThreadsPerBlock, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(input_data), output_data, element_count);
}

}  // namespace musa
}  // namespace onnxruntime
