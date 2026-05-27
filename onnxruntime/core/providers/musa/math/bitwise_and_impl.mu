// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/math/bitwise_and_impl.h"

#include <musa_runtime.h>
#include <stdint.h>

namespace onnxruntime {
namespace musa {
namespace {

constexpr int kBitwiseThreadsPerBlock = 256;
constexpr int kBitwiseMaxBlocks = 4096;

template <typename T>
__global__ void BitwiseAndKernel(const T* lhs,
                                 const T* rhs,
                                 T* output,
                                 BitwiseAndParams params) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t index = thread_id; index < params.total_elements; index += total_threads) {
    if (params.rank == 0) {
      output[index] = lhs[0] & rhs[0];
      continue;
    }

    int64_t lhs_index = 0;
    int64_t rhs_index = 0;
    int64_t remaining = index;
    for (int32_t dim = 0; dim < params.rank; ++dim) {
      const int64_t coord = remaining / params.output_strides[dim];
      remaining -= coord * params.output_strides[dim];
      lhs_index += coord * params.lhs_strides[dim];
      rhs_index += coord * params.rhs_strides[dim];
    }

    output[index] = lhs[lhs_index] & rhs[rhs_index];
  }
}

}  // namespace

template <typename T>
void LaunchBitwiseAndKernel(musaStream_t stream,
                            const T* lhs,
                            const T* rhs,
                            T* output,
                            const BitwiseAndParams& params) {
  if (params.total_elements == 0) {
    return;
  }

  int64_t blocks = (params.total_elements + kBitwiseThreadsPerBlock - 1) / kBitwiseThreadsPerBlock;
  if (blocks > kBitwiseMaxBlocks) {
    blocks = kBitwiseMaxBlocks;
  }

  BitwiseAndKernel<T><<<static_cast<int>(blocks), kBitwiseThreadsPerBlock, 0, stream>>>(
      lhs, rhs, output, params);
}

template void LaunchBitwiseAndKernel<int8_t>(musaStream_t, const int8_t*, const int8_t*, int8_t*, const BitwiseAndParams&);
template void LaunchBitwiseAndKernel<uint8_t>(musaStream_t, const uint8_t*, const uint8_t*, uint8_t*, const BitwiseAndParams&);
template void LaunchBitwiseAndKernel<int16_t>(musaStream_t, const int16_t*, const int16_t*, int16_t*, const BitwiseAndParams&);
template void LaunchBitwiseAndKernel<uint16_t>(musaStream_t, const uint16_t*, const uint16_t*, uint16_t*, const BitwiseAndParams&);
template void LaunchBitwiseAndKernel<int32_t>(musaStream_t, const int32_t*, const int32_t*, int32_t*, const BitwiseAndParams&);
template void LaunchBitwiseAndKernel<uint32_t>(musaStream_t, const uint32_t*, const uint32_t*, uint32_t*, const BitwiseAndParams&);
template void LaunchBitwiseAndKernel<int64_t>(musaStream_t, const int64_t*, const int64_t*, int64_t*, const BitwiseAndParams&);
template void LaunchBitwiseAndKernel<uint64_t>(musaStream_t, const uint64_t*, const uint64_t*, uint64_t*, const BitwiseAndParams&);

}  // namespace musa
}  // namespace onnxruntime
