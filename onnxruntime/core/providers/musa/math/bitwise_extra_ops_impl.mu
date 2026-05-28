// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/math/bitwise_extra_ops_impl.h"

#include <musa_runtime.h>
#include <stdint.h>

namespace onnxruntime {
namespace musa {
namespace {

constexpr int kBitwiseExtraThreadsPerBlock = 256;
constexpr int kBitwiseExtraMaxBlocks = 4096;

inline int BlocksForCount(int64_t count) {
  int64_t blocks = (count + kBitwiseExtraThreadsPerBlock - 1) / kBitwiseExtraThreadsPerBlock;
  if (blocks > kBitwiseExtraMaxBlocks) {
    blocks = kBitwiseExtraMaxBlocks;
  }
  return static_cast<int>(blocks);
}

__device__ __forceinline__ void ResolveBroadcastIndices(int64_t flat_index,
                                                        const BitwiseExtraBinaryParams& params,
                                                        int64_t* lhs_index,
                                                        int64_t* rhs_index) {
  *lhs_index = 0;
  *rhs_index = 0;
  if (params.rank == 0) {
    return;
  }

  int64_t remaining = flat_index;
  for (int32_t dim = 0; dim < params.rank; ++dim) {
    const int64_t coord = remaining / params.output_strides[dim];
    remaining -= coord * params.output_strides[dim];
    *lhs_index += coord * params.lhs_strides[dim];
    *rhs_index += coord * params.rhs_strides[dim];
  }
}

template <typename T>
__global__ void BitwiseOrKernel(const T* lhs,
                                const T* rhs,
                                T* output,
                                BitwiseExtraBinaryParams params) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t index = thread_id; index < params.total_elements; index += total_threads) {
    int64_t lhs_index = 0;
    int64_t rhs_index = 0;
    ResolveBroadcastIndices(index, params, &lhs_index, &rhs_index);
    output[index] = lhs[lhs_index] | rhs[rhs_index];
  }
}

template <typename T>
__global__ void BitwiseXorKernel(const T* lhs,
                                 const T* rhs,
                                 T* output,
                                 BitwiseExtraBinaryParams params) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t index = thread_id; index < params.total_elements; index += total_threads) {
    int64_t lhs_index = 0;
    int64_t rhs_index = 0;
    ResolveBroadcastIndices(index, params, &lhs_index, &rhs_index);
    output[index] = lhs[lhs_index] ^ rhs[rhs_index];
  }
}

template <typename T>
__global__ void BitwiseNotKernel(const T* input, T* output, int64_t count) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t index = thread_id; index < count; index += total_threads) {
    output[index] = ~input[index];
  }
}

}  // namespace

template <typename T>
void LaunchBitwiseOrKernel(musaStream_t stream,
                           const T* lhs,
                           const T* rhs,
                           T* output,
                           const BitwiseExtraBinaryParams& params) {
  if (params.total_elements > 0) {
    BitwiseOrKernel<T><<<BlocksForCount(params.total_elements), kBitwiseExtraThreadsPerBlock, 0, stream>>>(lhs, rhs, output, params);
  }
}

template <typename T>
void LaunchBitwiseXorKernel(musaStream_t stream,
                            const T* lhs,
                            const T* rhs,
                            T* output,
                            const BitwiseExtraBinaryParams& params) {
  if (params.total_elements > 0) {
    BitwiseXorKernel<T><<<BlocksForCount(params.total_elements), kBitwiseExtraThreadsPerBlock, 0, stream>>>(lhs, rhs, output, params);
  }
}

template <typename T>
void LaunchBitwiseNotKernel(musaStream_t stream, const T* input, T* output, int64_t count) {
  if (count > 0) {
    BitwiseNotKernel<T><<<BlocksForCount(count), kBitwiseExtraThreadsPerBlock, 0, stream>>>(input, output, count);
  }
}

#define INSTANTIATE_BITWISE_EXTRA(T)                                            \
  template void LaunchBitwiseOrKernel<T>(musaStream_t, const T*, const T*, T*,  \
                                         const BitwiseExtraBinaryParams&);      \
  template void LaunchBitwiseXorKernel<T>(musaStream_t, const T*, const T*, T*, \
                                          const BitwiseExtraBinaryParams&);     \
  template void LaunchBitwiseNotKernel<T>(musaStream_t, const T*, T*, int64_t);

INSTANTIATE_BITWISE_EXTRA(int8_t)
INSTANTIATE_BITWISE_EXTRA(uint8_t)
INSTANTIATE_BITWISE_EXTRA(int16_t)
INSTANTIATE_BITWISE_EXTRA(uint16_t)
INSTANTIATE_BITWISE_EXTRA(int32_t)
INSTANTIATE_BITWISE_EXTRA(uint32_t)
INSTANTIATE_BITWISE_EXTRA(int64_t)
INSTANTIATE_BITWISE_EXTRA(uint64_t)

#undef INSTANTIATE_BITWISE_EXTRA

}  // namespace musa
}  // namespace onnxruntime
