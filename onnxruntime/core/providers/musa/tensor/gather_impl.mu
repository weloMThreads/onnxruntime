// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <musa_runtime.h>
#include <stddef.h>
#include <stdint.h>

#include "core/providers/musa/tensor/gather_impl.h"

#define CUDA_LONG int32_t

struct GridDim {
  enum : CUDA_LONG {
    maxThreadsPerBlock = 256,
  };
};

#define CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N) \
  CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x; \
  if (id >= N) return;

namespace onnxruntime {
namespace musa {

__device__ inline int64_t GetIndexValue(const void* index_data, size_t index_element_size, size_t offset) {
  switch (index_element_size) {
    case sizeof(int32_t):
      return static_cast<int64_t>(reinterpret_cast<const int32_t*>(index_data)[offset]);
    case sizeof(int64_t):
      return reinterpret_cast<const int64_t*>(index_data)[offset];
    default:
      return 0;
  }
}

template <typename T>
__global__ void _GatherKernel(
    int64_t input_block_size,
    int64_t indices_max,
    int64_t output_block_size,
    int64_t block_size,
    const void* indices_data,
    size_t index_element_size,
    const T* input_data,
    T* output_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  const int64_t output_index = static_cast<int64_t>(id);
  const int64_t input_block_index = output_index / output_block_size;
  const int64_t block_offset = output_index % output_block_size;
  const int64_t indices_index = block_offset / block_size;
  const int64_t offset = block_offset % block_size;

  int64_t idx = GetIndexValue(indices_data, index_element_size, static_cast<size_t>(indices_index));
  idx = idx < 0 ? idx + indices_max : idx;
  if (idx < 0 || idx >= indices_max) {
    output_data[id] = 0;
    return;
  }

  const int64_t input_index = input_block_index * input_block_size + idx * block_size + offset;
  output_data[id] = input_data[input_index];
}

template <typename T>
inline void LaunchGatherKernel(
    musaStream_t stream,
    int64_t input_block_size,
    int64_t indices_max,
    int64_t output_block_size,
    int64_t block_size,
    const void* indices_data,
    size_t index_element_size,
    const void* input_data,
    void* output_data,
    size_t output_count) {
  const int blocks_per_grid = static_cast<int>(
      (output_count + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock);

  _GatherKernel<T><<<blocks_per_grid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input_block_size,
      indices_max,
      output_block_size,
      block_size,
      indices_data,
      index_element_size,
      reinterpret_cast<const T*>(input_data),
      reinterpret_cast<T*>(output_data),
      static_cast<CUDA_LONG>(output_count));
}

void GatherImpl(
    musaStream_t stream,
    int64_t input_block_size,
    int64_t indices_max,
    int64_t output_block_size,
    int64_t block_size,
    const void* indices_data,
    size_t index_element_size,
    const void* input_data,
    size_t element_size,
    void* output_data,
    size_t output_count) {
  switch (element_size) {
    case sizeof(int8_t):
      LaunchGatherKernel<int8_t>(
          stream, input_block_size, indices_max, output_block_size, block_size,
          indices_data, index_element_size, input_data, output_data, output_count);
      break;
    case sizeof(int16_t):
      LaunchGatherKernel<int16_t>(
          stream, input_block_size, indices_max, output_block_size, block_size,
          indices_data, index_element_size, input_data, output_data, output_count);
      break;
    case sizeof(int32_t):
      LaunchGatherKernel<int32_t>(
          stream, input_block_size, indices_max, output_block_size, block_size,
          indices_data, index_element_size, input_data, output_data, output_count);
      break;
    case sizeof(int64_t):
      LaunchGatherKernel<int64_t>(
          stream, input_block_size, indices_max, output_block_size, block_size,
          indices_data, index_element_size, input_data, output_data, output_count);
      break;
    default:
      break;
  }
}

}  // namespace musa
}  // namespace onnxruntime
