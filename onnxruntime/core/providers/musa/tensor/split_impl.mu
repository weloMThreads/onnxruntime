// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <stddef.h>
#include <stdint.h>

#include <musa_runtime.h>

#include "core/providers/musa/shared_inc/fast_divmod.h"
#include "core/providers/musa/tensor/split_impl.h"

namespace onnxruntime {
namespace musa {

#define CUDA_LONG int32_t

namespace {

struct GridDim {
  enum : CUDA_LONG {
    maxThreadsPerBlock = 256,
    maxElementsPerThread = 2,
  };
};

inline int CeilDiv(CUDA_LONG a, CUDA_LONG b) {
  return static_cast<int>((a + b - 1) / b);
}

inline Status CheckLaunchStatus() {
  musaError_t status = musaGetLastError();
  if (status != musaSuccess) {
    const char* error_str = musaGetErrorString(status);
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "MUSA split kernel launch failed: ",
                           error_str ? error_str : "Unknown error");
  }

  return Status::OK();
}

template <typename T>
__global__ void SplitKernelSameSplitDim(const fast_divmod block_size_including_axis_dim_div,
                                        const fast_divmod block_size_inside_axis_dim_div,
                                        const fast_divmod split_dim_size, const T* input_data,
                                        void* const* output_data, const CUDA_LONG n) {
  CUDA_LONG start = GridDim::maxElementsPerThread * GridDim::maxThreadsPerBlock * blockIdx.x + threadIdx.x;
  T value[GridDim::maxElementsPerThread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < GridDim::maxElementsPerThread; ++i) {
    if (id < n) {
      value[i] = input_data[id];
      id += GridDim::maxThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < GridDim::maxElementsPerThread; ++i) {
    if (id < n) {
      int outer_block_index, block_index, offset, output_index, block_offset;
      block_size_including_axis_dim_div.divmod(id, outer_block_index, offset);
      block_size_inside_axis_dim_div.divmod(offset, block_index, offset);
      split_dim_size.divmod(block_index, output_index, block_offset);
      CUDA_LONG output_pos =
          (outer_block_index * split_dim_size.d_ + block_offset) * block_size_inside_axis_dim_div.d_ + offset;
      reinterpret_cast<T*>(output_data[output_index])[output_pos] = value[i];
      id += GridDim::maxThreadsPerBlock;
    }
  }
}

template <typename T>
__global__ void SplitKernel(const fast_divmod block_size_including_axis_dim_div,
                            const fast_divmod block_size_inside_axis_dim_div, const int64_t* split_sizes,
                            const int64_t* split_sizes_range, const int64_t* axis_dimension_input_output_mapping,
                            const T* input_data, void* const* output_data, const CUDA_LONG n) {
  CUDA_LONG start = GridDim::maxElementsPerThread * GridDim::maxThreadsPerBlock * blockIdx.x + threadIdx.x;
  T value[GridDim::maxElementsPerThread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < GridDim::maxElementsPerThread; ++i) {
    if (id < n) {
      value[i] = input_data[id];
      id += GridDim::maxThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < GridDim::maxElementsPerThread; ++i) {
    if (id < n) {
      int outer_block_index, block_index, offset;
      block_size_including_axis_dim_div.divmod(id, outer_block_index, offset);
      block_size_inside_axis_dim_div.divmod(offset, block_index, offset);
      int output_index = static_cast<int>(axis_dimension_input_output_mapping[block_index]);
      int64_t range_left = output_index == 0 ? 0 : split_sizes_range[output_index - 1];
      int block_offset = block_index - static_cast<int>(range_left);
      CUDA_LONG output_pos =
          (outer_block_index * split_sizes[output_index] + block_offset) * block_size_inside_axis_dim_div.d_ + offset;
      reinterpret_cast<T*>(output_data[output_index])[output_pos] = value[i];
      id += GridDim::maxThreadsPerBlock;
    }
  }
}

template <typename T>
Status LaunchSplitSameSplitDimImpl(musaStream_t stream, int block_size_including_axis_dim,
                                   int block_size_inside_axis_dim, int64_t split_size, const void* input_data,
                                   void* const* output_data, size_t input_size) {
  CUDA_LONG n = static_cast<CUDA_LONG>(input_size);
  int blocks_per_grid = CeilDiv(n, GridDim::maxElementsPerThread * GridDim::maxThreadsPerBlock);
  fast_divmod block_size_including_axis_dim_div(block_size_including_axis_dim);
  fast_divmod block_size_inside_axis_dim_div(block_size_inside_axis_dim);
  fast_divmod split_size_div(static_cast<int>(split_size));

  SplitKernelSameSplitDim<T><<<blocks_per_grid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      block_size_including_axis_dim_div, block_size_inside_axis_dim_div, split_size_div,
      reinterpret_cast<const T*>(input_data), output_data, n);

  return CheckLaunchStatus();
}

template <typename T>
Status LaunchSplitImpl(musaStream_t stream, int block_size_including_axis_dim, int block_size_inside_axis_dim,
                       const int64_t* split_sizes, const int64_t* split_sizes_range,
                       const int64_t* axis_dimension_input_output_mapping, const void* input_data,
                       void* const* output_data, size_t input_size) {
  CUDA_LONG n = static_cast<CUDA_LONG>(input_size);
  int blocks_per_grid = CeilDiv(n, GridDim::maxElementsPerThread * GridDim::maxThreadsPerBlock);
  fast_divmod block_size_including_axis_dim_div(block_size_including_axis_dim);
  fast_divmod block_size_inside_axis_dim_div(block_size_inside_axis_dim);

  SplitKernel<T><<<blocks_per_grid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      block_size_including_axis_dim_div, block_size_inside_axis_dim_div, split_sizes, split_sizes_range,
      axis_dimension_input_output_mapping, reinterpret_cast<const T*>(input_data), output_data, n);

  return CheckLaunchStatus();
}

}  // namespace

Status SplitSameSplitDimImpl(musaStream_t stream, size_t element_size, int block_size_including_axis_dim,
                             int block_size_inside_axis_dim, int64_t split_size, int /*num_outputs*/,
                             const void* input_data, void* const* output_data, size_t input_size) {
  switch (element_size) {
    case sizeof(int8_t):
      return LaunchSplitSameSplitDimImpl<int8_t>(stream, block_size_including_axis_dim, block_size_inside_axis_dim,
                                                 split_size, input_data, output_data, input_size);
    case sizeof(int16_t):
      return LaunchSplitSameSplitDimImpl<int16_t>(stream, block_size_including_axis_dim, block_size_inside_axis_dim,
                                                  split_size, input_data, output_data, input_size);
    case sizeof(int32_t):
      return LaunchSplitSameSplitDimImpl<int32_t>(stream, block_size_including_axis_dim, block_size_inside_axis_dim,
                                                  split_size, input_data, output_data, input_size);
    case sizeof(int64_t):
      return LaunchSplitSameSplitDimImpl<int64_t>(stream, block_size_including_axis_dim, block_size_inside_axis_dim,
                                                  split_size, input_data, output_data, input_size);
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for Split operator");
  }
}

Status SplitImpl(musaStream_t stream, size_t element_size, int block_size_including_axis_dim,
                 int block_size_inside_axis_dim, const int64_t* split_sizes, const int64_t* split_sizes_range,
                 const int64_t* axis_dimension_input_output_mapping, int /*num_outputs*/, const void* input_data,
                 void* const* output_data, size_t input_size) {
  switch (element_size) {
    case sizeof(int8_t):
      return LaunchSplitImpl<int8_t>(stream, block_size_including_axis_dim, block_size_inside_axis_dim, split_sizes,
                                     split_sizes_range, axis_dimension_input_output_mapping, input_data, output_data,
                                     input_size);
    case sizeof(int16_t):
      return LaunchSplitImpl<int16_t>(stream, block_size_including_axis_dim, block_size_inside_axis_dim, split_sizes,
                                      split_sizes_range, axis_dimension_input_output_mapping, input_data, output_data,
                                      input_size);
    case sizeof(int32_t):
      return LaunchSplitImpl<int32_t>(stream, block_size_including_axis_dim, block_size_inside_axis_dim, split_sizes,
                                      split_sizes_range, axis_dimension_input_output_mapping, input_data, output_data,
                                      input_size);
    case sizeof(int64_t):
      return LaunchSplitImpl<int64_t>(stream, block_size_including_axis_dim, block_size_inside_axis_dim, split_sizes,
                                      split_sizes_range, axis_dimension_input_output_mapping, input_data, output_data,
                                      input_size);
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for Split operator");
  }
}

}  // namespace musa
}  // namespace onnxruntime
