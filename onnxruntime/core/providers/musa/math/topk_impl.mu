// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/math/topk_impl.h"

#include <musa_fp16.h>
#include <musa_runtime.h>

namespace onnxruntime {
namespace musa {
namespace {

constexpr int kTopKThreadsPerBlock = 128;
constexpr int kTopKMaxBlocks = 4096;

int BlocksForSlices(int64_t slices) {
  int64_t blocks = (slices + kTopKThreadsPerBlock - 1) / kTopKThreadsPerBlock;
  if (blocks > kTopKMaxBlocks) {
    blocks = kTopKMaxBlocks;
  }
  return static_cast<int>(blocks);
}

template <typename T>
__device__ __forceinline__ bool IsBetter(T lhs, int64_t lhs_index, T rhs, int64_t rhs_index, bool largest) {
  if (largest) {
    if (lhs > rhs) return true;
    if (lhs < rhs) return false;
  } else {
    if (lhs < rhs) return true;
    if (lhs > rhs) return false;
  }
  return lhs_index < rhs_index;
}

__device__ __forceinline__ bool IsBetterHalf(half lhs, int64_t lhs_index, half rhs, int64_t rhs_index, bool largest) {
  const float lhs_value = __half2float(lhs);
  const float rhs_value = __half2float(rhs);
  if (largest) {
    if (lhs_value > rhs_value) return true;
    if (lhs_value < rhs_value) return false;
  } else {
    if (lhs_value < rhs_value) return true;
    if (lhs_value > rhs_value) return false;
  }
  return lhs_index < rhs_index;
}

template <typename T>
__global__ void TopKKernel(const T* input, T* values, int64_t* indices, TopKLaunchParams params) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t slice_count = params.outer_size * params.inner_size;

  for (int64_t slice = thread_id; slice < slice_count; slice += total_threads) {
    const int64_t outer = slice / params.inner_size;
    const int64_t inner = slice - outer * params.inner_size;
    const int64_t input_base = outer * params.axis_dim * params.inner_size + inner;
    const int64_t output_base = outer * params.k * params.inner_size + inner;

    for (int64_t rank = 0; rank < params.k; ++rank) {
      bool has_best = false;
      int64_t best_index = 0;
      T best_value = T{};

      for (int64_t candidate = 0; candidate < params.axis_dim; ++candidate) {
        bool already_selected = false;
        for (int64_t previous_rank = 0; previous_rank < rank; ++previous_rank) {
          const int64_t previous_output_offset = output_base + previous_rank * params.inner_size;
          if (indices[previous_output_offset] == candidate) {
            already_selected = true;
            break;
          }
        }
        if (already_selected) {
          continue;
        }

        const T candidate_value = input[input_base + candidate * params.inner_size];
        if (!has_best || IsBetter(candidate_value, candidate, best_value, best_index, params.largest)) {
          has_best = true;
          best_value = candidate_value;
          best_index = candidate;
        }
      }

      const int64_t output_offset = output_base + rank * params.inner_size;
      values[output_offset] = best_value;
      indices[output_offset] = best_index;
    }
  }
}

__global__ void TopKHalfKernel(const half* input, half* values, int64_t* indices, TopKLaunchParams params) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t slice_count = params.outer_size * params.inner_size;

  for (int64_t slice = thread_id; slice < slice_count; slice += total_threads) {
    const int64_t outer = slice / params.inner_size;
    const int64_t inner = slice - outer * params.inner_size;
    const int64_t input_base = outer * params.axis_dim * params.inner_size + inner;
    const int64_t output_base = outer * params.k * params.inner_size + inner;

    for (int64_t rank = 0; rank < params.k; ++rank) {
      bool has_best = false;
      int64_t best_index = 0;
      half best_value = __float2half(0.0f);

      for (int64_t candidate = 0; candidate < params.axis_dim; ++candidate) {
        bool already_selected = false;
        for (int64_t previous_rank = 0; previous_rank < rank; ++previous_rank) {
          const int64_t previous_output_offset = output_base + previous_rank * params.inner_size;
          if (indices[previous_output_offset] == candidate) {
            already_selected = true;
            break;
          }
        }
        if (already_selected) {
          continue;
        }

        const half candidate_value = input[input_base + candidate * params.inner_size];
        if (!has_best || IsBetterHalf(candidate_value, candidate, best_value, best_index, params.largest)) {
          has_best = true;
          best_value = candidate_value;
          best_index = candidate;
        }
      }

      const int64_t output_offset = output_base + rank * params.inner_size;
      values[output_offset] = best_value;
      indices[output_offset] = best_index;
    }
  }
}

}  // namespace

template <typename T>
void LaunchTopKKernel(musaStream_t stream,
                      const T* input,
                      T* values,
                      int64_t* indices,
                      const TopKLaunchParams& params) {
  const int64_t slice_count = params.outer_size * params.inner_size;
  if (slice_count == 0 || params.k == 0) {
    return;
  }
  TopKKernel<T><<<BlocksForSlices(slice_count), kTopKThreadsPerBlock, 0, stream>>>(input, values, indices, params);
}

void LaunchTopKKernelHalf(musaStream_t stream,
                          const void* input,
                          void* values,
                          int64_t* indices,
                          const TopKLaunchParams& params) {
  const int64_t slice_count = params.outer_size * params.inner_size;
  if (slice_count == 0 || params.k == 0) {
    return;
  }
  TopKHalfKernel<<<BlocksForSlices(slice_count), kTopKThreadsPerBlock, 0, stream>>>(
      static_cast<const half*>(input), static_cast<half*>(values), indices, params);
}

template void LaunchTopKKernel<float>(musaStream_t, const float*, float*, int64_t*, const TopKLaunchParams&);
template void LaunchTopKKernel<double>(musaStream_t, const double*, double*, int64_t*, const TopKLaunchParams&);
template void LaunchTopKKernel<int32_t>(musaStream_t, const int32_t*, int32_t*, int64_t*, const TopKLaunchParams&);
template void LaunchTopKKernel<int64_t>(musaStream_t, const int64_t*, int64_t*, int64_t*, const TopKLaunchParams&);

}  // namespace musa
}  // namespace onnxruntime
