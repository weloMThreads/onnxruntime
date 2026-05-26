// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include <musa_fp16.h>
#include <musa_runtime.h>
#include <stdint.h>

#include "core/providers/musa/math/elementwise_safe_impl.h"
#include "core/providers/musa/mu_inc/common.muh"

namespace onnxruntime {
namespace musa {

namespace {

constexpr int kPowThreadsPerBlock = 256;
constexpr int kPowMaxBlocks = 4096;

template <typename T>
__global__ void PowSameTypeKernel(const T* lhs_data,
                                  const T* rhs_data,
                                  T* output_data,
                                  PowSameTypeParams params) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t index = thread_id; index < params.total_elements; index += total_threads) {
    if (params.rank == 0) {
      output_data[index] = _Pow(lhs_data[0], rhs_data[0]);
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

    output_data[index] = _Pow(lhs_data[lhs_index], rhs_data[rhs_index]);
  }
}

}  // namespace

template <typename T>
void LaunchPowSameTypeKernel(musaStream_t stream,
                             const T* lhs_data,
                             const T* rhs_data,
                             T* output_data,
                             const PowSameTypeParams& params) {
  if (params.total_elements == 0) {
    return;
  }

  int64_t blocks = (params.total_elements + kPowThreadsPerBlock - 1) / kPowThreadsPerBlock;
  if (blocks > kPowMaxBlocks) {
    blocks = kPowMaxBlocks;
  }

  PowSameTypeKernel<T><<<static_cast<int>(blocks), kPowThreadsPerBlock, 0, stream>>>(
      lhs_data, rhs_data, output_data, params);
}

void LaunchPowSameTypeKernelHalf(musaStream_t stream,
                                 const void* lhs_data,
                                 const void* rhs_data,
                                 void* output_data,
                                 const PowSameTypeParams& params) {
  LaunchPowSameTypeKernel<half>(stream,
                                static_cast<const half*>(lhs_data),
                                static_cast<const half*>(rhs_data),
                                static_cast<half*>(output_data),
                                params);
}

template void LaunchPowSameTypeKernel<float>(musaStream_t,
                                             const float*,
                                             const float*,
                                             float*,
                                             const PowSameTypeParams&);

template void LaunchPowSameTypeKernel<int32_t>(musaStream_t,
                                               const int32_t*,
                                               const int32_t*,
                                               int32_t*,
                                               const PowSameTypeParams&);

template void LaunchPowSameTypeKernel<int64_t>(musaStream_t,
                                               const int64_t*,
                                               const int64_t*,
                                               int64_t*,
                                               const PowSameTypeParams&);

}  // namespace musa
}  // namespace onnxruntime
