// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <stdint.h>

#include "core/providers/musa/math/unary_elementwise_safe_impl.h"
#include "core/providers/musa/mu_inc/common.muh"

namespace onnxruntime {
namespace musa {

namespace {

constexpr int kUnaryThreadsPerBlock = 256;
constexpr int kUnaryMaxBlocks = 4096;

template <typename T>
__global__ void SqrtSameTypeKernel(const T* input_data,
                                   T* output_data,
                                   UnarySameTypeParams params) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t index = thread_id; index < params.total_elements; index += total_threads) {
    if (params.rank == 0) {
      output_data[index] = _Sqrt(input_data[0]);
      continue;
    }

    int64_t input_index = 0;
    int64_t remaining = index;
    for (int32_t dim = 0; dim < params.rank; ++dim) {
      const int64_t coord = remaining / params.output_strides[dim];
      remaining -= coord * params.output_strides[dim];
      input_index += coord * params.input_strides[dim];
    }

    output_data[index] = _Sqrt(input_data[input_index]);
  }
}

}  // namespace

template <typename T>
void LaunchSqrtSameTypeKernel(musaStream_t stream,
                              const T* input_data,
                              T* output_data,
                              const UnarySameTypeParams& params) {
  if (params.total_elements == 0) {
    return;
  }

  int64_t blocks = (params.total_elements + kUnaryThreadsPerBlock - 1) / kUnaryThreadsPerBlock;
  if (blocks > kUnaryMaxBlocks) {
    blocks = kUnaryMaxBlocks;
  }

  SqrtSameTypeKernel<T><<<static_cast<int>(blocks), kUnaryThreadsPerBlock, 0, stream>>>(
      input_data, output_data, params);
}

void LaunchSqrtSameTypeKernelHalf(musaStream_t stream,
                                  const void* input_data,
                                  void* output_data,
                                  const UnarySameTypeParams& params) {
  LaunchSqrtSameTypeKernel<half>(stream,
                                 static_cast<const half*>(input_data),
                                 static_cast<half*>(output_data),
                                 params);
}

void LaunchSqrtSameTypeKernelBFloat16(musaStream_t stream,
                                      const void* input_data,
                                      void* output_data,
                                      const UnarySameTypeParams& params) {
  LaunchSqrtSameTypeKernel<__mt_bfloat16>(stream,
                                          static_cast<const __mt_bfloat16*>(input_data),
                                          static_cast<__mt_bfloat16*>(output_data),
                                          params);
}

template void LaunchSqrtSameTypeKernel<float>(musaStream_t,
                                              const float*,
                                              float*,
                                              const UnarySameTypeParams&);

template void LaunchSqrtSameTypeKernel<double>(musaStream_t,
                                               const double*,
                                               double*,
                                               const UnarySameTypeParams&);

template void LaunchSqrtSameTypeKernel<half>(musaStream_t,
                                             const half*,
                                             half*,
                                             const UnarySameTypeParams&);

template void LaunchSqrtSameTypeKernel<__mt_bfloat16>(musaStream_t,
                                                      const __mt_bfloat16*,
                                                      __mt_bfloat16*,
                                                      const UnarySameTypeParams&);

}  // namespace musa
}  // namespace onnxruntime
