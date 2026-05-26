// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include <math.h>
#include <stdint.h>

#include <musa_fp16.h>
#include <musa_runtime.h>

#include "core/providers/musa/reduction/reduce_l2_keepdims_impl.h"

namespace onnxruntime {
namespace musa {

namespace {

constexpr int kReduceL2ThreadsPerBlock = 256;
constexpr int kReduceL2MaxBlocks = 4096;

template <typename T>
__device__ __forceinline__ float ReduceL2ToFloat(T value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float ReduceL2ToFloat<half>(half value) {
  return __half2float(value);
}

template <typename T>
__device__ __forceinline__ T ReduceL2FromFloat(float value) {
  return static_cast<T>(value);
}

template <>
__device__ __forceinline__ half ReduceL2FromFloat<half>(float value) {
  return __float2half(value);
}

template <typename T>
__global__ void ReduceL2KeepDimsKernel(const T* input_data,
                                       T* output_data,
                                       ReduceL2KeepDimsParams params) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t output_index = thread_id; output_index < params.output_size; output_index += total_threads) {
    int64_t input_base = 0;
    int64_t remaining_output = output_index;

    for (int32_t dim = 0; dim < params.rank; ++dim) {
      if (params.reduced_axes[dim]) {
        continue;
      }

      const int64_t coord = remaining_output / params.output_strides[dim];
      remaining_output -= coord * params.output_strides[dim];
      input_base += coord * params.input_strides[dim];
    }

    float sum = 0.0f;
    for (int64_t reduce_index = 0; reduce_index < params.reduce_size; ++reduce_index) {
      int64_t input_index = input_base;
      int64_t remaining_reduce = reduce_index;

      for (int32_t axis_index = params.num_axes - 1; axis_index >= 0; --axis_index) {
        const int32_t dim = params.axes[axis_index];
        const int64_t coord = remaining_reduce % params.input_dims[dim];
        remaining_reduce /= params.input_dims[dim];
        input_index += coord * params.input_strides[dim];
      }

      const float value = ReduceL2ToFloat<T>(input_data[input_index]);
      sum += value * value;
    }

    output_data[output_index] = ReduceL2FromFloat<T>(sqrtf(sum));
  }
}

template <typename T>
musaError_t LaunchReduceL2KeepDimsTyped(musaStream_t stream,
                                        const T* input_data,
                                        T* output_data,
                                        const ReduceL2KeepDimsParams& params) {
  if (params.output_size == 0) {
    return musaSuccess;
  }

  int64_t blocks = (params.output_size + kReduceL2ThreadsPerBlock - 1) / kReduceL2ThreadsPerBlock;
  if (blocks > kReduceL2MaxBlocks) {
    blocks = kReduceL2MaxBlocks;
  }

  ReduceL2KeepDimsKernel<T><<<static_cast<int>(blocks), kReduceL2ThreadsPerBlock, 0, stream>>>(
      input_data, output_data, params);
  return musaGetLastError();
}

}  // namespace

musaError_t LaunchReduceL2KeepDimsFloat(musaStream_t stream,
                                        const float* input_data,
                                        float* output_data,
                                        const ReduceL2KeepDimsParams& params) {
  return LaunchReduceL2KeepDimsTyped<float>(stream, input_data, output_data, params);
}

musaError_t LaunchReduceL2KeepDimsHalf(musaStream_t stream,
                                       const void* input_data,
                                       void* output_data,
                                       const ReduceL2KeepDimsParams& params) {
  return LaunchReduceL2KeepDimsTyped<half>(stream,
                                           static_cast<const half*>(input_data),
                                           static_cast<half*>(output_data),
                                           params);
}

}  // namespace musa
}  // namespace onnxruntime
