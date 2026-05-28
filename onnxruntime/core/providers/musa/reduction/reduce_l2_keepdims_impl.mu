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

template <int BLOCK_SIZE>
__device__ __forceinline__ float BlockReduceSum(float value) {
  __shared__ float shared[BLOCK_SIZE / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, mask);
  }

  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();

  value = 0.0f;
  if (warp == 0) {
    value = (threadIdx.x < (BLOCK_SIZE / 32)) ? shared[lane] : 0.0f;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      value += __shfl_xor_sync(0xffffffff, value, mask);
    }
  }

  return value;
}

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
__global__ void ReduceSumSquareKeepDimsKernel(const T* input_data,
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

    output_data[output_index] = ReduceL2FromFloat<T>(sum);
  }
}

template <typename T>
__global__ void ReduceMeanRank2Axis1Kernel(const T* input_data,
                                           T* output_data,
                                           int64_t rows,
                                           int64_t cols) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  float sum = 0.0f;
  const T* row_data = input_data + row * cols;
  for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
    sum += ReduceL2ToFloat<T>(row_data[col]);
  }

  const float total = BlockReduceSum<kReduceL2ThreadsPerBlock>(sum);
  if (threadIdx.x == 0) {
    output_data[row] = ReduceL2FromFloat<T>(total / static_cast<float>(cols));
  }
}

template <typename T>
__global__ void ReduceMeanRank3Axis1Kernel(const T* input_data,
                                           T* output_data,
                                           int64_t outer,
                                           int64_t reduce_dim,
                                           int64_t inner) {
  const int64_t row = blockIdx.x;
  if (row >= outer) {
    return;
  }

  const T* row_data = input_data + row * reduce_dim * inner;
  T* row_output = output_data + row * inner;
  for (int64_t col = threadIdx.x; col < inner; col += blockDim.x) {
    float sum = 0.0f;
    for (int64_t r = 0; r < reduce_dim; ++r) {
      sum += ReduceL2ToFloat<T>(row_data[r * inner + col]);
    }
    row_output[col] = ReduceL2FromFloat<T>(sum / static_cast<float>(reduce_dim));
  }
}

template <typename T>
__global__ void ReduceMeanRank4Axis2Kernel(const T* input_data,
                                           T* output_data,
                                           int64_t dim0,
                                           int64_t dim1,
                                           int64_t reduce_dim,
                                           int64_t inner) {
  const int64_t row = blockIdx.x;
  const int64_t rows = dim0 * dim1;
  if (row >= rows) {
    return;
  }

  const int64_t d0 = row / dim1;
  const int64_t d1 = row - d0 * dim1;
  const T* row_data = input_data + ((d0 * dim1 + d1) * reduce_dim * inner);
  T* row_output = output_data + row * inner;
  for (int64_t col = threadIdx.x; col < inner; col += blockDim.x) {
    float sum = 0.0f;
    for (int64_t r = 0; r < reduce_dim; ++r) {
      sum += ReduceL2ToFloat<T>(row_data[r * inner + col]);
    }
    row_output[col] = ReduceL2FromFloat<T>(sum / static_cast<float>(reduce_dim));
  }
}

template <typename T>
__global__ void ReduceMeanRank4Axes023Kernel(const T* input_data,
                                             T* output_data,
                                             int64_t dim0,
                                             int64_t dim1,
                                             int64_t dim2,
                                             int64_t dim3) {
  const int64_t channel = blockIdx.x;
  if (channel >= dim1) {
    return;
  }

  const int64_t reduce_size = dim0 * dim2 * dim3;
  float sum = 0.0f;
  for (int64_t linear = threadIdx.x; linear < reduce_size; linear += blockDim.x) {
    const int64_t d0 = linear / (dim2 * dim3);
    const int64_t rem = linear - d0 * dim2 * dim3;
    const int64_t d2 = rem / dim3;
    const int64_t d3 = rem - d2 * dim3;
    const int64_t input_index = ((d0 * dim1 + channel) * dim2 + d2) * dim3 + d3;
    sum += ReduceL2ToFloat<T>(input_data[input_index]);
  }

  const float total = BlockReduceSum<kReduceL2ThreadsPerBlock>(sum);
  if (threadIdx.x == 0) {
    output_data[channel] = ReduceL2FromFloat<T>(total / static_cast<float>(reduce_size));
  }
}

template <typename T>
musaError_t LaunchReduceSumSquareTyped(musaStream_t stream,
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

  ReduceSumSquareKeepDimsKernel<T><<<static_cast<int>(blocks), kReduceL2ThreadsPerBlock, 0, stream>>>(
      input_data, output_data, params);
  return musaGetLastError();
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

template <typename T>
musaError_t LaunchReduceMeanTyped(musaStream_t stream,
                                  const T* input_data,
                                  T* output_data,
                                  const ReduceL2KeepDimsParams& params) {
  if (params.output_size == 0) {
    return musaSuccess;
  }

  if (params.rank == 2 && params.num_axes == 1 && params.axes[0] == 1) {
    ReduceMeanRank2Axis1Kernel<T><<<static_cast<int>(params.input_dims[0]),
                                    kReduceL2ThreadsPerBlock, 0, stream>>>(
        input_data, output_data, params.input_dims[0], params.input_dims[1]);
    return musaGetLastError();
  }

  if (params.rank == 3 && params.num_axes == 1 && params.axes[0] == 1) {
    ReduceMeanRank3Axis1Kernel<T><<<static_cast<int>(params.input_dims[0]),
                                    kReduceL2ThreadsPerBlock, 0, stream>>>(
        input_data, output_data, params.input_dims[0], params.input_dims[1], params.input_dims[2]);
    return musaGetLastError();
  }

  if (params.rank == 4 && params.num_axes == 1 && params.axes[0] == 2) {
    const int64_t rows = params.input_dims[0] * params.input_dims[1];
    ReduceMeanRank4Axis2Kernel<T><<<static_cast<int>(rows),
                                    kReduceL2ThreadsPerBlock, 0, stream>>>(
        input_data, output_data, params.input_dims[0], params.input_dims[1],
        params.input_dims[2], params.input_dims[3]);
    return musaGetLastError();
  }

  if (params.rank == 4 && params.num_axes == 3 &&
      params.reduced_axes[0] && params.reduced_axes[2] && params.reduced_axes[3]) {
    ReduceMeanRank4Axes023Kernel<T><<<static_cast<int>(params.input_dims[1]),
                                      kReduceL2ThreadsPerBlock, 0, stream>>>(
        input_data, output_data, params.input_dims[0], params.input_dims[1],
        params.input_dims[2], params.input_dims[3]);
    return musaGetLastError();
  }

  return musaErrorInvalidValue;
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

musaError_t LaunchReduceSumSquareFloat(musaStream_t stream,
                                       const float* input_data,
                                       float* output_data,
                                       const ReduceL2KeepDimsParams& params) {
  return LaunchReduceSumSquareTyped<float>(stream, input_data, output_data, params);
}

musaError_t LaunchReduceSumSquareHalf(musaStream_t stream,
                                      const void* input_data,
                                      void* output_data,
                                      const ReduceL2KeepDimsParams& params) {
  return LaunchReduceSumSquareTyped<half>(stream,
                                          static_cast<const half*>(input_data),
                                          static_cast<half*>(output_data),
                                          params);
}

musaError_t LaunchReduceMeanFloat(musaStream_t stream,
                                  const float* input_data,
                                  float* output_data,
                                  const ReduceL2KeepDimsParams& params) {
  return LaunchReduceMeanTyped<float>(stream, input_data, output_data, params);
}

musaError_t LaunchReduceMeanHalf(musaStream_t stream,
                                 const void* input_data,
                                 void* output_data,
                                 const ReduceL2KeepDimsParams& params) {
  return LaunchReduceMeanTyped<half>(stream,
                                     static_cast<const half*>(input_data),
                                     static_cast<half*>(output_data),
                                     params);
}

}  // namespace musa
}  // namespace onnxruntime
