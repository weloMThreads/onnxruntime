// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/reduction/reduce_prod_int32_impl.h"

namespace onnxruntime {
namespace musa {
namespace {

constexpr int kReduceProdThreadsPerBlock = 256;
constexpr int kReduceProdMaxBlocks = 4096;

__device__ inline void AtomicMulInt32(int32_t* address, int32_t value) {
  int32_t old = *address;
  int32_t assumed;
  do {
    assumed = old;
    old = atomicCAS(address, assumed, assumed * value);
  } while (assumed != old);
}

__device__ inline void AtomicMulFloat(float* address, float value) {
  int32_t* address_as_int = reinterpret_cast<int32_t*>(address);
  int32_t old = *address_as_int;
  int32_t assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_int, assumed,
                    __float_as_int(__int_as_float(assumed) * value));
  } while (assumed != old);
}

__global__ void FillInt32Kernel(int32_t* output, int64_t output_size, int32_t value) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < output_size; index += total_threads) {
    output[index] = value;
  }
}

__global__ void FillFloatKernel(float* output, int64_t output_size, float value) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < output_size; index += total_threads) {
    output[index] = value;
  }
}

__global__ void ReduceProdInt32Kernel(const int32_t* input,
                                      int32_t* output,
                                      ReduceProdInt32Params params) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t index = thread_id; index < params.input_size; index += total_threads) {
    int64_t remaining = index;
    int64_t output_index = 0;
    for (int32_t dim = 0; dim < params.rank; ++dim) {
      const int64_t coord = remaining / params.input_strides[dim];
      remaining -= coord * params.input_strides[dim];
      output_index += coord * params.output_strides[dim];
    }
    AtomicMulInt32(output + output_index, input[index]);
  }
}

__global__ void ReduceProdFloatKernel(const float* input,
                                      float* output,
                                      ReduceProdInt32Params params) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t index = thread_id; index < params.input_size; index += total_threads) {
    int64_t remaining = index;
    int64_t output_index = 0;
    for (int32_t dim = 0; dim < params.rank; ++dim) {
      const int64_t coord = remaining / params.input_strides[dim];
      remaining -= coord * params.input_strides[dim];
      output_index += coord * params.output_strides[dim];
    }
    AtomicMulFloat(output + output_index, input[index]);
  }
}

__global__ void ReduceProdInt64ScalarKernel(const int64_t* input,
                                            int64_t* output,
                                            int64_t input_size) {
  int64_t product = 1;
  for (int64_t i = 0; i < input_size; ++i) {
    product *= input[i];
  }
  *output = product;
}

int BlocksForSize(int64_t size) {
  int64_t blocks = (size + kReduceProdThreadsPerBlock - 1) / kReduceProdThreadsPerBlock;
  if (blocks > kReduceProdMaxBlocks) {
    blocks = kReduceProdMaxBlocks;
  }
  return static_cast<int>(blocks);
}

}  // namespace

musaError_t LaunchFillInt32Kernel(musaStream_t stream, int32_t* output, int64_t output_size, int32_t value) {
  if (output_size == 0) {
    return musaSuccess;
  }

  FillInt32Kernel<<<BlocksForSize(output_size), kReduceProdThreadsPerBlock, 0, stream>>>(
      output, output_size, value);
  return musaGetLastError();
}

musaError_t LaunchFillFloatKernel(musaStream_t stream, float* output, int64_t output_size, float value) {
  if (output_size == 0) {
    return musaSuccess;
  }

  FillFloatKernel<<<BlocksForSize(output_size), kReduceProdThreadsPerBlock, 0, stream>>>(
      output, output_size, value);
  return musaGetLastError();
}

musaError_t LaunchReduceProdInt32Kernel(musaStream_t stream,
                                        const int32_t* input,
                                        int32_t* output,
                                        const ReduceProdInt32Params& params) {
  if (params.input_size == 0 || params.output_size == 0) {
    return musaSuccess;
  }

  ReduceProdInt32Kernel<<<BlocksForSize(params.input_size), kReduceProdThreadsPerBlock, 0, stream>>>(
      input, output, params);
  return musaGetLastError();
}

musaError_t LaunchReduceProdFloatKernel(musaStream_t stream,
                                        const float* input,
                                        float* output,
                                        const ReduceProdInt32Params& params) {
  if (params.input_size == 0 || params.output_size == 0) {
    return musaSuccess;
  }

  ReduceProdFloatKernel<<<BlocksForSize(params.input_size), kReduceProdThreadsPerBlock, 0, stream>>>(
      input, output, params);
  return musaGetLastError();
}

musaError_t LaunchReduceProdInt64ScalarKernel(musaStream_t stream,
                                              const int64_t* input,
                                              int64_t* output,
                                              int64_t input_size) {
  if (input_size == 0) {
    return musaSuccess;
  }

  ReduceProdInt64ScalarKernel<<<1, 1, 0, stream>>>(input, output, input_size);
  return musaGetLastError();
}

}  // namespace musa
}  // namespace onnxruntime
