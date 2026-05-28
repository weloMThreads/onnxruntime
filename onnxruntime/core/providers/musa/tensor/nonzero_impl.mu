// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/tensor/nonzero_impl.h"

#include <musa_fp16.h>
#include <musa_runtime.h>

namespace onnxruntime {
namespace musa {
namespace {

template <typename T>
__device__ __forceinline__ bool IsNonZeroValue(T value) {
  return value != static_cast<T>(0);
}

__device__ __forceinline__ bool IsNonZeroHalf(half value) {
  return __half2float(value) != 0.0f;
}

template <typename T>
__global__ void NonZeroCountKernel(const T* input, int64_t total_elements, int64_t* count) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  int64_t local_count = 0;
  for (int64_t index = 0; index < total_elements; ++index) {
    if (IsNonZeroValue(input[index])) {
      ++local_count;
    }
  }
  *count = local_count;
}

__global__ void NonZeroCountHalfKernel(const half* input, int64_t total_elements, int64_t* count) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  int64_t local_count = 0;
  for (int64_t index = 0; index < total_elements; ++index) {
    if (IsNonZeroHalf(input[index])) {
      ++local_count;
    }
  }
  *count = local_count;
}

template <typename T>
__global__ void NonZeroFillKernel(const T* input, int64_t* output, NonZeroLaunchParams params) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  int64_t output_index = 0;
  for (int64_t linear_index = 0; linear_index < params.total_elements; ++linear_index) {
    if (!IsNonZeroValue(input[linear_index])) {
      continue;
    }

    if (params.rank == 0) {
      output[output_index] = 0;
    } else {
      int64_t remaining = linear_index;
      for (int32_t dim = 0; dim < params.rank; ++dim) {
        const int64_t coord = remaining / params.strides[dim];
        remaining -= coord * params.strides[dim];
        output[static_cast<int64_t>(dim) * params.nonzero_count + output_index] = coord;
      }
    }
    ++output_index;
  }
}

__global__ void NonZeroFillHalfKernel(const half* input, int64_t* output, NonZeroLaunchParams params) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  int64_t output_index = 0;
  for (int64_t linear_index = 0; linear_index < params.total_elements; ++linear_index) {
    if (!IsNonZeroHalf(input[linear_index])) {
      continue;
    }

    if (params.rank == 0) {
      output[output_index] = 0;
    } else {
      int64_t remaining = linear_index;
      for (int32_t dim = 0; dim < params.rank; ++dim) {
        const int64_t coord = remaining / params.strides[dim];
        remaining -= coord * params.strides[dim];
        output[static_cast<int64_t>(dim) * params.nonzero_count + output_index] = coord;
      }
    }
    ++output_index;
  }
}

}  // namespace

template <typename T>
void LaunchNonZeroCountKernel(musaStream_t stream,
                              const T* input,
                              int64_t total_elements,
                              int64_t* count) {
  NonZeroCountKernel<T><<<1, 1, 0, stream>>>(input, total_elements, count);
}

void LaunchNonZeroCountKernelHalf(musaStream_t stream,
                                  const void* input,
                                  int64_t total_elements,
                                  int64_t* count) {
  NonZeroCountHalfKernel<<<1, 1, 0, stream>>>(static_cast<const half*>(input), total_elements, count);
}

template <typename T>
void LaunchNonZeroFillKernel(musaStream_t stream,
                             const T* input,
                             int64_t* output,
                             const NonZeroLaunchParams& params) {
  if (params.nonzero_count == 0) {
    return;
  }
  NonZeroFillKernel<T><<<1, 1, 0, stream>>>(input, output, params);
}

void LaunchNonZeroFillKernelHalf(musaStream_t stream,
                                 const void* input,
                                 int64_t* output,
                                 const NonZeroLaunchParams& params) {
  if (params.nonzero_count == 0) {
    return;
  }
  NonZeroFillHalfKernel<<<1, 1, 0, stream>>>(static_cast<const half*>(input), output, params);
}

template void LaunchNonZeroCountKernel<bool>(musaStream_t, const bool*, int64_t, int64_t*);
template void LaunchNonZeroCountKernel<uint8_t>(musaStream_t, const uint8_t*, int64_t, int64_t*);
template void LaunchNonZeroCountKernel<int32_t>(musaStream_t, const int32_t*, int64_t, int64_t*);
template void LaunchNonZeroCountKernel<int64_t>(musaStream_t, const int64_t*, int64_t, int64_t*);
template void LaunchNonZeroCountKernel<float>(musaStream_t, const float*, int64_t, int64_t*);
template void LaunchNonZeroCountKernel<double>(musaStream_t, const double*, int64_t, int64_t*);

template void LaunchNonZeroFillKernel<bool>(musaStream_t, const bool*, int64_t*, const NonZeroLaunchParams&);
template void LaunchNonZeroFillKernel<uint8_t>(musaStream_t, const uint8_t*, int64_t*, const NonZeroLaunchParams&);
template void LaunchNonZeroFillKernel<int32_t>(musaStream_t, const int32_t*, int64_t*, const NonZeroLaunchParams&);
template void LaunchNonZeroFillKernel<int64_t>(musaStream_t, const int64_t*, int64_t*, const NonZeroLaunchParams&);
template void LaunchNonZeroFillKernel<float>(musaStream_t, const float*, int64_t*, const NonZeroLaunchParams&);
template void LaunchNonZeroFillKernel<double>(musaStream_t, const double*, int64_t*, const NonZeroLaunchParams&);

}  // namespace musa
}  // namespace onnxruntime
