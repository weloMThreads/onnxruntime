// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include <musa_fp16.h>
#include <musa_runtime.h>
#include <math.h>
#include <stdint.h>

#include "core/providers/musa/math/elementwise_safe_impl.h"
#include "core/providers/musa/mu_inc/common.muh"

namespace onnxruntime {
namespace musa {

namespace {

constexpr int kPowThreadsPerBlock = 256;
constexpr int kPowMaxBlocks = 4096;

template <typename T>
__device__ __forceinline__ T DivNoNanValue(T lhs, T rhs) {
  return rhs == static_cast<T>(0) ? static_cast<T>(0) : lhs / rhs;
}

__device__ __forceinline__ half DivNoNanValue(half lhs, half rhs) {
  const float rhs_f = __half2float(rhs);
  if (rhs_f == 0.0f) {
    return __float2half(0.0f);
  }
  return __float2half(__half2float(lhs) / rhs_f);
}

template <typename T>
__device__ __forceinline__ T SquaredDifferenceValue(T lhs, T rhs) {
  const T diff = lhs - rhs;
  return diff * diff;
}

__device__ __forceinline__ half SquaredDifferenceValue(half lhs, half rhs) {
  const float diff = __half2float(lhs) - __half2float(rhs);
  return __float2half(diff * diff);
}

template <typename T>
__device__ __forceinline__ T FloorDivValue(T lhs, T rhs) {
  const T quot = lhs / rhs;
  const T rem = lhs % rhs;
  return (rem != static_cast<T>(0) && ((rem > static_cast<T>(0)) != (rhs > static_cast<T>(0)))) ? static_cast<T>(quot - static_cast<T>(1)) : quot;
}

__device__ __forceinline__ float FloorDivValue(float lhs, float rhs) {
  return floorf(lhs / rhs);
}

__device__ __forceinline__ double FloorDivValue(double lhs, double rhs) {
  return floor(lhs / rhs);
}

__device__ __forceinline__ half FloorDivValue(half lhs, half rhs) {
  return __float2half(floorf(__half2float(lhs) / __half2float(rhs)));
}

template <typename T>
__device__ __forceinline__ T FloorModValue(T lhs, T rhs) {
  return lhs - FloorDivValue(lhs, rhs) * rhs;
}

__device__ __forceinline__ half FloorModValue(half lhs, half rhs) {
  const float lhs_f = __half2float(lhs);
  const float rhs_f = __half2float(rhs);
  return __float2half(lhs_f - floorf(lhs_f / rhs_f) * rhs_f);
}

template <typename T>
__device__ __forceinline__ void ResolveBroadcastIndices(int64_t index,
                                                        const PowSameTypeParams& params,
                                                        int64_t& lhs_index,
                                                        int64_t& rhs_index) {
  lhs_index = 0;
  rhs_index = 0;
  if (params.rank == 0) {
    return;
  }

  int64_t remaining = index;
  for (int32_t dim = 0; dim < params.rank; ++dim) {
    const int64_t coord = remaining / params.output_strides[dim];
    remaining -= coord * params.output_strides[dim];
    lhs_index += coord * params.lhs_strides[dim];
    rhs_index += coord * params.rhs_strides[dim];
  }
}

template <typename T>
__global__ void PowSameTypeKernel(const T* lhs_data,
                                  const T* rhs_data,
                                  T* output_data,
                                  PowSameTypeParams params) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t index = thread_id; index < params.total_elements; index += total_threads) {
    int64_t lhs_index = 0;
    int64_t rhs_index = 0;
    ResolveBroadcastIndices<T>(index, params, lhs_index, rhs_index);
    output_data[index] = _Pow(lhs_data[lhs_index], rhs_data[rhs_index]);
  }
}

template <typename T>
__global__ void DivNoNanSameTypeKernel(const T* lhs_data,
                                       const T* rhs_data,
                                       T* output_data,
                                       PowSameTypeParams params) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t index = thread_id; index < params.total_elements; index += total_threads) {
    int64_t lhs_index = 0;
    int64_t rhs_index = 0;
    ResolveBroadcastIndices<T>(index, params, lhs_index, rhs_index);
    output_data[index] = DivNoNanValue(lhs_data[lhs_index], rhs_data[rhs_index]);
  }
}

template <typename T>
__global__ void SquaredDifferenceSameTypeKernel(const T* lhs_data,
                                                const T* rhs_data,
                                                T* output_data,
                                                PowSameTypeParams params) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t index = thread_id; index < params.total_elements; index += total_threads) {
    int64_t lhs_index = 0;
    int64_t rhs_index = 0;
    ResolveBroadcastIndices<T>(index, params, lhs_index, rhs_index);
    output_data[index] = SquaredDifferenceValue(lhs_data[lhs_index], rhs_data[rhs_index]);
  }
}

template <typename T>
__global__ void FloorDivSameTypeKernel(const T* lhs_data,
                                       const T* rhs_data,
                                       T* output_data,
                                       PowSameTypeParams params) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t index = thread_id; index < params.total_elements; index += total_threads) {
    int64_t lhs_index = 0;
    int64_t rhs_index = 0;
    ResolveBroadcastIndices<T>(index, params, lhs_index, rhs_index);
    output_data[index] = FloorDivValue(lhs_data[lhs_index], rhs_data[rhs_index]);
  }
}

template <typename T>
__global__ void FloorModSameTypeKernel(const T* lhs_data,
                                       const T* rhs_data,
                                       T* output_data,
                                       PowSameTypeParams params) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t index = thread_id; index < params.total_elements; index += total_threads) {
    int64_t lhs_index = 0;
    int64_t rhs_index = 0;
    ResolveBroadcastIndices<T>(index, params, lhs_index, rhs_index);
    output_data[index] = FloorModValue(lhs_data[lhs_index], rhs_data[rhs_index]);
  }
}

template <typename T>
__global__ void LastDimBiasAddKernel(const T* value_data,
                                     const T* bias_data,
                                     T* output_data,
                                     int64_t total_elements,
                                     int64_t channels) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t index = thread_id; index < total_elements; index += total_threads) {
    output_data[index] = value_data[index] + bias_data[index % channels];
  }
}

}  // namespace

inline int BlocksForBinaryCount(int64_t count) {
  int64_t blocks = (count + kPowThreadsPerBlock - 1) / kPowThreadsPerBlock;
  if (blocks > kPowMaxBlocks) {
    blocks = kPowMaxBlocks;
  }
  return static_cast<int>(blocks);
}

template <typename T>
void LaunchPowSameTypeKernel(musaStream_t stream,
                             const T* lhs_data,
                             const T* rhs_data,
                             T* output_data,
                             const PowSameTypeParams& params) {
  if (params.total_elements == 0) {
    return;
  }

  PowSameTypeKernel<T><<<BlocksForBinaryCount(params.total_elements), kPowThreadsPerBlock, 0, stream>>>(
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

template <typename T>
void LaunchDivNoNanSameTypeKernel(musaStream_t stream,
                                  const T* lhs_data,
                                  const T* rhs_data,
                                  T* output_data,
                                  const PowSameTypeParams& params) {
  if (params.total_elements == 0) {
    return;
  }

  DivNoNanSameTypeKernel<T><<<BlocksForBinaryCount(params.total_elements), kPowThreadsPerBlock, 0, stream>>>(
      lhs_data, rhs_data, output_data, params);
}

void LaunchDivNoNanSameTypeKernelHalf(musaStream_t stream,
                                      const void* lhs_data,
                                      const void* rhs_data,
                                      void* output_data,
                                      const PowSameTypeParams& params) {
  LaunchDivNoNanSameTypeKernel<half>(stream,
                                     static_cast<const half*>(lhs_data),
                                     static_cast<const half*>(rhs_data),
                                     static_cast<half*>(output_data),
                                     params);
}

template <typename T>
void LaunchSquaredDifferenceSameTypeKernel(musaStream_t stream,
                                           const T* lhs_data,
                                           const T* rhs_data,
                                           T* output_data,
                                           const PowSameTypeParams& params) {
  if (params.total_elements == 0) {
    return;
  }

  SquaredDifferenceSameTypeKernel<T><<<BlocksForBinaryCount(params.total_elements), kPowThreadsPerBlock, 0, stream>>>(
      lhs_data, rhs_data, output_data, params);
}

void LaunchSquaredDifferenceSameTypeKernelHalf(musaStream_t stream,
                                               const void* lhs_data,
                                               const void* rhs_data,
                                               void* output_data,
                                               const PowSameTypeParams& params) {
  LaunchSquaredDifferenceSameTypeKernel<half>(stream,
                                              static_cast<const half*>(lhs_data),
                                              static_cast<const half*>(rhs_data),
                                              static_cast<half*>(output_data),
                                              params);
}

template <typename T>
void LaunchFloorDivSameTypeKernel(musaStream_t stream,
                                  const T* lhs_data,
                                  const T* rhs_data,
                                  T* output_data,
                                  const PowSameTypeParams& params) {
  if (params.total_elements == 0) {
    return;
  }

  FloorDivSameTypeKernel<T><<<BlocksForBinaryCount(params.total_elements), kPowThreadsPerBlock, 0, stream>>>(
      lhs_data, rhs_data, output_data, params);
}

void LaunchFloorDivSameTypeKernelHalf(musaStream_t stream,
                                      const void* lhs_data,
                                      const void* rhs_data,
                                      void* output_data,
                                      const PowSameTypeParams& params) {
  LaunchFloorDivSameTypeKernel<half>(stream,
                                     static_cast<const half*>(lhs_data),
                                     static_cast<const half*>(rhs_data),
                                     static_cast<half*>(output_data),
                                     params);
}

template <typename T>
void LaunchFloorModSameTypeKernel(musaStream_t stream,
                                  const T* lhs_data,
                                  const T* rhs_data,
                                  T* output_data,
                                  const PowSameTypeParams& params) {
  if (params.total_elements == 0) {
    return;
  }

  FloorModSameTypeKernel<T><<<BlocksForBinaryCount(params.total_elements), kPowThreadsPerBlock, 0, stream>>>(
      lhs_data, rhs_data, output_data, params);
}

void LaunchFloorModSameTypeKernelHalf(musaStream_t stream,
                                      const void* lhs_data,
                                      const void* rhs_data,
                                      void* output_data,
                                      const PowSameTypeParams& params) {
  LaunchFloorModSameTypeKernel<half>(stream,
                                     static_cast<const half*>(lhs_data),
                                     static_cast<const half*>(rhs_data),
                                     static_cast<half*>(output_data),
                                     params);
}

void LaunchLastDimBiasAddFloat(musaStream_t stream,
                               const float* value_data,
                               const float* bias_data,
                               float* output_data,
                               int64_t total_elements,
                               int64_t channels) {
  if (total_elements == 0) {
    return;
  }

  LastDimBiasAddKernel<float><<<BlocksForBinaryCount(total_elements), kPowThreadsPerBlock, 0, stream>>>(
      value_data, bias_data, output_data, total_elements, channels);
}

void LaunchLastDimBiasAddHalf(musaStream_t stream,
                              const void* value_data,
                              const void* bias_data,
                              void* output_data,
                              int64_t total_elements,
                              int64_t channels) {
  if (total_elements == 0) {
    return;
  }

  LastDimBiasAddKernel<half><<<BlocksForBinaryCount(total_elements), kPowThreadsPerBlock, 0, stream>>>(
      static_cast<const half*>(value_data),
      static_cast<const half*>(bias_data),
      static_cast<half*>(output_data),
      total_elements,
      channels);
}

template void LaunchPowSameTypeKernel<float>(musaStream_t,
                                             const float*,
                                             const float*,
                                             float*,
                                             const PowSameTypeParams&);

template void LaunchPowSameTypeKernel<double>(musaStream_t,
                                              const double*,
                                              const double*,
                                              double*,
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

template void LaunchDivNoNanSameTypeKernel<float>(musaStream_t,
                                                  const float*,
                                                  const float*,
                                                  float*,
                                                  const PowSameTypeParams&);

template void LaunchDivNoNanSameTypeKernel<double>(musaStream_t,
                                                   const double*,
                                                   const double*,
                                                   double*,
                                                   const PowSameTypeParams&);

template void LaunchDivNoNanSameTypeKernel<int32_t>(musaStream_t,
                                                    const int32_t*,
                                                    const int32_t*,
                                                    int32_t*,
                                                    const PowSameTypeParams&);

template void LaunchDivNoNanSameTypeKernel<int64_t>(musaStream_t,
                                                    const int64_t*,
                                                    const int64_t*,
                                                    int64_t*,
                                                    const PowSameTypeParams&);

template void LaunchSquaredDifferenceSameTypeKernel<float>(musaStream_t,
                                                           const float*,
                                                           const float*,
                                                           float*,
                                                           const PowSameTypeParams&);

template void LaunchSquaredDifferenceSameTypeKernel<double>(musaStream_t,
                                                            const double*,
                                                            const double*,
                                                            double*,
                                                            const PowSameTypeParams&);

template void LaunchSquaredDifferenceSameTypeKernel<int32_t>(musaStream_t,
                                                             const int32_t*,
                                                             const int32_t*,
                                                             int32_t*,
                                                             const PowSameTypeParams&);

template void LaunchSquaredDifferenceSameTypeKernel<int64_t>(musaStream_t,
                                                             const int64_t*,
                                                             const int64_t*,
                                                             int64_t*,
                                                             const PowSameTypeParams&);

template void LaunchFloorDivSameTypeKernel<float>(musaStream_t,
                                                  const float*,
                                                  const float*,
                                                  float*,
                                                  const PowSameTypeParams&);

template void LaunchFloorDivSameTypeKernel<double>(musaStream_t,
                                                   const double*,
                                                   const double*,
                                                   double*,
                                                   const PowSameTypeParams&);

template void LaunchFloorDivSameTypeKernel<int32_t>(musaStream_t,
                                                    const int32_t*,
                                                    const int32_t*,
                                                    int32_t*,
                                                    const PowSameTypeParams&);

template void LaunchFloorDivSameTypeKernel<int64_t>(musaStream_t,
                                                    const int64_t*,
                                                    const int64_t*,
                                                    int64_t*,
                                                    const PowSameTypeParams&);

template void LaunchFloorModSameTypeKernel<float>(musaStream_t,
                                                  const float*,
                                                  const float*,
                                                  float*,
                                                  const PowSameTypeParams&);

template void LaunchFloorModSameTypeKernel<double>(musaStream_t,
                                                   const double*,
                                                   const double*,
                                                   double*,
                                                   const PowSameTypeParams&);

template void LaunchFloorModSameTypeKernel<int32_t>(musaStream_t,
                                                    const int32_t*,
                                                    const int32_t*,
                                                    int32_t*,
                                                    const PowSameTypeParams&);

template void LaunchFloorModSameTypeKernel<int64_t>(musaStream_t,
                                                    const int64_t*,
                                                    const int64_t*,
                                                    int64_t*,
                                                    const PowSameTypeParams&);

}  // namespace musa
}  // namespace onnxruntime
