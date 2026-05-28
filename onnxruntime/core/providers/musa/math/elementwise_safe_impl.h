// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <musa_runtime.h>
#include <stdint.h>

namespace onnxruntime {
namespace musa {

constexpr int32_t kPowSameTypeMaxDims = 8;

struct PowSameTypeParams {
  int32_t rank;
  int64_t total_elements;
  int64_t output_strides[kPowSameTypeMaxDims];
  int64_t lhs_strides[kPowSameTypeMaxDims];
  int64_t rhs_strides[kPowSameTypeMaxDims];
};

template <typename T>
void LaunchPowSameTypeKernel(musaStream_t stream,
                             const T* lhs_data,
                             const T* rhs_data,
                             T* output_data,
                             const PowSameTypeParams& params);

void LaunchPowSameTypeKernelHalf(musaStream_t stream,
                                 const void* lhs_data,
                                 const void* rhs_data,
                                 void* output_data,
                                 const PowSameTypeParams& params);

template <typename T>
void LaunchDivNoNanSameTypeKernel(musaStream_t stream,
                                  const T* lhs_data,
                                  const T* rhs_data,
                                  T* output_data,
                                  const PowSameTypeParams& params);

void LaunchDivNoNanSameTypeKernelHalf(musaStream_t stream,
                                      const void* lhs_data,
                                      const void* rhs_data,
                                      void* output_data,
                                      const PowSameTypeParams& params);

template <typename T>
void LaunchSquaredDifferenceSameTypeKernel(musaStream_t stream,
                                           const T* lhs_data,
                                           const T* rhs_data,
                                           T* output_data,
                                           const PowSameTypeParams& params);

void LaunchSquaredDifferenceSameTypeKernelHalf(musaStream_t stream,
                                               const void* lhs_data,
                                               const void* rhs_data,
                                               void* output_data,
                                               const PowSameTypeParams& params);

template <typename T>
void LaunchFloorDivSameTypeKernel(musaStream_t stream,
                                  const T* lhs_data,
                                  const T* rhs_data,
                                  T* output_data,
                                  const PowSameTypeParams& params);

void LaunchFloorDivSameTypeKernelHalf(musaStream_t stream,
                                      const void* lhs_data,
                                      const void* rhs_data,
                                      void* output_data,
                                      const PowSameTypeParams& params);

template <typename T>
void LaunchFloorModSameTypeKernel(musaStream_t stream,
                                  const T* lhs_data,
                                  const T* rhs_data,
                                  T* output_data,
                                  const PowSameTypeParams& params);

void LaunchFloorModSameTypeKernelHalf(musaStream_t stream,
                                      const void* lhs_data,
                                      const void* rhs_data,
                                      void* output_data,
                                      const PowSameTypeParams& params);

void LaunchLastDimBiasAddFloat(musaStream_t stream,
                               const float* value_data,
                               const float* bias_data,
                               float* output_data,
                               int64_t total_elements,
                               int64_t channels);

void LaunchLastDimBiasAddHalf(musaStream_t stream,
                              const void* value_data,
                              const void* bias_data,
                              void* output_data,
                              int64_t total_elements,
                              int64_t channels);

extern template void LaunchPowSameTypeKernel<float>(musaStream_t,
                                                    const float*,
                                                    const float*,
                                                    float*,
                                                    const PowSameTypeParams&);

extern template void LaunchPowSameTypeKernel<double>(musaStream_t,
                                                     const double*,
                                                     const double*,
                                                     double*,
                                                     const PowSameTypeParams&);

extern template void LaunchPowSameTypeKernel<int32_t>(musaStream_t,
                                                      const int32_t*,
                                                      const int32_t*,
                                                      int32_t*,
                                                      const PowSameTypeParams&);

extern template void LaunchPowSameTypeKernel<int64_t>(musaStream_t,
                                                      const int64_t*,
                                                      const int64_t*,
                                                      int64_t*,
                                                      const PowSameTypeParams&);

extern template void LaunchDivNoNanSameTypeKernel<float>(musaStream_t,
                                                         const float*,
                                                         const float*,
                                                         float*,
                                                         const PowSameTypeParams&);

extern template void LaunchDivNoNanSameTypeKernel<double>(musaStream_t,
                                                          const double*,
                                                          const double*,
                                                          double*,
                                                          const PowSameTypeParams&);

extern template void LaunchDivNoNanSameTypeKernel<int32_t>(musaStream_t,
                                                           const int32_t*,
                                                           const int32_t*,
                                                           int32_t*,
                                                           const PowSameTypeParams&);

extern template void LaunchDivNoNanSameTypeKernel<int64_t>(musaStream_t,
                                                           const int64_t*,
                                                           const int64_t*,
                                                           int64_t*,
                                                           const PowSameTypeParams&);

extern template void LaunchSquaredDifferenceSameTypeKernel<float>(musaStream_t,
                                                                  const float*,
                                                                  const float*,
                                                                  float*,
                                                                  const PowSameTypeParams&);

extern template void LaunchSquaredDifferenceSameTypeKernel<double>(musaStream_t,
                                                                   const double*,
                                                                   const double*,
                                                                   double*,
                                                                   const PowSameTypeParams&);

extern template void LaunchSquaredDifferenceSameTypeKernel<int32_t>(musaStream_t,
                                                                    const int32_t*,
                                                                    const int32_t*,
                                                                    int32_t*,
                                                                    const PowSameTypeParams&);

extern template void LaunchSquaredDifferenceSameTypeKernel<int64_t>(musaStream_t,
                                                                    const int64_t*,
                                                                    const int64_t*,
                                                                    int64_t*,
                                                                    const PowSameTypeParams&);

extern template void LaunchFloorDivSameTypeKernel<float>(musaStream_t,
                                                         const float*,
                                                         const float*,
                                                         float*,
                                                         const PowSameTypeParams&);

extern template void LaunchFloorDivSameTypeKernel<double>(musaStream_t,
                                                          const double*,
                                                          const double*,
                                                          double*,
                                                          const PowSameTypeParams&);

extern template void LaunchFloorDivSameTypeKernel<int32_t>(musaStream_t,
                                                           const int32_t*,
                                                           const int32_t*,
                                                           int32_t*,
                                                           const PowSameTypeParams&);

extern template void LaunchFloorDivSameTypeKernel<int64_t>(musaStream_t,
                                                           const int64_t*,
                                                           const int64_t*,
                                                           int64_t*,
                                                           const PowSameTypeParams&);

extern template void LaunchFloorModSameTypeKernel<float>(musaStream_t,
                                                         const float*,
                                                         const float*,
                                                         float*,
                                                         const PowSameTypeParams&);

extern template void LaunchFloorModSameTypeKernel<double>(musaStream_t,
                                                          const double*,
                                                          const double*,
                                                          double*,
                                                          const PowSameTypeParams&);

extern template void LaunchFloorModSameTypeKernel<int32_t>(musaStream_t,
                                                           const int32_t*,
                                                           const int32_t*,
                                                           int32_t*,
                                                           const PowSameTypeParams&);

extern template void LaunchFloorModSameTypeKernel<int64_t>(musaStream_t,
                                                           const int64_t*,
                                                           const int64_t*,
                                                           int64_t*,
                                                           const PowSameTypeParams&);

}  // namespace musa
}  // namespace onnxruntime
