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

extern template void LaunchPowSameTypeKernel<float>(musaStream_t,
                                                    const float*,
                                                    const float*,
                                                    float*,
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

}  // namespace musa
}  // namespace onnxruntime
