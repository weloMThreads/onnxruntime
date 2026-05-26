// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <musa_runtime.h>
#include <stdint.h>

namespace onnxruntime {
namespace musa {

constexpr int32_t kUnarySameTypeMaxDims = 8;

struct UnarySameTypeParams {
  int32_t rank;
  int64_t total_elements;
  int64_t input_strides[kUnarySameTypeMaxDims];
  int64_t output_strides[kUnarySameTypeMaxDims];
};

template <typename T>
void LaunchSqrtSameTypeKernel(musaStream_t stream,
                              const T* input_data,
                              T* output_data,
                              const UnarySameTypeParams& params);

void LaunchSqrtSameTypeKernelHalf(musaStream_t stream,
                                  const void* input_data,
                                  void* output_data,
                                  const UnarySameTypeParams& params);

void LaunchSqrtSameTypeKernelBFloat16(musaStream_t stream,
                                      const void* input_data,
                                      void* output_data,
                                      const UnarySameTypeParams& params);

extern template void LaunchSqrtSameTypeKernel<float>(musaStream_t,
                                                     const float*,
                                                     float*,
                                                     const UnarySameTypeParams&);

extern template void LaunchSqrtSameTypeKernel<double>(musaStream_t,
                                                      const double*,
                                                      double*,
                                                      const UnarySameTypeParams&);

}  // namespace musa
}  // namespace onnxruntime
