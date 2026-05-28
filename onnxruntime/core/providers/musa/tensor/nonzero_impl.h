// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

#include <musa_runtime.h>

namespace onnxruntime {
namespace musa {

constexpr int32_t kNonZeroMaxRank = 8;

struct NonZeroLaunchParams {
  int64_t total_elements = 0;
  int64_t nonzero_count = 0;
  int32_t rank = 0;
  int32_t coordinate_size = 1;
  int64_t dims[kNonZeroMaxRank] = {};
  int64_t strides[kNonZeroMaxRank] = {};
};

template <typename T>
void LaunchNonZeroCountKernel(musaStream_t stream,
                              const T* input,
                              int64_t total_elements,
                              int64_t* count);

void LaunchNonZeroCountKernelHalf(musaStream_t stream,
                                  const void* input,
                                  int64_t total_elements,
                                  int64_t* count);

template <typename T>
void LaunchNonZeroFillKernel(musaStream_t stream,
                             const T* input,
                             int64_t* output,
                             const NonZeroLaunchParams& params);

void LaunchNonZeroFillKernelHalf(musaStream_t stream,
                                 const void* input,
                                 int64_t* output,
                                 const NonZeroLaunchParams& params);

}  // namespace musa
}  // namespace onnxruntime
