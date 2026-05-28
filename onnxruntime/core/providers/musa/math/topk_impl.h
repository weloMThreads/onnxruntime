// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

#include <musa_runtime.h>

namespace onnxruntime {
namespace musa {

struct TopKLaunchParams {
  int64_t outer_size = 0;
  int64_t axis_dim = 0;
  int64_t inner_size = 0;
  int64_t k = 0;
  bool largest = true;
};

template <typename T>
void LaunchTopKKernel(musaStream_t stream,
                      const T* input,
                      T* values,
                      int64_t* indices,
                      const TopKLaunchParams& params);

void LaunchTopKKernelHalf(musaStream_t stream,
                          const void* input,
                          void* values,
                          int64_t* indices,
                          const TopKLaunchParams& params);

}  // namespace musa
}  // namespace onnxruntime
