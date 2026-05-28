// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <musa_runtime.h>
#include <stdint.h>

namespace onnxruntime {
namespace musa {

struct BatchNormalizationParams {
  int64_t total_elements;
  int64_t channels;
  int64_t spatial_size;
  float epsilon;
};

musaError_t LaunchBatchNormalizationFloat(musaStream_t stream,
                                          const float* input,
                                          const float* scale,
                                          const float* bias,
                                          const float* mean,
                                          const float* variance,
                                          float* output,
                                          const BatchNormalizationParams& params);

musaError_t LaunchBatchNormalizationHalf(musaStream_t stream,
                                         const void* input,
                                         const void* scale,
                                         const void* bias,
                                         const void* mean,
                                         const void* variance,
                                         void* output,
                                         const BatchNormalizationParams& params);

}  // namespace musa
}  // namespace onnxruntime
