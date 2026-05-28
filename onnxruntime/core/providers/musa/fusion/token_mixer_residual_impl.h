// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>

#include <musa_runtime.h>

namespace onnxruntime {
namespace musa {

musaError_t LaunchTokenMixerResidualFloat(musaStream_t stream,
                                          const float* input_data,
                                          float* output_data,
                                          int64_t batch,
                                          int64_t num_t,
                                          int64_t num_h,
                                          int64_t d_k);

musaError_t LaunchTokenMixerResidualHalf(musaStream_t stream,
                                         const void* input_data,
                                         void* output_data,
                                         int64_t batch,
                                         int64_t num_t,
                                         int64_t num_h,
                                         int64_t d_k);

}  // namespace musa
}  // namespace onnxruntime
