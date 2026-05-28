// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>
#include <musa_runtime.h>

namespace onnxruntime {
namespace musa {

musaError_t LaunchFeatureNormFloat(musaStream_t stream,
                                   const float* input,
                                   const float* gamma,
                                   const float* beta,
                                   float* output,
                                   int64_t rows,
                                   int64_t cols,
                                   float epsilon);

}  // namespace musa
}  // namespace onnxruntime
