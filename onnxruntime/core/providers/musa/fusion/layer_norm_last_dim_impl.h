// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <musa_runtime.h>

namespace onnxruntime {
namespace musa {

musaError_t LaunchLayerNormLastDimFloat(musaStream_t stream,
                                        const float* input,
                                        const float* gamma,
                                        const float* beta,
                                        float* output,
                                        int rows,
                                        int cols,
                                        int gamma_count,
                                        int beta_count,
                                        float clip_min,
                                        float clip_max);

}  // namespace musa
}  // namespace onnxruntime
