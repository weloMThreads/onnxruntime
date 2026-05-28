// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>

#include <musa_runtime.h>

namespace onnxruntime {
namespace musa {

musaError_t LaunchTranspose021MatMulFloat(musaStream_t stream,
                                          const float* input,
                                          const float* weight,
                                          float* output,
                                          int64_t batch,
                                          int64_t k,
                                          int64_t tokens,
                                          int64_t out_channels,
                                          bool transpose_b);

}  // namespace musa
}  // namespace onnxruntime
