// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <musa_runtime.h>
#include <stdint.h>

namespace onnxruntime {
namespace musa {

void LaunchFloatToBFloat16Kernel(musaStream_t stream,
                                 const float* input_data,
                                 void* output_data,
                                 int64_t element_count);

void LaunchBFloat16ToFloatKernel(musaStream_t stream,
                                 const void* input_data,
                                 float* output_data,
                                 int64_t element_count);

}  // namespace musa
}  // namespace onnxruntime
