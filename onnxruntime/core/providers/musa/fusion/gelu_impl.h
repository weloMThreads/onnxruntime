// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>
#include <musa_runtime.h>

namespace onnxruntime {
namespace musa {

musaError_t LaunchGeluFloat(musaStream_t stream, const float* input, float* output, int64_t count);

}  // namespace musa
}  // namespace onnxruntime
