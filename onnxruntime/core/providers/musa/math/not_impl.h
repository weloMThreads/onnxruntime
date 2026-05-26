// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <musa_runtime.h>
#include <stdint.h>

namespace onnxruntime {
namespace musa {

void NotImpl(musaStream_t stream,
             const bool* input_data,
             bool* output_data,
             int64_t count);

}  // namespace musa
}  // namespace onnxruntime
