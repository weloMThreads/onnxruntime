// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <musa_runtime.h>

namespace onnxruntime {
namespace musa {

void GatherImpl(
    musaStream_t stream,
    int64_t input_block_size,
    int64_t indices_max,
    int64_t output_block_size,
    int64_t block_size,
    const void* indices_data,
    size_t index_element_size,
    const void* input_data,
    size_t element_size,
    void* output_data,
    size_t output_count);

}  // namespace musa
}  // namespace onnxruntime
