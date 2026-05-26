// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <musa_runtime.h>

#include "core/common/common.h"

namespace onnxruntime {
namespace musa {

Status SplitSameSplitDimImpl(musaStream_t stream, size_t element_size, int block_size_including_axis_dim,
                             int block_size_inside_axis_dim, int64_t split_size, int num_outputs,
                             const void* input_data, void* const* output_data, size_t input_size);

Status SplitImpl(musaStream_t stream, size_t element_size, int block_size_including_axis_dim,
                 int block_size_inside_axis_dim, const int64_t* split_sizes, const int64_t* split_sizes_range,
                 const int64_t* axis_dimension_input_output_mapping, int num_outputs, const void* input_data,
                 void* const* output_data, size_t input_size);

}  // namespace musa
}  // namespace onnxruntime
