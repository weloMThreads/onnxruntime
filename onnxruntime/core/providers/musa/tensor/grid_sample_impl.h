// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <musa_runtime.h>

namespace onnxruntime {
namespace contrib {
namespace musa {

template <typename T, bool IsNHWC>
void GridSampleImpl(
    musaStream_t stream,
    const T* input_data,
    const T* grid_data,
    const int64_t mode,
    const int64_t padding_mode,
    const int64_t align_corners,
    const int64_t dims_input[4],
    const int64_t H_out,
    const int64_t W_out,
    T* output_data);

// C-linkage wrapper for half type (callable from host .cc code without half type)
// IsNHWC = false (NCHW) and true (NHWC) variants
void GridSampleImplHalf(
    musaStream_t stream,
    const void* input_data,
    const void* grid_data,
    const int64_t mode,
    const int64_t padding_mode,
    const int64_t align_corners,
    const int64_t dims_input[4],
    const int64_t H_out,
    const int64_t W_out,
    void* output_data,
    bool is_nhwc);

}  // namespace musa
}  // namespace contrib
}  // namespace onnxruntime
