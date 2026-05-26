// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <musa_runtime.h>
#include <stdint.h>

#include "core/providers/musa/shared_inc/fast_divmod.h"

namespace onnxruntime {
namespace musa {

template <typename T>
void CumSumImpl(musaStream_t stream,
                const T* input_data,
                const fast_divmod& input_dim_along_axis,
                const fast_divmod& input_stride_along_axis,
                T* output_data,
                int64_t output_size,
                bool exclusive,
                bool reverse);

void CumSumImplHalf(musaStream_t stream,
                    const void* input_data,
                    const fast_divmod& input_dim_along_axis,
                    const fast_divmod& input_stride_along_axis,
                    void* output_data,
                    int64_t output_size,
                    bool exclusive,
                    bool reverse);

}  // namespace musa
}  // namespace onnxruntime
