// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <musa_runtime.h>

namespace onnxruntime {
namespace contrib {
namespace musa {

template <typename T>
void LaunchQuickGeluKernel(musaStream_t stream, int64_t input_size,
                           const T* input, T* output, float alpha);

// MLFloat16 specialization (uses __half internally)
void LaunchQuickGeluKernelHalf(musaStream_t stream, int64_t input_size,
                               const void* input, void* output, float alpha);

}  // namespace musa
}  // namespace contrib
}  // namespace onnxruntime
