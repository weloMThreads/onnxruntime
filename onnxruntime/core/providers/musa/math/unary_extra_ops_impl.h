// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <musa_runtime.h>
#include <stdint.h>

namespace onnxruntime {
namespace musa {

template <typename T>
void LaunchReciprocalKernel(musaStream_t stream, const T* input, T* output, int64_t count);
void LaunchReciprocalKernelHalf(musaStream_t stream, const void* input, void* output, int64_t count);

template <typename T>
void LaunchLog1pKernel(musaStream_t stream, const T* input, T* output, int64_t count);
void LaunchLog1pKernelHalf(musaStream_t stream, const void* input, void* output, int64_t count);

template <typename T>
void LaunchExpm1Kernel(musaStream_t stream, const T* input, T* output, int64_t count);
void LaunchExpm1KernelHalf(musaStream_t stream, const void* input, void* output, int64_t count);

template <typename T>
void LaunchSquareKernel(musaStream_t stream, const T* input, T* output, int64_t count);
void LaunchSquareKernelHalf(musaStream_t stream, const void* input, void* output, int64_t count);

template <typename T>
void LaunchRsqrtKernel(musaStream_t stream, const T* input, T* output, int64_t count);
void LaunchRsqrtKernelHalf(musaStream_t stream, const void* input, void* output, int64_t count);

template <typename T>
void LaunchFloorKernel(musaStream_t stream, const T* input, T* output, int64_t count);
void LaunchFloorKernelHalf(musaStream_t stream, const void* input, void* output, int64_t count);

template <typename T>
void LaunchCeilKernel(musaStream_t stream, const T* input, T* output, int64_t count);
void LaunchCeilKernelHalf(musaStream_t stream, const void* input, void* output, int64_t count);

template <typename T>
void LaunchSignKernel(musaStream_t stream, const T* input, T* output, int64_t count);
void LaunchSignKernelHalf(musaStream_t stream, const void* input, void* output, int64_t count);

template <typename T>
void LaunchIsNaNKernel(musaStream_t stream, const T* input, bool* output, int64_t count);
void LaunchIsNaNKernelHalf(musaStream_t stream, const void* input, bool* output, int64_t count);

}  // namespace musa
}  // namespace onnxruntime
