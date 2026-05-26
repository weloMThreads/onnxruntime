// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <musa_runtime.h>
#include <stdint.h>

namespace onnxruntime {
namespace musa {

constexpr int32_t kMaxTileDimensions = 8;

struct TileKernelParams {
  int32_t rank;
  int64_t total_elements;
  int64_t input_dims[kMaxTileDimensions];
  int64_t input_strides[kMaxTileDimensions];
  int64_t output_strides[kMaxTileDimensions];
};

template <typename T>
void LaunchTileKernel(
    const T* input_data,
    T* output_data,
    const TileKernelParams& params,
    musaStream_t stream);

template <typename T>
void TileMemcpyImpl(
    musaStream_t stream,
    const T* input_data,
    T* output_data,
    size_t num_input_elements,
    size_t repeats);

template <typename T>
void TileBatchedMemcpyImpl(
    musaStream_t stream,
    const T* input_data,
    T* output_data,
    size_t size_input_row,
    size_t num_input_elements,
    size_t batch_repeats,
    size_t repeats_per_batch);

extern template void LaunchTileKernel<float>(
    const float*, float*, const TileKernelParams&, musaStream_t);

extern template void LaunchTileKernel<double>(
    const double*, double*, const TileKernelParams&, musaStream_t);

extern template void LaunchTileKernel<int32_t>(
    const int32_t*, int32_t*, const TileKernelParams&, musaStream_t);

extern template void LaunchTileKernel<int64_t>(
    const int64_t*, int64_t*, const TileKernelParams&, musaStream_t);

extern template void LaunchTileKernel<int16_t>(
    const int16_t*, int16_t*, const TileKernelParams&, musaStream_t);

extern template void LaunchTileKernel<int8_t>(
    const int8_t*, int8_t*, const TileKernelParams&, musaStream_t);

extern template void LaunchTileKernel<uint8_t>(
    const uint8_t*, uint8_t*, const TileKernelParams&, musaStream_t);

extern template void TileMemcpyImpl<float>(
    musaStream_t, const float*, float*, size_t, size_t);

extern template void TileMemcpyImpl<double>(
    musaStream_t, const double*, double*, size_t, size_t);

extern template void TileBatchedMemcpyImpl<float>(
    musaStream_t, const float*, float*, size_t, size_t, size_t, size_t);

extern template void TileBatchedMemcpyImpl<double>(
    musaStream_t, const double*, double*, size_t, size_t, size_t, size_t);

extern template void TileMemcpyImpl<uint16_t>(
    musaStream_t, const uint16_t*, uint16_t*, size_t, size_t);

extern template void TileBatchedMemcpyImpl<uint16_t>(
    musaStream_t, const uint16_t*, uint16_t*, size_t, size_t, size_t, size_t);

}  // namespace musa
}  // namespace onnxruntime
