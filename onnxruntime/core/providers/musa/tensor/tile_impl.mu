// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include <musa_runtime.h>
#include <stdint.h>
#include <algorithm>
#include "tile_impl.h"
#include "core/providers/musa/shared_inc/fast_divmod.h"

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kMaxBlocks = 65535;

template <typename T, int vec_size>
struct alignas(sizeof(T) * vec_size) aligned_vector {
  T val[vec_size];
};

template <typename INT, typename INT2>
__host__ __device__ INT CeilDiv(INT a, INT2 b) {
  return (INT)(((size_t)a + (size_t)b - 1) / (size_t)b);
}

}  // namespace

template <typename T>
__global__ void TileKernel(
    const T* input_data,
    T* output_data,
    onnxruntime::musa::TileKernelParams params) {

  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total_threads = gridDim.x * blockDim.x;

  for (int64_t out_idx = tid; out_idx < params.total_elements;
       out_idx += total_threads) {

    int64_t input_idx = 0;
    int64_t remaining = out_idx;

    for (int32_t dim = 0; dim < params.rank; dim++) {
      int64_t out_coord = remaining / params.output_strides[dim];
      remaining = remaining % params.output_strides[dim];
      int64_t in_coord = out_coord % params.input_dims[dim];
      input_idx += in_coord * params.input_strides[dim];
    }

    output_data[out_idx] = input_data[input_idx];
  }
}

template <typename T>
__global__ void TileMemcpyKernelFromInput(
    const T* input_data,
    T* output_data,
    int64_t N,
    size_t repeats) {
  int64_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= N) return;

  T input_val = input_data[id];
  for (size_t i = 0; i < repeats; ++i) {
    output_data[id] = input_val;
    id += N;
  }
}

template <typename T>
__global__ void TileMemcpyKernelFromOutput(
    const T* input_data,
    T* output_data,
    onnxruntime::musa::fast_divmod divmod_num_input_elements,
    int64_t N) {
  int64_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= N) return;

  output_data[id] = input_data[divmod_num_input_elements.mod(static_cast<int>(id))];
}

template <typename T>
__global__ void TileBatchedMemcpyKernelFromInput(
    const T* input_data,
    T* output_data,
    onnxruntime::musa::fast_divmod divmod_size_input_row,
    int64_t size_input_row,
    int64_t size_output_row,
    int64_t size_output_batch,
    size_t batch_repeats,
    size_t repeats_per_batch,
    int64_t N) {
  int64_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= N) return;

  T input_val = input_data[id];
  int q, r;
  divmod_size_input_row.divmod(static_cast<int>(id), q, r);
  int64_t batch_offset = q * size_output_row + r;

  for (size_t i = 0; i < batch_repeats; ++i) {
    int64_t offset = batch_offset;
    for (size_t j = 0; j < repeats_per_batch; ++j) {
      output_data[offset] = input_val;
      offset += size_input_row;
    }
    batch_offset += size_output_batch;
  }
}

template <typename T>
__global__ void TileBatchedMemcpyKernelFromOutput(
    const T* input_data,
    T* output_data,
    onnxruntime::musa::fast_divmod divmod_size_output_row,
    size_t size_input_row,
    onnxruntime::musa::fast_divmod divmod_batch,
    onnxruntime::musa::fast_divmod divmod_size_input_row,
    int64_t N) {
  int64_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= N) return;

  int batch_idx, element_idx;
  divmod_size_output_row.divmod(static_cast<int>(id), batch_idx, element_idx);
  int64_t input_idx = divmod_batch.mod(batch_idx) * size_input_row +
                      divmod_size_input_row.mod(element_idx);
  output_data[id] = input_data[input_idx];
}

template <typename T>
void onnxruntime::musa::LaunchTileKernel(
    const T* input_data,
    T* output_data,
    const TileKernelParams& params,
    musaStream_t stream) {

  if (params.total_elements == 0) return;

  int64_t blocks = CeilDiv(params.total_elements, static_cast<int64_t>(kThreadsPerBlock));
  blocks = std::min(blocks, static_cast<int64_t>(kMaxBlocks));

  TileKernel<T><<<static_cast<int>(blocks), kThreadsPerBlock, 0, stream>>>(
      input_data, output_data, params);
}

template <typename T>
void onnxruntime::musa::TileMemcpyImpl(
    musaStream_t stream,
    const T* input_data,
    T* output_data,
    size_t num_input_elements,
    size_t repeats) {

  if (num_input_elements == 0) return;

  int64_t N = static_cast<int64_t>(num_input_elements);
  int blocksPerGrid = static_cast<int>(CeilDiv(N, static_cast<int64_t>(kThreadsPerBlock)));

  if (blocksPerGrid < 128) {
    int64_t total_elements = N * static_cast<int64_t>(repeats);
    blocksPerGrid = static_cast<int>(CeilDiv(total_elements, static_cast<int64_t>(kThreadsPerBlock)));
    TileMemcpyKernelFromOutput<T><<<blocksPerGrid, kThreadsPerBlock, 0, stream>>>(
        input_data, output_data,
        fast_divmod(static_cast<int>(num_input_elements)),
        total_elements);
    return;
  }

  TileMemcpyKernelFromInput<T><<<blocksPerGrid, kThreadsPerBlock, 0, stream>>>(
      input_data, output_data, N, repeats);
}

template <typename T>
void onnxruntime::musa::TileBatchedMemcpyImpl(
    musaStream_t stream,
    const T* input_data,
    T* output_data,
    size_t size_input_row,
    size_t num_input_elements,
    size_t batch_repeats,
    size_t repeats_per_batch) {

  if (num_input_elements == 0) return;

  int64_t N = static_cast<int64_t>(num_input_elements);
  int blocksPerGrid = static_cast<int>(CeilDiv(N, static_cast<int64_t>(kThreadsPerBlock)));

  if (blocksPerGrid < 128) {
    int64_t total_elements = N * static_cast<int64_t>(batch_repeats * repeats_per_batch);
    blocksPerGrid = static_cast<int>(CeilDiv(total_elements, static_cast<int64_t>(kThreadsPerBlock)));
    TileBatchedMemcpyKernelFromOutput<T><<<blocksPerGrid, kThreadsPerBlock, 0, stream>>>(
        input_data, output_data,
        fast_divmod(static_cast<int>(size_input_row * repeats_per_batch)),
        size_input_row,
        fast_divmod(static_cast<int>(num_input_elements / size_input_row)),
        fast_divmod(static_cast<int>(size_input_row)),
        total_elements);
    return;
  }

  int64_t size_output_row = static_cast<int64_t>(size_input_row * repeats_per_batch);
  int64_t size_output_batch = N * static_cast<int64_t>(repeats_per_batch);

  TileBatchedMemcpyKernelFromInput<T><<<blocksPerGrid, kThreadsPerBlock, 0, stream>>>(
      input_data, output_data,
      fast_divmod(static_cast<int>(size_input_row)),
      static_cast<int64_t>(size_input_row),
      size_output_row,
      size_output_batch,
      batch_repeats,
      repeats_per_batch,
      N);
}

template void onnxruntime::musa::LaunchTileKernel<float>(
    const float*, float*, const onnxruntime::musa::TileKernelParams&, musaStream_t);

template void onnxruntime::musa::LaunchTileKernel<double>(
    const double*, double*, const onnxruntime::musa::TileKernelParams&, musaStream_t);

template void onnxruntime::musa::LaunchTileKernel<int32_t>(
    const int32_t*, int32_t*, const onnxruntime::musa::TileKernelParams&, musaStream_t);

template void onnxruntime::musa::LaunchTileKernel<int64_t>(
    const int64_t*, int64_t*, const onnxruntime::musa::TileKernelParams&, musaStream_t);

template void onnxruntime::musa::LaunchTileKernel<int16_t>(
    const int16_t*, int16_t*, const onnxruntime::musa::TileKernelParams&, musaStream_t);

template void onnxruntime::musa::LaunchTileKernel<int8_t>(
    const int8_t*, int8_t*, const onnxruntime::musa::TileKernelParams&, musaStream_t);

template void onnxruntime::musa::LaunchTileKernel<uint8_t>(
    const uint8_t*, uint8_t*, const onnxruntime::musa::TileKernelParams&, musaStream_t);

template void onnxruntime::musa::TileMemcpyImpl<float>(
    musaStream_t, const float*, float*, size_t, size_t);

template void onnxruntime::musa::TileMemcpyImpl<double>(
    musaStream_t, const double*, double*, size_t, size_t);

template void onnxruntime::musa::TileBatchedMemcpyImpl<float>(
    musaStream_t, const float*, float*, size_t, size_t, size_t, size_t);

template void onnxruntime::musa::TileBatchedMemcpyImpl<double>(
    musaStream_t, const double*, double*, size_t, size_t, size_t, size_t);

#include "core/common/common.h"
#include "core/common/float16.h"

template void onnxruntime::musa::LaunchTileKernel<onnxruntime::MLFloat16>(
    const onnxruntime::MLFloat16*, onnxruntime::MLFloat16*,
    const onnxruntime::musa::TileKernelParams&, musaStream_t);

template void onnxruntime::musa::TileMemcpyImpl<onnxruntime::MLFloat16>(
    musaStream_t, const onnxruntime::MLFloat16*, onnxruntime::MLFloat16*, size_t, size_t);

template void onnxruntime::musa::TileBatchedMemcpyImpl<onnxruntime::MLFloat16>(
    musaStream_t, const onnxruntime::MLFloat16*, onnxruntime::MLFloat16*,
    size_t, size_t, size_t, size_t);

template void onnxruntime::musa::TileMemcpyImpl<uint16_t>(
    musaStream_t, const uint16_t*, uint16_t*, size_t, size_t);

template void onnxruntime::musa::TileBatchedMemcpyImpl<uint16_t>(
    musaStream_t, const uint16_t*, uint16_t*, size_t, size_t, size_t, size_t);
