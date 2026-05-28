// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <musa_runtime.h>
#include <stddef.h>
#include <stdint.h>

#include "core/providers/musa/tensor/scatter_nd_impl.h"

namespace onnxruntime {
namespace musa {
namespace {

constexpr int kThreadsPerBlock = 256;

template <typename T>
__global__ void ScatterNDUpdateKernel(const ScatterNDKernelArgs args,
                                      const int64_t* indices_data,
                                      const T* updates_data,
                                      T* output_data,
                                      int64_t updates_count) {
  const int64_t id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (id >= updates_count) {
    return;
  }

  const int64_t update_index = id / args.slice_size;
  const int64_t slice_offset = id % args.slice_size;
  int64_t output_offset = slice_offset;

  const int64_t* update_indices = indices_data + update_index * args.last_index_dimension;
  for (int64_t dim_idx = 0; dim_idx < args.last_index_dimension; ++dim_idx) {
    const int64_t dim_size = args.input_dims[static_cast<int32_t>(dim_idx)];
    int64_t index = update_indices[dim_idx];
    if (index < 0) {
      index += dim_size;
    }
    output_offset += index * args.input_strides[static_cast<int32_t>(dim_idx)];
  }

  output_data[output_offset] = updates_data[id];
}

template <typename T>
void LaunchScatterNDUpdateKernel(musaStream_t stream,
                                 const ScatterNDKernelArgs& args,
                                 const int64_t* indices_data,
                                 const void* updates_data,
                                 void* output_data,
                                 size_t updates_count) {
  if (updates_count == 0) {
    return;
  }
  const int blocks = static_cast<int>((updates_count + kThreadsPerBlock - 1) / kThreadsPerBlock);
  ScatterNDUpdateKernel<T><<<blocks, kThreadsPerBlock, 0, stream>>>(
      args,
      indices_data,
      reinterpret_cast<const T*>(updates_data),
      reinterpret_cast<T*>(output_data),
      static_cast<int64_t>(updates_count));
}

}  // namespace

void ScatterNDImpl(musaStream_t stream,
                   const ScatterNDKernelArgs& args,
                   const int64_t* indices_data,
                   const void* updates_data,
                   void* output_data,
                   size_t element_size,
                   size_t updates_count) {
  switch (element_size) {
    case sizeof(uint8_t):
      LaunchScatterNDUpdateKernel<uint8_t>(stream, args, indices_data, updates_data, output_data, updates_count);
      break;
    case sizeof(uint16_t):
      LaunchScatterNDUpdateKernel<uint16_t>(stream, args, indices_data, updates_data, output_data, updates_count);
      break;
    case sizeof(uint32_t):
      LaunchScatterNDUpdateKernel<uint32_t>(stream, args, indices_data, updates_data, output_data, updates_count);
      break;
    case sizeof(uint64_t):
      LaunchScatterNDUpdateKernel<uint64_t>(stream, args, indices_data, updates_data, output_data, updates_count);
      break;
    default:
      break;
  }
}

}  // namespace musa
}  // namespace onnxruntime
