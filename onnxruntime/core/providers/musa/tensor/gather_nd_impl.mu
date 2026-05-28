// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <musa_runtime.h>
#include <stddef.h>
#include <stdint.h>

#include "core/providers/musa/tensor/gather_nd_impl.h"

namespace onnxruntime {
namespace musa {
namespace {

constexpr int kThreadsPerBlock = 256;

template <typename T, typename TIndex>
__global__ void GatherNDKernel(const GatherNDKernelArgs args,
                               const TIndex* indices_data,
                               const T* input_data,
                               T* output_data,
                               int64_t output_count) {
  const int64_t id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (id >= output_count) {
    return;
  }

  const int64_t slice_idx = id / args.slice_size;
  const int64_t slice_offset = id % args.slice_size;
  const int64_t batch_idx = slice_idx / args.num_slices_per_batch;

  int64_t input_offset = batch_idx * args.input_batch_stride + slice_offset;
  const TIndex* slice_indices = indices_data + slice_idx * args.num_slice_dims;
  for (int64_t dim_idx = 0; dim_idx < args.num_slice_dims; ++dim_idx) {
    const int64_t input_dim_idx = args.batch_dims + dim_idx;
    const int64_t dim_size = args.input_dims[static_cast<int32_t>(input_dim_idx)];
    int64_t index = static_cast<int64_t>(slice_indices[dim_idx]);
    if (index < 0) {
      index += dim_size;
    }
    input_offset += index * args.slice_strides[static_cast<int32_t>(dim_idx)];
  }

  output_data[id] = input_data[input_offset];
}

template <typename T, typename TIndex>
void LaunchGatherNDKernel(musaStream_t stream,
                          const GatherNDKernelArgs& args,
                          const void* indices_data,
                          const void* input_data,
                          void* output_data,
                          size_t output_count) {
  if (output_count == 0) {
    return;
  }
  const int blocks = static_cast<int>((output_count + kThreadsPerBlock - 1) / kThreadsPerBlock);
  GatherNDKernel<T, TIndex><<<blocks, kThreadsPerBlock, 0, stream>>>(
      args,
      reinterpret_cast<const TIndex*>(indices_data),
      reinterpret_cast<const T*>(input_data),
      reinterpret_cast<T*>(output_data),
      static_cast<int64_t>(output_count));
}

template <typename TIndex>
void DispatchGatherNDKernel(musaStream_t stream,
                            const GatherNDKernelArgs& args,
                            const void* indices_data,
                            const void* input_data,
                            void* output_data,
                            size_t element_size,
                            size_t output_count) {
  switch (element_size) {
    case sizeof(uint8_t):
      LaunchGatherNDKernel<uint8_t, TIndex>(stream, args, indices_data, input_data, output_data, output_count);
      break;
    case sizeof(uint16_t):
      LaunchGatherNDKernel<uint16_t, TIndex>(stream, args, indices_data, input_data, output_data, output_count);
      break;
    case sizeof(uint32_t):
      LaunchGatherNDKernel<uint32_t, TIndex>(stream, args, indices_data, input_data, output_data, output_count);
      break;
    case sizeof(uint64_t):
      LaunchGatherNDKernel<uint64_t, TIndex>(stream, args, indices_data, input_data, output_data, output_count);
      break;
    default:
      break;
  }
}

}  // namespace

template <typename TIndex>
void GatherNDImpl(musaStream_t stream,
                  const GatherNDKernelArgs& args,
                  const void* indices_data,
                  const void* input_data,
                  void* output_data,
                  size_t element_size,
                  size_t output_count) {
  DispatchGatherNDKernel<TIndex>(stream, args, indices_data, input_data, output_data, element_size, output_count);
}

template void GatherNDImpl<int64_t>(musaStream_t stream,
                                    const GatherNDKernelArgs& args,
                                    const void* indices_data,
                                    const void* input_data,
                                    void* output_data,
                                    size_t element_size,
                                    size_t output_count);

}  // namespace musa
}  // namespace onnxruntime
