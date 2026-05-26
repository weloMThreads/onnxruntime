// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <musa_fp16.h>
#include <musa_runtime.h>
#include <stdint.h>

#include "core/providers/musa/math/cumsum_impl.h"

#define CUDA_LONG int32_t

struct GridDim {
  enum : CUDA_LONG {
    maxThreadsPerBlock = 256,
  };
};

#define CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N)        \
  CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;   \
  if (id >= N) return;

namespace onnxruntime {
namespace musa {

template <typename T>
__global__ void CumSumKernel(const T* input_data,
                             const fast_divmod input_dim_along_axis,
                             const fast_divmod input_stride_along_axis,
                             T* output_data,
                             const int64_t output_size,
                             const bool exclusive,
                             const bool reverse) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(indices_index, output_size);

  const int input_dim = static_cast<int>(input_dim_along_axis.d_);
  const int input_stride = static_cast<int>(input_stride_along_axis.d_);

  int axis_dim = 0;
  int div = input_stride_along_axis.div(static_cast<int>(indices_index));
  input_dim_along_axis.divmod(div, div, axis_dim);

  int start = 0;
  int end = 0;
  if (!reverse && !exclusive) {
    start = 0;
    end = axis_dim;
  } else if (reverse && !exclusive) {
    start = axis_dim;
    end = input_dim - 1;
  } else if (!reverse && exclusive) {
    start = 0;
    end = axis_dim - 1;
  } else {
    start = axis_dim + 1;
    end = input_dim - 1;
  }

  int count = end - start + 1;
  if (count <= 0) {
    output_data[indices_index] = T{};
    return;
  }

  int data_index = static_cast<int>(indices_index) + (start - axis_dim) * input_stride;
  T sum = T{};
  while (count != 0) {
    sum += input_data[data_index];
    data_index += input_stride;
    --count;
  }

  output_data[indices_index] = sum;
}

template <typename T>
void CumSumImpl(musaStream_t stream,
                const T* input_data,
                const fast_divmod& input_dim_along_axis,
                const fast_divmod& input_stride_along_axis,
                T* output_data,
                int64_t output_size,
                bool exclusive,
                bool reverse) {
  if (output_size <= 0) {
    return;
  }

  const int blocks_per_grid = static_cast<int>(
      (output_size + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock);
  CumSumKernel<T><<<blocks_per_grid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input_data,
      input_dim_along_axis,
      input_stride_along_axis,
      output_data,
      output_size,
      exclusive,
      reverse);
}

void CumSumImplHalf(musaStream_t stream,
                    const void* input_data,
                    const fast_divmod& input_dim_along_axis,
                    const fast_divmod& input_stride_along_axis,
                    void* output_data,
                    int64_t output_size,
                    bool exclusive,
                    bool reverse) {
  CumSumImpl<half>(stream,
                   static_cast<const half*>(input_data),
                   input_dim_along_axis,
                   input_stride_along_axis,
                   static_cast<half*>(output_data),
                   output_size,
                   exclusive,
                   reverse);
}

template void CumSumImpl<int32_t>(musaStream_t,
                                  const int32_t*,
                                  const fast_divmod&,
                                  const fast_divmod&,
                                  int32_t*,
                                  int64_t,
                                  bool,
                                  bool);

template void CumSumImpl<int64_t>(musaStream_t,
                                  const int64_t*,
                                  const fast_divmod&,
                                  const fast_divmod&,
                                  int64_t*,
                                  int64_t,
                                  bool,
                                  bool);

template void CumSumImpl<uint32_t>(musaStream_t,
                                   const uint32_t*,
                                   const fast_divmod&,
                                   const fast_divmod&,
                                   uint32_t*,
                                   int64_t,
                                   bool,
                                   bool);

template void CumSumImpl<uint64_t>(musaStream_t,
                                   const uint64_t*,
                                   const fast_divmod&,
                                   const fast_divmod&,
                                   uint64_t*,
                                   int64_t,
                                   bool,
                                   bool);

template void CumSumImpl<float>(musaStream_t,
                                const float*,
                                const fast_divmod&,
                                const fast_divmod&,
                                float*,
                                int64_t,
                                bool,
                                bool);

template void CumSumImpl<double>(musaStream_t,
                                 const double*,
                                 const fast_divmod&,
                                 const fast_divmod&,
                                 double*,
                                 int64_t,
                                 bool,
                                 bool);

template void CumSumImpl<half>(musaStream_t,
                               const half*,
                               const fast_divmod&,
                               const fast_divmod&,
                               half*,
                               int64_t,
                               bool,
                               bool);

}  // namespace musa
}  // namespace onnxruntime
