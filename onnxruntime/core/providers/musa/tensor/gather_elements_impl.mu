// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <musa_fp16.h>
#include <musa_runtime.h>
#include <stddef.h>
#include <stdint.h>

#include "core/providers/musa/tensor/gather_elements_impl.h"
#include "core/providers/musa/shared_inc/fast_divmod.h"

#define CUDA_LONG int32_t

struct GridDim {
  enum : CUDA_LONG {
    maxThreadsPerBlock = 256,
  };
};

#define CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N) \
  CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x; \
  if (id >= N) return;

namespace onnxruntime {
namespace musa {

using onnxruntime::musa::fast_divmod;

namespace {

constexpr int kThreadsPerBlock = GridDim::maxThreadsPerBlock;
constexpr int kThreadWorkSize = 4;

inline int CeilDiv(int a, int b) {
  return (a + b - 1) / b;
}

template <bool IsStridedIndices>
struct OffsetCalculator {
  OffsetCalculator(const int rank, const TArray<int64_t>& masked_input_strides, const TArray<fast_divmod>& indices_fdms,
                   const TArray<int64_t>& indices_strides)
      : rank_(rank), indices_fdms_(indices_fdms) {
    masked_input_strides_.SetSize(rank);
    if (IsStridedIndices) {
      indices_strides_.SetSize(rank);
    }
    for (int dim = 0; dim < rank; ++dim) {
      masked_input_strides_[dim] = static_cast<CUDA_LONG>(masked_input_strides[dim]);
      if (IsStridedIndices) {
        indices_strides_[dim] = static_cast<CUDA_LONG>(indices_strides[dim]);
      }
    }
  }

  __device__ __forceinline__ TArray<CUDA_LONG, 2> get(CUDA_LONG linear_idx) const {
    TArray<CUDA_LONG, 2> offsets;
    offsets[0] = 0;
    offsets[1] = IsStridedIndices ? 0 : linear_idx;
    CUDA_LONG q, r = linear_idx;
#pragma unroll
    for (int dim = 0; dim < indices_fdms_.Capacity(); ++dim) {
      if (dim == rank_) {
        break;
      }
      indices_fdms_[dim].divmod(r, q, r);
      offsets[0] += masked_input_strides_[dim] * q;
      if (IsStridedIndices) {
        offsets[1] += indices_strides_[dim] * q;
      }
    }
    return offsets;
  }

  int rank_;
  TArray<fast_divmod> indices_fdms_;
  TArray<CUDA_LONG> masked_input_strides_;
  TArray<CUDA_LONG> indices_strides_;
};

template <bool IsOuterAxis, bool IsStridedIndices>
struct OffsetCalculatorFor2D {
  OffsetCalculatorFor2D(const fast_divmod indices_row_size_fdm, const int64_t input_row_size,
                        const TArray<int64_t>& indices_strides)
      : indices_row_size_fdm_(indices_row_size_fdm), input_row_size_(static_cast<CUDA_LONG>(input_row_size)) {
    if (IsStridedIndices) {
      indices_strides_.SetSize(2);
      indices_strides_[0] = static_cast<CUDA_LONG>(indices_strides[0]);
      indices_strides_[1] = static_cast<CUDA_LONG>(indices_strides[1]);
    }
  }

  __device__ __forceinline__ TArray<CUDA_LONG, 2> get(CUDA_LONG linear_idx) const {
    TArray<CUDA_LONG, 2> offsets;
    if (IsStridedIndices) {
      CUDA_LONG q, r = linear_idx;
      indices_row_size_fdm_.divmod(r, q, r);
      offsets[0] = IsOuterAxis ? r : q * input_row_size_;
      offsets[1] = q * indices_strides_[0] + r * indices_strides_[1];
    } else {
      offsets[0] =
          IsOuterAxis ? indices_row_size_fdm_.mod(linear_idx) : indices_row_size_fdm_.div(linear_idx) * input_row_size_;
      offsets[1] = linear_idx;
    }
    return offsets;
  }

  fast_divmod indices_row_size_fdm_;
  CUDA_LONG input_row_size_;
  TArray<CUDA_LONG> indices_strides_;
};

}  // namespace

template <class T>
struct FuncAssignment {
  __device__ __inline__ void operator()(T* start_addr, size_t index, T value) const {
    start_addr[index] = value;
  }
};

template <typename T, typename TIndex, typename OffsetCalcT, typename TFunc>
__global__ void _GatherScatterElementsKernel(const T* src_data, const TIndex* indices_data, T* output_data,
                                             const int64_t input_dim_along_axis, const int64_t input_stride_along_axis,
                                             const OffsetCalcT offset_calc, const TFunc func, CUDA_LONG N) {
  CUDA_LONG start = kThreadsPerBlock * kThreadWorkSize * blockIdx.x + threadIdx.x;
  CUDA_LONG id = start;
  T value[kThreadWorkSize] = {};

#pragma unroll
  for (int work = 0; work < kThreadWorkSize; ++work) {
    if (id < N) {
      TArray<CUDA_LONG, 2> offsets = offset_calc.get(id);
      int64_t input_offset_along_axis = static_cast<int64_t>(indices_data[offsets[1]]);
      if (input_offset_along_axis >= -input_dim_along_axis && input_offset_along_axis < input_dim_along_axis) {
        if (input_offset_along_axis < 0) {
          input_offset_along_axis += input_dim_along_axis;
        }
        CUDA_LONG input_offset = offsets[0] + static_cast<CUDA_LONG>(input_offset_along_axis * input_stride_along_axis);
        func(value, static_cast<size_t>(work), src_data[input_offset]);
      }

      id += kThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int work = 0; work < kThreadWorkSize; ++work) {
    if (id < N) {
      output_data[id] = value[work];
      id += kThreadsPerBlock;
    }
  }
}

#define LAUNCH_GATHER_SCATTER_ELEMENTS_2D_KERNEL(is_outer_axis, is_strided_indices)                                  \
  auto offset_calc = OffsetCalculatorFor2D<is_outer_axis, is_strided_indices>(args.indices_fdms[0], input_row_size, \
                                                                              args.indices_strides);                 \
  _GatherScatterElementsKernel<T, TIndex, decltype(offset_calc), decltype(func)>                                     \
      <<<blocksPerGrid, kThreadsPerBlock, 0, stream>>>(input_data, indices_data, output_data, args.input_dim_along_axis, \
                                                       args.input_stride_along_axis, offset_calc, func, N)

#define LAUNCH_GATHER_SCATTER_ELEMENTS_KERNEL(is_strided_indices)                                                    \
  auto offset_calc =                                                                                                  \
      OffsetCalculator<is_strided_indices>(rank, args.masked_input_strides, args.indices_fdms, args.indices_strides); \
  _GatherScatterElementsKernel<T, TIndex, decltype(offset_calc), decltype(func)>                                     \
      <<<blocksPerGrid, kThreadsPerBlock, 0, stream>>>(input_data, indices_data, output_data, args.input_dim_along_axis, \
                                                       args.input_stride_along_axis, offset_calc, func, N)

#define HANDLE_GATHER_SCATTER_ELEMENTS_2D_IS_STRIDED_INDICES(is_outer_axis) \
  if (args.indices_strides.Size() > 0) {                                     \
    LAUNCH_GATHER_SCATTER_ELEMENTS_2D_KERNEL(is_outer_axis, true);           \
  } else {                                                                   \
    LAUNCH_GATHER_SCATTER_ELEMENTS_2D_KERNEL(is_outer_axis, false);          \
  }

template <typename T, typename TIndex>
void GatherElementsImpl(musaStream_t stream, const T* input_data, const TIndex* indices_data, T* output_data,
                        const GatherScatterElementsArgs& args) {
  CUDA_LONG N = static_cast<CUDA_LONG>(args.indices_size);
  int blocksPerGrid = CeilDiv(static_cast<int>(N), kThreadsPerBlock * kThreadWorkSize);
  auto func = FuncAssignment<T>();
  if (args.rank == 2) {
    int64_t input_row_size = args.masked_input_strides[0];
    if (args.axis == 0) {
      HANDLE_GATHER_SCATTER_ELEMENTS_2D_IS_STRIDED_INDICES(true);
    } else {
      HANDLE_GATHER_SCATTER_ELEMENTS_2D_IS_STRIDED_INDICES(false);
    }
    return;
  }

  int rank = static_cast<int>(args.rank);
  if (args.indices_strides.Size() > 0) {
    LAUNCH_GATHER_SCATTER_ELEMENTS_KERNEL(true);
  } else {
    if (args.rank == args.axis + 1) {
      rank -= 1;
    }
    LAUNCH_GATHER_SCATTER_ELEMENTS_KERNEL(false);
  }
}

void GatherElementsImplHalf(musaStream_t stream, const void* input_data, const void* indices_data, void* output_data,
                            size_t index_element_size, const GatherScatterElementsArgs& args) {
  switch (index_element_size) {
    case sizeof(int32_t):
      GatherElementsImpl<half, int32_t>(stream,
                                        static_cast<const half*>(input_data),
                                        static_cast<const int32_t*>(indices_data),
                                        static_cast<half*>(output_data),
                                        args);
      break;
    case sizeof(int64_t):
      GatherElementsImpl<half, int64_t>(stream,
                                        static_cast<const half*>(input_data),
                                        static_cast<const int64_t*>(indices_data),
                                        static_cast<half*>(output_data),
                                        args);
      break;
    default:
      break;
  }
}

#define GATHER_ELEMENTS_SPECIALIZED_TINDEX_IMPL(T, TIndex)                                                         \
  template void GatherElementsImpl<T, TIndex>(musaStream_t stream, const T* input_data,                           \
                                              const TIndex* indices_data, T* output_data,                          \
                                              const GatherScatterElementsArgs& args);

#define GATHER_ELEMENTS_SPECIALIZED_IMPL(T) \
  GATHER_ELEMENTS_SPECIALIZED_TINDEX_IMPL(T, int32_t) \
  GATHER_ELEMENTS_SPECIALIZED_TINDEX_IMPL(T, int64_t)

GATHER_ELEMENTS_SPECIALIZED_IMPL(int8_t)
GATHER_ELEMENTS_SPECIALIZED_IMPL(int16_t)
GATHER_ELEMENTS_SPECIALIZED_IMPL(half)
GATHER_ELEMENTS_SPECIALIZED_IMPL(float)
GATHER_ELEMENTS_SPECIALIZED_IMPL(double)

}  // namespace musa
}  // namespace onnxruntime
