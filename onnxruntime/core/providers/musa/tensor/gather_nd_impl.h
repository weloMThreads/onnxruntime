// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <musa_runtime.h>

#include "core/common/common.h"

namespace onnxruntime {
namespace musa {

template <typename T, int32_t capacity = 8>
struct GatherNDFixedArray {
  GatherNDFixedArray() = default;

  void SetSize(int32_t size) {
    ORT_ENFORCE(0 <= size && size <= capacity,
                "GatherNDFixedArray size must be within range [0, ", capacity, "]. Actual: ", size);
    size_ = size;
  }

  __host__ __device__ int32_t Size() const { return size_; }
  __host__ __device__ T& operator[](int32_t index) { return data_[index]; }
  __host__ __device__ const T& operator[](int32_t index) const { return data_[index]; }
  static constexpr int32_t Capacity() { return capacity; }

 private:
  int32_t size_ = 0;
  T data_[capacity] = {};
};

struct GatherNDKernelArgs {
  int64_t batch_dims = 0;
  int64_t num_slice_dims = 0;
  int64_t num_slices = 0;
  int64_t num_batches = 0;
  int64_t num_slices_per_batch = 0;
  int64_t input_batch_stride = 0;
  int64_t slice_size = 0;
  GatherNDFixedArray<int64_t> input_dims;
  GatherNDFixedArray<int64_t> slice_strides;
};

template <typename TIndex>
void GatherNDImpl(musaStream_t stream,
                  const GatherNDKernelArgs& args,
                  const void* indices_data,
                  const void* input_data,
                  void* output_data,
                  size_t element_size,
                  size_t output_count);

extern template void GatherNDImpl<int64_t>(musaStream_t stream,
                                           const GatherNDKernelArgs& args,
                                           const void* indices_data,
                                           const void* input_data,
                                           void* output_data,
                                           size_t element_size,
                                           size_t output_count);

}  // namespace musa
}  // namespace onnxruntime
