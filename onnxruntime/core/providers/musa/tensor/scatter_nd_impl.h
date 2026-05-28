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
struct ScatterNDFixedArray {
  ScatterNDFixedArray() = default;

  void SetSize(int32_t size) {
    ORT_ENFORCE(0 <= size && size <= capacity,
                "ScatterNDFixedArray size must be within range [0, ", capacity, "]. Actual: ", size);
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

struct ScatterNDKernelArgs {
  int64_t num_indices = 0;
  int64_t last_index_dimension = 0;
  int64_t slice_size = 0;
  ScatterNDFixedArray<int64_t> input_dims;
  ScatterNDFixedArray<int64_t> input_strides;
};

void ScatterNDImpl(musaStream_t stream,
                   const ScatterNDKernelArgs& args,
                   const int64_t* indices_data,
                   const void* updates_data,
                   void* output_data,
                   size_t element_size,
                   size_t updates_count);

}  // namespace musa
}  // namespace onnxruntime
