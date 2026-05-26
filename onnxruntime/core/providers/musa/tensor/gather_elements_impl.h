// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <musa_runtime.h>

#include "core/common/common.h"
#include "core/providers/musa/shared_inc/fast_divmod.h"

namespace onnxruntime {
namespace musa {

template <typename T, int32_t capacity = 8>
struct TArray {
  TArray() = default;
  TArray(const TArray&) = default;
  TArray& operator=(const TArray&) = default;

  explicit TArray(int32_t size) : size_(size), data_() {
    ORT_ENFORCE(0 <= size && size <= capacity,
                "TArray size must be within range [0, ", capacity, "]. Actual: ", size);
  }

  void SetSize(int32_t size) {
    ORT_ENFORCE(0 <= size && size <= capacity,
                "TArray size must be within range [0, ", capacity, "]. Actual: ", size);
    size_ = size;
  }

  __host__ __device__ int32_t Size() const { return size_; }
  __host__ __device__ T& operator[](int32_t index) { return data_[index]; }
  __host__ __device__ __forceinline__ const T& operator[](int32_t index) const { return data_[index]; }
  __host__ __device__ T* Data() { return data_; }
  __host__ __device__ const T* Data() const { return data_; }
  static constexpr int32_t Capacity() { return capacity; }

 private:
  int32_t size_ = 0;
  T data_[capacity] = {};
};

struct GatherScatterElementsArgs {
  int64_t rank;
  int64_t axis;
  int64_t input_size;
  int64_t input_dim_along_axis;
  int64_t input_stride_along_axis;
  TArray<int64_t> masked_input_strides;
  TArray<fast_divmod> indices_fdms;
  TArray<int64_t> indices_strides;
  int64_t indices_size;
};

template <typename T, typename TIndex>
void GatherElementsImpl(musaStream_t stream, const T* input_data, const TIndex* indices_data, T* output_data,
                        const GatherScatterElementsArgs& args);

void GatherElementsImplHalf(musaStream_t stream, const void* input_data, const void* indices_data, void* output_data,
                            size_t index_element_size, const GatherScatterElementsArgs& args);

}  // namespace musa
}  // namespace onnxruntime
