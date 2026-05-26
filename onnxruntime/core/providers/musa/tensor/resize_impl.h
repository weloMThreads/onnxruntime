// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <musa_runtime.h>
#include <stdint.h>

#include <cstring>
#include <type_traits>
#include <vector>

#include <gsl/gsl>

#include "core/common/common.h"
#include "core/providers/cpu/tensor/upsamplebase.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/shared_inc/fast_divmod.h"

namespace onnxruntime {

// AccumulateType<half> only in device compilation (__MCC__ defined by mcc)
#ifdef __MCC__
template <>
struct AccumulateType<half> {
  using type = float;
};
#endif

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

  explicit TArray(const std::vector<T>& vec) : TArray(static_cast<int32_t>(vec.size())) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
    std::memcpy(data_, vec.data(), vec.size() * sizeof(T));
  }

  explicit TArray(gsl::span<const T> vec) : TArray(static_cast<int32_t>(vec.size())) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
    std::memcpy(data_, vec.data(), vec.size() * sizeof(T));
  }

  void SetSize(int32_t size) {
    ORT_ENFORCE(0 <= size && size <= capacity,
                "TArray size must be within range [0, ", capacity, "]. Actual: ", size);
    size_ = size;
  }

  __host__ __device__ int32_t Size() const {
    return size_;
  }

  __host__ __device__ T& operator[](int32_t index) {
    return data_[index];
  }

  __host__ __device__ __forceinline__ const T& operator[](int32_t index) const {
    return data_[index];
  }

  __host__ __device__ T* Data() {
    return data_;
  }

  __host__ __device__ const T* Data() const {
    return data_;
  }

  static constexpr int32_t Capacity() { return capacity; }

 private:
  int32_t size_ = 0;
  T data_[capacity] = {};
};

struct TransformCoordinate_ASYMMETRIC {
  __device__ __host__ __forceinline__ float operator()(float x_resized, float x_scale,
                                                       float, float, float, float) const {
    return x_resized / x_scale;
  }
};

struct TransformCoordinate_HALF_PIXEL {
  __device__ __host__ __forceinline__ float operator()(float x_resized, float x_scale,
                                                       float, float, float, float) const {
    return ((x_resized + 0.5f) / x_scale) - 0.5f;
  }
};

struct TransformCoordinate_PYTORCH_HALF_PIXEL {
  __device__ __host__ __forceinline__ float operator()(float x_resized, float x_scale, float length_resized, float,
                                                       float, float) const {
    return length_resized > 1 ? (x_resized + 0.5f) / x_scale - 0.5f : 0.0f;
  }
};

struct TransformCoordinate_TF_HALF_PIXEL_FOR_NN {
  __device__ __host__ __forceinline__ float operator()(float x_resized, float x_scale,
                                                       float, float, float, float) const {
    return (x_resized + 0.5f) / x_scale;
  }
};

struct TransformCoordinate_ALIGN_CORNERS {
  __device__ __host__ __forceinline__ float operator()(float x_resized, float, float length_resized,
                                                       float length_original, float, float) const {
    return length_resized == 1 ? 0 : x_resized * (length_original - 1) / (length_resized - 1);
  }
};

struct TransformCoordinate_TF_CROP_AND_RESIZE {
  __device__ __host__ __forceinline__ float operator()(float x_resized, float, float length_resized,
                                                       float length_original, float roi_start, float roi_end) const {
    auto orig = length_resized > 1
                    ? roi_start * (length_original - 1) +
                          (x_resized * (roi_end - roi_start) * (length_original - 1)) / (length_resized - 1)
                    : 0.5 * (roi_start + roi_end) * (length_original - 1);
    return static_cast<float>(orig);
  }
};

size_t CalcResizeBufferSize(const onnxruntime::UpsampleMode upsample_mode,
                            const int64_t* output_dims, int32_t rank);

template <typename T>
void ResizeImpl(
    musaStream_t stream,
    const onnxruntime::UpsampleMode upsample_mode,
    const int rank,
    TArray<int64_t>& input_shape,
    TArray<int64_t>& output_shape,
    TArray<int64_t>& input_strides,
    TArray<fast_divmod>& output_div_pitches,
    TArray<float>& scales_vals,
    TArray<float, 10>& roi,
    const T* input_data,
    T* output_data,
    const size_t N,
    bool extrapolation_enabled,
    const T extrapolation_value,
    float cubic_coeff_a,
    bool exclude_outside,
    onnxruntime::ResizeCoordinateTransformationMode coordinate_transform_mode,
    onnxruntime::ResizeNearestMode nearest_mode,
    void* dims_mapping);

}  // namespace musa
}  // namespace onnxruntime
