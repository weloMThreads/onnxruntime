// Lightweight header for resize_impl.mu — MUSA kernel-side only.
// Cannot include ORT headers (mcc doesn't support C++17 features).
// Definitions here must be ABI-compatible with resize_impl.h used by host code.

#pragma once

#include <musa_runtime.h>
#include <cstring>

#include "core/providers/musa/shared_inc/fast_divmod.h"

namespace onnxruntime {

// Mirror enums from core/providers/cpu/tensor/upsamplebase.h
enum UpsampleMode {
  NN = 0,
  LINEAR = 1,
  CUBIC = 2,
};

enum ResizeCoordinateTransformationMode {
  HALF_PIXEL = 0,
  ASYMMETRIC = 1,
  PYTORCH_HALF_PIXEL = 2,
  TF_HALF_PIXEL_FOR_NN = 3,
  ALIGN_CORNERS = 4,
  TF_CROP_AND_RESIZE = 5,
  HALF_PIXEL_SYMMETRIC = 6,
};

enum ResizeNearestMode {
  SIMPLE = 0,
  ROUND_PREFER_FLOOR = 1,
  ROUND_PREFER_CEIL = 2,
  FLOOR = 3,
  CEIL = 4,
};

template <typename T>
struct AccumulateType { using type = T; };
template <>
struct AccumulateType<half> { using type = float; };


namespace musa {

// GPU-only TArray: lightweight fixed-capacity array for kernel parameters.
// Must be layout-compatible with the full TArray in resize_impl.h.
template <typename T, int32_t capacity = 8>
struct TArray {
  TArray() = default;

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

// Coordinate transform structs (identical to CudaEP)
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

}  // namespace musa
}  // namespace onnxruntime
