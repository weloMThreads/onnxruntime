// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <musa_fp16.h>
#include <musa_runtime.h>

#include <cstdlib>
#include <cmath>
#include <cstdint>

#include "core/providers/musa/tensor/grid_sample_impl.h"

#ifndef CUDA_LONG
#define CUDA_LONG int32_t
#endif

struct GridDim {
  enum : CUDA_LONG {
    maxThreadsPerBlock = 256,
  };
};

#define CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N) \
  CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x; \
  if (id >= N) return;

namespace onnxruntime {
namespace contrib {
namespace musa {

constexpr bool LAYOUT_NCHW = false;
constexpr bool LAYOUT_NHWC = true;

template <bool IsNHWC>
struct Channels;

template <>
struct Channels<LAYOUT_NHWC> {
  static constexpr size_t N = 0;
  static constexpr size_t H = 1;
  static constexpr size_t W = 2;
  static constexpr size_t C = 3;
};

template <>
struct Channels<LAYOUT_NCHW> {
  static constexpr size_t N = 0;
  static constexpr size_t C = 1;
  static constexpr size_t H = 2;
  static constexpr size_t W = 3;
};

template <typename T>
__device__ T GsDenormalize(T n, int64_t length, bool align_corners) {
  T x = {};
  if (align_corners) {
    x = (n + static_cast<T>(1)) / static_cast<T>(2) * (length - 1);
  } else {
    x = ((n + static_cast<T>(1)) * length - static_cast<T>(1)) / static_cast<T>(2);
  }
  return x;
}

template <typename T>
__device__ T GsReflect(T x, float x_min, float x_max) {
  float fx = static_cast<float>(x);
  float dx = {};
  float range = x_max - x_min;
  if (fx < x_min) {
    dx = x_min - fx;
    int n = static_cast<int>(dx / range);
    float r = dx - n * range;
    if (n % 2 == 0) {
      fx = x_min + r;
    } else {
      fx = x_max - r;
    }
  } else if (fx > x_max) {
    dx = fx - x_max;
    int n = static_cast<int>(dx / range);
    float r = dx - n * range;
    if (n % 2 == 0) {
      fx = x_max - r;
    } else {
      fx = x_min + r;
    }
  }
  return static_cast<T>(fx);
}

template <typename T, bool Layout>
__device__ T PixelAtGrid(const T* input_data, int64_t bIdx, int64_t cIdx, int64_t y, int64_t x,
                         int64_t padding_mode, int64_t N, int64_t C, int64_t H, int64_t W, float border[4]) {
  T pixel = 0.0f;

  auto PixelOffset = [bIdx, cIdx, C, H, W](int64_t x_loc, int64_t y_loc) -> int64_t {
    return Layout == LAYOUT_NCHW
               ? (bIdx * C * H * W + cIdx * H * W + y_loc * W + x_loc)
               : (bIdx * H * W * C + y_loc * W * C + x_loc * C + cIdx);
  };

  if (padding_mode == 0) {
    if (x >= 0 && x < W && y >= 0 && y < H) {
      pixel = input_data[PixelOffset(x, y)];
    }
  } else if (padding_mode == 1) {
    x = x < 0 ? 0 : (x > W - 1 ? W - 1 : x);
    y = y < 0 ? 0 : (y > H - 1 ? H - 1 : y);
    pixel = input_data[PixelOffset(x, y)];
  } else {
    x = (int64_t)GsReflect<T>(x, border[0], border[2]);
    y = (int64_t)GsReflect<T>(y, border[1], border[3]);
    pixel = input_data[PixelOffset(x, y)];
  }
  return pixel;
}

template <typename T>
__device__ T PixelAtGridSingleChannel(const T* input_channel_data, int64_t y, int64_t x,
                                      int64_t padding_mode, int64_t H, int64_t W, float border[4]) {
  T pixel = 0.0f;
  if (padding_mode == 0) {
    if (x >= 0 && x < W && y >= 0 && y < H) {
      pixel = input_channel_data[y * W + x];
    }
  } else if (padding_mode == 1) {
    x = x < 0 ? 0 : (x > W - 1 ? W - 1 : x);
    y = y < 0 ? 0 : (y > H - 1 ? H - 1 : y);
    pixel = input_channel_data[y * W + x];
  } else {
    x = static_cast<int64_t>(GsReflect<T>(x, border[0], border[2]));
    y = static_cast<int64_t>(GsReflect<T>(y, border[1], border[3]));
    pixel = input_channel_data[y * W + x];
  }
  return pixel;
}

template <typename T, int PaddingMode>
__device__ __forceinline__ T PixelAtGridSingleChannel_T(const T* input_channel_data, int64_t y, int64_t x,
                                                        int64_t H, int64_t W, float border[4]) {
  T pixel = 0.0f;
  if (PaddingMode == 0) {
    if (x >= 0 && x < W && y >= 0 && y < H) {
      pixel = input_channel_data[y * W + x];
    }
  } else if (PaddingMode == 1) {
    x = x < 0 ? 0 : (x > W - 1 ? W - 1 : x);
    y = y < 0 ? 0 : (y > H - 1 ? H - 1 : y);
    pixel = input_channel_data[y * W + x];
  } else {
    x = static_cast<int64_t>(GsReflect<T>(x, border[0], border[2]));
    y = static_cast<int64_t>(GsReflect<T>(y, border[1], border[3]));
    pixel = input_channel_data[y * W + x];
  }
  return pixel;
}

template <typename T>
__device__ const T* PixelBaseAtGridNhwcC3(const T* input_data, int64_t bIdx, int64_t y, int64_t x,
                                          int64_t padding_mode, int64_t H, int64_t W, float border[4]) {
  if (padding_mode == 0) {
    if (x < 0 || x >= W || y < 0 || y >= H) {
      return nullptr;
    }
  } else if (padding_mode == 1) {
    x = x < 0 ? 0 : (x > W - 1 ? W - 1 : x);
    y = y < 0 ? 0 : (y > H - 1 ? H - 1 : y);
  } else {
    x = static_cast<int64_t>(GsReflect<T>(x, border[0], border[2]));
    y = static_cast<int64_t>(GsReflect<T>(y, border[1], border[3]));
  }

  return input_data + (bIdx * H * W + y * W + x) * 3;
}

template <typename T, int PaddingMode>
__device__ __forceinline__ const T* PixelBaseAtGridNhwcC3_T(const T* input_data, int64_t bIdx, int64_t y, int64_t x,
                                                            int64_t H, int64_t W, float border[4]) {
  if (PaddingMode == 0) {
    if (x < 0 || x >= W || y < 0 || y >= H) {
      return nullptr;
    }
  } else if (PaddingMode == 1) {
    x = x < 0 ? 0 : (x > W - 1 ? W - 1 : x);
    y = y < 0 ? 0 : (y > H - 1 ? H - 1 : y);
  } else {
    x = static_cast<int64_t>(GsReflect<T>(x, border[0], border[2]));
    y = static_cast<int64_t>(GsReflect<T>(y, border[1], border[3]));
  }

  return input_data + (bIdx * H * W + y * W + x) * 3;
}

__device__ void GsGetCubicCoeffs(float x, float coeffs[4]) {
  float cubic_alpha = -0.75f;
  x = abs(x);
  coeffs[0] = (((cubic_alpha * (x + 1) - 5 * cubic_alpha) * (x + 1) + 8 * cubic_alpha) * (x + 1) - 4 * cubic_alpha);
  coeffs[1] = (((cubic_alpha + 2) * x - (cubic_alpha + 3)) * x * x + 1);
  coeffs[2] = (((cubic_alpha + 2) * (1 - x) - (cubic_alpha + 3)) * (1 - x) * (1 - x) + 1);
  coeffs[3] = (((cubic_alpha * (2 - x) - 5 * cubic_alpha) * (2 - x) + 8 * cubic_alpha) * (2 - x) - 4 * cubic_alpha);
}

template <typename T>
__device__ T GsBicubicInterpolate(T p[4][4], float x, float y) {
  float v[4] = {};
  float coeffs[4] = {};
  GsGetCubicCoeffs(x, coeffs);
  for (int64_t i = 0; i < 4; i++) {
    v[i] = coeffs[0] * static_cast<float>(p[i][0]) + coeffs[1] * static_cast<float>(p[i][1]) +
           coeffs[2] * static_cast<float>(p[i][2]) + coeffs[3] * static_cast<float>(p[i][3]);
  }
  GsGetCubicCoeffs(y, coeffs);
  T pixel = static_cast<T>(coeffs[0] * v[0] + coeffs[1] * v[1] + coeffs[2] * v[2] + coeffs[3] * v[3]);
  return pixel;
}

template <typename T, bool Layout>
__global__ void _GridSampleKernel(
    const T* input_data,
    const T* grid_data,
    const int64_t mode,
    const int64_t padding_mode,
    const int64_t align_corners,
    const int64_t N,
    const int64_t C,
    const int64_t H_in,
    const int64_t W_in,
    const int64_t H_out,
    const int64_t W_out,
    T* output_data) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(idx, N * C * H_out * W_out);
  int BIdx, yIdx, xIdx, cIdx;
  if (Layout == LAYOUT_NCHW) {
    BIdx = idx / (C * H_out * W_out);
    int tmpBCnt = BIdx * (C * H_out * W_out);

    cIdx = (idx - tmpBCnt) / (H_out * W_out);
    int tmpCCnt = tmpBCnt + cIdx * (H_out * W_out);

    yIdx = (idx - tmpCCnt) / W_out;
    int tmpHCnt = tmpCCnt + yIdx * W_out;

    xIdx = (idx - tmpHCnt);
  } else {
    BIdx = idx / (H_out * W_out * C);
    int tmpBCnt = BIdx * (H_out * W_out * C);

    yIdx = (idx - tmpBCnt) / (W_out * C);
    int tmpHCnt = tmpBCnt + yIdx * (W_out * C);

    xIdx = (idx - tmpHCnt) / C;
    int tmpWCnt = tmpHCnt + xIdx * C;

    cIdx = (idx - tmpWCnt);
  }

  int grid_idx = BIdx * H_out * W_out + yIdx * W_out + xIdx;
  T grid_X = grid_data[grid_idx * 2 + 0];
  T grid_Y = grid_data[grid_idx * 2 + 1];
  int outIdx = idx;

  float grid_x_imgSpace = static_cast<float>(grid_X);
  float grid_y_imgSpace = static_cast<float>(grid_Y);
  if (align_corners == 1) {
    grid_x_imgSpace = (grid_x_imgSpace + 1.0f) / 2.0f * (W_in - 1);
    grid_y_imgSpace = (grid_y_imgSpace + 1.0f) / 2.0f * (H_in - 1);
  } else {
    grid_x_imgSpace = ((grid_x_imgSpace + 1.0f) * W_in - 1.0f) / 2.0f;
    grid_y_imgSpace = ((grid_y_imgSpace + 1.0f) * H_in - 1.0f) / 2.0f;
  }
  if (mode == 1) {
    grid_x_imgSpace = nearbyintf(grid_x_imgSpace);
    grid_y_imgSpace = nearbyintf(grid_y_imgSpace);
  }
  float x_min = -0.5f;
  float x_max = W_in - 0.5f;
  float y_min = -0.5f;
  float y_max = H_in - 0.5f;

  if (align_corners) {
    x_min = 0.0f;
    x_max = W_in - 1.0;
    y_min = 0.0f;
    y_max = H_in - 1.0f;
  }
  float border[] = {x_min, y_min, x_max, y_max};
  if (grid_x_imgSpace < x_min || grid_x_imgSpace > x_max ||
      grid_y_imgSpace < y_min || grid_y_imgSpace > y_max) {
    if (padding_mode == 1) {
    } else if (padding_mode == 2) {
      grid_x_imgSpace = static_cast<float>(GsReflect(grid_x_imgSpace, x_min, x_max));
      grid_y_imgSpace = static_cast<float>(GsReflect(grid_y_imgSpace, y_min, y_max));
    }
  }

  if (mode == 0) {
    float x_img = grid_x_imgSpace;
    float y_img = grid_y_imgSpace;
    int x1 = static_cast<int>(floorf(x_img));
    int y1 = static_cast<int>(floorf(y_img));
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    float w_r = x_img - static_cast<float>(x1);
    float w_l = 1.0f - w_r;
    float w_b = y_img - static_cast<float>(y1);
    float w_t = 1.0f - w_b;

    T lt_v = PixelAtGrid<T, Layout>(input_data, BIdx, cIdx, y1, x1, padding_mode, N, C, H_in, W_in, border);
    T rt_v = PixelAtGrid<T, Layout>(input_data, BIdx, cIdx, y1, x2, padding_mode, N, C, H_in, W_in, border);
    T lb_v = PixelAtGrid<T, Layout>(input_data, BIdx, cIdx, y2, x1, padding_mode, N, C, H_in, W_in, border);
    T rb_v = PixelAtGrid<T, Layout>(input_data, BIdx, cIdx, y2, x2, padding_mode, N, C, H_in, W_in, border);
    float interpoV = w_t * w_l * static_cast<float>(lt_v) + w_t * w_r * static_cast<float>(rt_v) +
                     w_b * w_l * static_cast<float>(lb_v) + w_b * w_r * static_cast<float>(rb_v);
    output_data[outIdx] = static_cast<T>(interpoV);
    return;
  }
  if (mode == 1) {
    int x_n = static_cast<int>(grid_x_imgSpace);
    int y_n = static_cast<int>(grid_y_imgSpace);
    output_data[outIdx] =
        PixelAtGrid<T, Layout>(input_data, BIdx, cIdx, y_n, x_n, padding_mode, N, C, H_in, W_in, border);
    return;
  }
  if (mode == 2) {
    int64_t x0 = static_cast<int64_t>(floorf(static_cast<float>(grid_x_imgSpace))) - 1;
    int64_t y0 = static_cast<int64_t>(floorf(static_cast<float>(grid_y_imgSpace))) - 1;
    T p[4][4] = {};
    for (int64_t h = 0; h < 4; h++) {
      for (int64_t w = 0; w < 4; w++) {
        p[h][w] = PixelAtGrid<T, Layout>(input_data, BIdx, cIdx, h + y0, w + x0, padding_mode, N, C, H_in, W_in,
                                         border);
      }
    }
    float dx = static_cast<float>(grid_x_imgSpace) - static_cast<float>(x0) - 1.0f;
    float dy = static_cast<float>(grid_y_imgSpace) - static_cast<float>(y0) - 1.0f;
    output_data[outIdx] = GsBicubicInterpolate(p, dx, dy);
  }
}

template <typename T, int PaddingMode, int AlignCorners>
__global__ void _GridSampleKernelVecNchwC3Bilinear_T(
    const T* input_data,
    const T* grid_data,
    const int64_t N,
    const int64_t H_in,
    const int64_t W_in,
    const int64_t H_out,
    const int64_t W_out,
    T* output_data) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(idx, N * H_out * W_out);

  int BIdx = idx / (H_out * W_out);
  int tmpBCnt = BIdx * (H_out * W_out);
  int yIdx = (idx - tmpBCnt) / W_out;
  int xIdx = idx - tmpBCnt - yIdx * W_out;

  T grid_X = grid_data[idx * 2 + 0];
  T grid_Y = grid_data[idx * 2 + 1];

  float grid_x_imgSpace = static_cast<float>(grid_X);
  float grid_y_imgSpace = static_cast<float>(grid_Y);
  if (AlignCorners == 1) {
    grid_x_imgSpace = (grid_x_imgSpace + 1.0f) / 2.0f * (W_in - 1);
    grid_y_imgSpace = (grid_y_imgSpace + 1.0f) / 2.0f * (H_in - 1);
  } else {
    grid_x_imgSpace = ((grid_x_imgSpace + 1.0f) * W_in - 1.0f) / 2.0f;
    grid_y_imgSpace = ((grid_y_imgSpace + 1.0f) * H_in - 1.0f) / 2.0f;
  }

  float x_min = -0.5f;
  float x_max = W_in - 0.5f;
  float y_min = -0.5f;
  float y_max = H_in - 0.5f;
  if (AlignCorners) {
    x_min = 0.0f;
    x_max = W_in - 1.0f;
    y_min = 0.0f;
    y_max = H_in - 1.0f;
  }
  float border[] = {x_min, y_min, x_max, y_max};
  if (grid_x_imgSpace < x_min || grid_x_imgSpace > x_max ||
      grid_y_imgSpace < y_min || grid_y_imgSpace > y_max) {
    if (PaddingMode == 2) {
      grid_x_imgSpace = static_cast<float>(GsReflect(grid_x_imgSpace, x_min, x_max));
      grid_y_imgSpace = static_cast<float>(GsReflect(grid_y_imgSpace, y_min, y_max));
    }
  }

  int x1 = static_cast<int>(floorf(grid_x_imgSpace));
  int y1 = static_cast<int>(floorf(grid_y_imgSpace));
  int x2 = x1 + 1;
  int y2 = y1 + 1;

  float w_r = grid_x_imgSpace - static_cast<float>(x1);
  float w_l = 1.0f - w_r;
  float w_b = grid_y_imgSpace - static_cast<float>(y1);
  float w_t = 1.0f - w_b;

  const int64_t input_plane_stride = H_in * W_in;
  const int64_t output_plane_stride = H_out * W_out;
  const int64_t input_batch_base = static_cast<int64_t>(BIdx) * 3 * input_plane_stride;
  const int64_t output_batch_base = static_cast<int64_t>(BIdx) * 3 * output_plane_stride;
  const int64_t output_spatial_offset = static_cast<int64_t>(yIdx) * W_out + xIdx;

  for (int c = 0; c < 3; ++c) {
    const T* input_channel_data = input_data + input_batch_base + static_cast<int64_t>(c) * input_plane_stride;
    T lt_v = PixelAtGridSingleChannel_T<T, PaddingMode>(input_channel_data, y1, x1, H_in, W_in, border);
    T rt_v = PixelAtGridSingleChannel_T<T, PaddingMode>(input_channel_data, y1, x2, H_in, W_in, border);
    T lb_v = PixelAtGridSingleChannel_T<T, PaddingMode>(input_channel_data, y2, x1, H_in, W_in, border);
    T rb_v = PixelAtGridSingleChannel_T<T, PaddingMode>(input_channel_data, y2, x2, H_in, W_in, border);
    float interpoV = w_t * w_l * static_cast<float>(lt_v) + w_t * w_r * static_cast<float>(rt_v) +
                     w_b * w_l * static_cast<float>(lb_v) + w_b * w_r * static_cast<float>(rb_v);
    output_data[output_batch_base + static_cast<int64_t>(c) * output_plane_stride + output_spatial_offset] =
        static_cast<T>(interpoV);
  }
}

template <typename T, int PaddingMode, int AlignCorners>
__global__ void _GridSampleKernelVecNhwcC3Bilinear_T(
    const T* input_data,
    const T* grid_data,
    const int64_t N,
    const int64_t H_in,
    const int64_t W_in,
    const int64_t H_out,
    const int64_t W_out,
    T* output_data) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(idx, N * H_out * W_out);

  int BIdx = idx / (H_out * W_out);
  int tmpBCnt = BIdx * (H_out * W_out);
  int yIdx = (idx - tmpBCnt) / W_out;
  int xIdx = idx - tmpBCnt - yIdx * W_out;

  T grid_X = grid_data[idx * 2 + 0];
  T grid_Y = grid_data[idx * 2 + 1];

  float grid_x_imgSpace = static_cast<float>(grid_X);
  float grid_y_imgSpace = static_cast<float>(grid_Y);
  if (AlignCorners == 1) {
    grid_x_imgSpace = (grid_x_imgSpace + 1.0f) / 2.0f * (W_in - 1);
    grid_y_imgSpace = (grid_y_imgSpace + 1.0f) / 2.0f * (H_in - 1);
  } else {
    grid_x_imgSpace = ((grid_x_imgSpace + 1.0f) * W_in - 1.0f) / 2.0f;
    grid_y_imgSpace = ((grid_y_imgSpace + 1.0f) * H_in - 1.0f) / 2.0f;
  }

  float x_min = -0.5f;
  float x_max = W_in - 0.5f;
  float y_min = -0.5f;
  float y_max = H_in - 0.5f;
  if (AlignCorners) {
    x_min = 0.0f;
    x_max = W_in - 1.0f;
    y_min = 0.0f;
    y_max = H_in - 1.0f;
  }
  float border[] = {x_min, y_min, x_max, y_max};
  if (grid_x_imgSpace < x_min || grid_x_imgSpace > x_max ||
      grid_y_imgSpace < y_min || grid_y_imgSpace > y_max) {
    if (PaddingMode == 2) {
      grid_x_imgSpace = static_cast<float>(GsReflect(grid_x_imgSpace, x_min, x_max));
      grid_y_imgSpace = static_cast<float>(GsReflect(grid_y_imgSpace, y_min, y_max));
    }
  }

  int x1 = static_cast<int>(floorf(grid_x_imgSpace));
  int y1 = static_cast<int>(floorf(grid_y_imgSpace));
  int x2 = x1 + 1;
  int y2 = y1 + 1;

  float w_r = grid_x_imgSpace - static_cast<float>(x1);
  float w_l = 1.0f - w_r;
  float w_b = grid_y_imgSpace - static_cast<float>(y1);
  float w_t = 1.0f - w_b;

  const T* lt_ptr = PixelBaseAtGridNhwcC3_T<T, PaddingMode>(input_data, BIdx, y1, x1, H_in, W_in, border);
  const T* rt_ptr = PixelBaseAtGridNhwcC3_T<T, PaddingMode>(input_data, BIdx, y1, x2, H_in, W_in, border);
  const T* lb_ptr = PixelBaseAtGridNhwcC3_T<T, PaddingMode>(input_data, BIdx, y2, x1, H_in, W_in, border);
  const T* rb_ptr = PixelBaseAtGridNhwcC3_T<T, PaddingMode>(input_data, BIdx, y2, x2, H_in, W_in, border);

  T* output_ptr = output_data + (BIdx * H_out * W_out + yIdx * W_out + xIdx) * 3;
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    T lt_v = lt_ptr == nullptr ? static_cast<T>(0.0f) : lt_ptr[c];
    T rt_v = rt_ptr == nullptr ? static_cast<T>(0.0f) : rt_ptr[c];
    T lb_v = lb_ptr == nullptr ? static_cast<T>(0.0f) : lb_ptr[c];
    T rb_v = rb_ptr == nullptr ? static_cast<T>(0.0f) : rb_ptr[c];
    float interpoV = w_t * w_l * static_cast<float>(lt_v) + w_t * w_r * static_cast<float>(rt_v) +
                     w_b * w_l * static_cast<float>(lb_v) + w_b * w_r * static_cast<float>(rb_v);
    output_ptr[c] = static_cast<T>(interpoV);
  }
}

template <typename T, int PaddingMode, int AlignCorners>
inline void LaunchGridSampleVecNchwC3Bilinear(
    musaStream_t stream,
    const T* input_data,
    const T* grid_data,
    const int64_t N,
    const int64_t H_in,
    const int64_t W_in,
    const int64_t H_out,
    const int64_t W_out,
    T* output_data) {
  int blocksPerGrid = static_cast<int>(
      ceil(static_cast<float>(N * H_out * W_out) / GridDim::maxThreadsPerBlock));
  _GridSampleKernelVecNchwC3Bilinear_T<T, PaddingMode, AlignCorners><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input_data, grid_data, N, H_in, W_in, H_out, W_out, output_data);
}

template <typename T, int PaddingMode, int AlignCorners>
inline void LaunchGridSampleVecNhwcC3Bilinear(
    musaStream_t stream,
    const T* input_data,
    const T* grid_data,
    const int64_t N,
    const int64_t H_in,
    const int64_t W_in,
    const int64_t H_out,
    const int64_t W_out,
    T* output_data) {
  int blocksPerGrid = static_cast<int>(
      ceil(static_cast<float>(N * H_out * W_out) / GridDim::maxThreadsPerBlock));
  _GridSampleKernelVecNhwcC3Bilinear_T<T, PaddingMode, AlignCorners><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input_data, grid_data, N, H_in, W_in, H_out, W_out, output_data);
}

template <typename T>
inline bool TryLaunchGridSampleVecNchwC3Bilinear(
    musaStream_t stream,
    const T* input_data,
    const T* grid_data,
    const int64_t padding_mode,
    const int64_t align_corners,
    const int64_t N,
    const int64_t H_in,
    const int64_t W_in,
    const int64_t H_out,
    const int64_t W_out,
    T* output_data) {
  switch (padding_mode) {
    case 0:
      if (align_corners == 0) {
        LaunchGridSampleVecNchwC3Bilinear<T, 0, 0>(stream, input_data, grid_data, N, H_in, W_in, H_out, W_out,
                                                   output_data);
        return true;
      }
      if (align_corners == 1) {
        LaunchGridSampleVecNchwC3Bilinear<T, 0, 1>(stream, input_data, grid_data, N, H_in, W_in, H_out, W_out,
                                                   output_data);
        return true;
      }
      break;
    case 1:
      if (align_corners == 0) {
        LaunchGridSampleVecNchwC3Bilinear<T, 1, 0>(stream, input_data, grid_data, N, H_in, W_in, H_out, W_out,
                                                   output_data);
        return true;
      }
      if (align_corners == 1) {
        LaunchGridSampleVecNchwC3Bilinear<T, 1, 1>(stream, input_data, grid_data, N, H_in, W_in, H_out, W_out,
                                                   output_data);
        return true;
      }
      break;
    case 2:
      if (align_corners == 0) {
        LaunchGridSampleVecNchwC3Bilinear<T, 2, 0>(stream, input_data, grid_data, N, H_in, W_in, H_out, W_out,
                                                   output_data);
        return true;
      }
      if (align_corners == 1) {
        LaunchGridSampleVecNchwC3Bilinear<T, 2, 1>(stream, input_data, grid_data, N, H_in, W_in, H_out, W_out,
                                                   output_data);
        return true;
      }
      break;
  }

  return false;
}

template <typename T>
inline bool TryLaunchGridSampleVecNhwcC3Bilinear(
    musaStream_t stream,
    const T* input_data,
    const T* grid_data,
    const int64_t padding_mode,
    const int64_t align_corners,
    const int64_t N,
    const int64_t H_in,
    const int64_t W_in,
    const int64_t H_out,
    const int64_t W_out,
    T* output_data) {
  switch (padding_mode) {
    case 0:
      if (align_corners == 0) {
        LaunchGridSampleVecNhwcC3Bilinear<T, 0, 0>(stream, input_data, grid_data, N, H_in, W_in, H_out, W_out,
                                                   output_data);
        return true;
      }
      if (align_corners == 1) {
        LaunchGridSampleVecNhwcC3Bilinear<T, 0, 1>(stream, input_data, grid_data, N, H_in, W_in, H_out, W_out,
                                                   output_data);
        return true;
      }
      break;
    case 1:
      if (align_corners == 0) {
        LaunchGridSampleVecNhwcC3Bilinear<T, 1, 0>(stream, input_data, grid_data, N, H_in, W_in, H_out, W_out,
                                                   output_data);
        return true;
      }
      if (align_corners == 1) {
        LaunchGridSampleVecNhwcC3Bilinear<T, 1, 1>(stream, input_data, grid_data, N, H_in, W_in, H_out, W_out,
                                                   output_data);
        return true;
      }
      break;
    case 2:
      if (align_corners == 0) {
        LaunchGridSampleVecNhwcC3Bilinear<T, 2, 0>(stream, input_data, grid_data, N, H_in, W_in, H_out, W_out,
                                                   output_data);
        return true;
      }
      if (align_corners == 1) {
        LaunchGridSampleVecNhwcC3Bilinear<T, 2, 1>(stream, input_data, grid_data, N, H_in, W_in, H_out, W_out,
                                                   output_data);
        return true;
      }
      break;
  }

  return false;
}

template <typename T, bool IsNHWC>
void GridSampleImpl(
    musaStream_t stream,
    const T* input_data,
    const T* grid_data,
    const int64_t mode,
    const int64_t padding_mode,
    const int64_t align_corners,
    const int64_t dims[4],
    const int64_t H_out,
    const int64_t W_out,
    T* output_data) {
  using Ch = Channels<IsNHWC>;
  const bool disable_fast_path = std::getenv("MUSA_GRIDSAMPLE_DISABLE_FASTPATH") != nullptr;

  if (!disable_fast_path && !IsNHWC && mode == 0 && dims[Ch::C] == 3) {
    if (TryLaunchGridSampleVecNchwC3Bilinear(
            stream, input_data, grid_data, padding_mode, align_corners,
            dims[Ch::N], dims[Ch::H], dims[Ch::W], H_out, W_out, output_data)) {
      return;
    }
  }

  if (!disable_fast_path && IsNHWC && mode == 0 && dims[Ch::C] == 3) {
    if (TryLaunchGridSampleVecNhwcC3Bilinear(
            stream, input_data, grid_data, padding_mode, align_corners,
            dims[Ch::N], dims[Ch::H], dims[Ch::W], H_out, W_out, output_data)) {
      return;
    }
  }

  int blocksPerGrid = static_cast<int>(
      ceil(static_cast<float>(dims[Ch::N] * dims[Ch::C] * H_out * W_out) / GridDim::maxThreadsPerBlock));
  _GridSampleKernel<T, IsNHWC><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input_data, grid_data, mode, padding_mode, align_corners,
      dims[Ch::N], dims[Ch::C], dims[Ch::H], dims[Ch::W],
      H_out, W_out, output_data);
}

#define SPECIALIZED_IMPL(T, IsNHWC)                                                                                  \
  template void GridSampleImpl<T, IsNHWC>(musaStream_t stream, const T* input_data, const T* grid_data,            \
                                          const int64_t mode, const int64_t padding_mode, const int64_t align_corners, \
                                          const int64_t[4], const int64_t H_out, const int64_t W_out, T* output_data);

SPECIALIZED_IMPL(float, false)
SPECIALIZED_IMPL(float, true)
SPECIALIZED_IMPL(half, false)
SPECIALIZED_IMPL(half, true)

// C-linkage wrapper for half type (callable from host .cc without half type)
void GridSampleImplHalf(
    musaStream_t stream,
    const void* input_data,
    const void* grid_data,
    const int64_t mode,
    const int64_t padding_mode,
    const int64_t align_corners,
    const int64_t dims_input[4],
    const int64_t H_out,
    const int64_t W_out,
    void* output_data,
    bool is_nhwc) {
  if (is_nhwc) {
    GridSampleImpl<half, true>(stream,
        static_cast<const half*>(input_data),
        static_cast<const half*>(grid_data),
        mode, padding_mode, align_corners, dims_input, H_out, W_out,
        static_cast<half*>(output_data));
  } else {
    GridSampleImpl<half, false>(stream,
        static_cast<const half*>(input_data),
        static_cast<const half*>(grid_data),
        mode, padding_mode, align_corners, dims_input, H_out, W_out,
        static_cast<half*>(output_data));
  }
}

}  // namespace musa
}  // namespace contrib
}  // namespace onnxruntime
