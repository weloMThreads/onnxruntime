// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <musa_runtime.h>
#include <stdint.h>

namespace onnxruntime {
namespace musa {

constexpr int32_t kBitwiseExtraMaxDims = 8;

struct BitwiseExtraBinaryParams {
  int32_t rank;
  int64_t total_elements;
  int64_t output_strides[kBitwiseExtraMaxDims];
  int64_t lhs_strides[kBitwiseExtraMaxDims];
  int64_t rhs_strides[kBitwiseExtraMaxDims];
};

template <typename T>
void LaunchBitwiseOrKernel(musaStream_t stream,
                           const T* lhs,
                           const T* rhs,
                           T* output,
                           const BitwiseExtraBinaryParams& params);

template <typename T>
void LaunchBitwiseXorKernel(musaStream_t stream,
                            const T* lhs,
                            const T* rhs,
                            T* output,
                            const BitwiseExtraBinaryParams& params);

template <typename T>
void LaunchBitwiseNotKernel(musaStream_t stream, const T* input, T* output, int64_t count);

#define DECLARE_BITWISE_EXTRA_BINARY(T)                                                \
  extern template void LaunchBitwiseOrKernel<T>(musaStream_t, const T*, const T*, T*,  \
                                                const BitwiseExtraBinaryParams&);      \
  extern template void LaunchBitwiseXorKernel<T>(musaStream_t, const T*, const T*, T*, \
                                                 const BitwiseExtraBinaryParams&);

#define DECLARE_BITWISE_EXTRA_NOT(T) \
  extern template void LaunchBitwiseNotKernel<T>(musaStream_t, const T*, T*, int64_t);

DECLARE_BITWISE_EXTRA_BINARY(int8_t)
DECLARE_BITWISE_EXTRA_BINARY(uint8_t)
DECLARE_BITWISE_EXTRA_BINARY(int16_t)
DECLARE_BITWISE_EXTRA_BINARY(uint16_t)
DECLARE_BITWISE_EXTRA_BINARY(int32_t)
DECLARE_BITWISE_EXTRA_BINARY(uint32_t)
DECLARE_BITWISE_EXTRA_BINARY(int64_t)
DECLARE_BITWISE_EXTRA_BINARY(uint64_t)

DECLARE_BITWISE_EXTRA_NOT(int8_t)
DECLARE_BITWISE_EXTRA_NOT(uint8_t)
DECLARE_BITWISE_EXTRA_NOT(int16_t)
DECLARE_BITWISE_EXTRA_NOT(uint16_t)
DECLARE_BITWISE_EXTRA_NOT(int32_t)
DECLARE_BITWISE_EXTRA_NOT(uint32_t)
DECLARE_BITWISE_EXTRA_NOT(int64_t)
DECLARE_BITWISE_EXTRA_NOT(uint64_t)

#undef DECLARE_BITWISE_EXTRA_BINARY
#undef DECLARE_BITWISE_EXTRA_NOT

}  // namespace musa
}  // namespace onnxruntime
