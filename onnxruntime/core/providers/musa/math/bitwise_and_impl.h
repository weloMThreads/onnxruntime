// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <musa_runtime.h>
#include <stdint.h>

namespace onnxruntime {
namespace musa {

constexpr int32_t kBitwiseAndMaxDims = 8;

struct BitwiseAndParams {
  int32_t rank;
  int64_t total_elements;
  int64_t output_strides[kBitwiseAndMaxDims];
  int64_t lhs_strides[kBitwiseAndMaxDims];
  int64_t rhs_strides[kBitwiseAndMaxDims];
};

template <typename T>
void LaunchBitwiseAndKernel(musaStream_t stream,
                            const T* lhs,
                            const T* rhs,
                            T* output,
                            const BitwiseAndParams& params);

extern template void LaunchBitwiseAndKernel<int8_t>(musaStream_t, const int8_t*, const int8_t*, int8_t*, const BitwiseAndParams&);
extern template void LaunchBitwiseAndKernel<uint8_t>(musaStream_t, const uint8_t*, const uint8_t*, uint8_t*, const BitwiseAndParams&);
extern template void LaunchBitwiseAndKernel<int16_t>(musaStream_t, const int16_t*, const int16_t*, int16_t*, const BitwiseAndParams&);
extern template void LaunchBitwiseAndKernel<uint16_t>(musaStream_t, const uint16_t*, const uint16_t*, uint16_t*, const BitwiseAndParams&);
extern template void LaunchBitwiseAndKernel<int32_t>(musaStream_t, const int32_t*, const int32_t*, int32_t*, const BitwiseAndParams&);
extern template void LaunchBitwiseAndKernel<uint32_t>(musaStream_t, const uint32_t*, const uint32_t*, uint32_t*, const BitwiseAndParams&);
extern template void LaunchBitwiseAndKernel<int64_t>(musaStream_t, const int64_t*, const int64_t*, int64_t*, const BitwiseAndParams&);
extern template void LaunchBitwiseAndKernel<uint64_t>(musaStream_t, const uint64_t*, const uint64_t*, uint64_t*, const BitwiseAndParams&);

}  // namespace musa
}  // namespace onnxruntime
