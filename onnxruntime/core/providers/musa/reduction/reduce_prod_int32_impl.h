// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <musa_runtime.h>

namespace onnxruntime {
namespace musa {

constexpr int32_t kReduceProdInt32MaxRank = 8;

struct ReduceProdInt32Params {
  int32_t rank;
  int64_t input_size;
  int64_t output_size;
  int64_t input_strides[kReduceProdInt32MaxRank];
  int64_t output_strides[kReduceProdInt32MaxRank];
};

musaError_t LaunchFillInt32Kernel(musaStream_t stream, int32_t* output, int64_t output_size, int32_t value);

musaError_t LaunchFillFloatKernel(musaStream_t stream, float* output, int64_t output_size, float value);

musaError_t LaunchReduceProdInt32Kernel(musaStream_t stream,
                                        const int32_t* input,
                                        int32_t* output,
                                        const ReduceProdInt32Params& params);

musaError_t LaunchReduceProdFloatKernel(musaStream_t stream,
                                        const float* input,
                                        float* output,
                                        const ReduceProdInt32Params& params);

}  // namespace musa
}  // namespace onnxruntime
