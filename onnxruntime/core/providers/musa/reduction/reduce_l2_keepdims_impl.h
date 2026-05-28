// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>

#include <musa_runtime.h>

namespace onnxruntime {
namespace musa {

constexpr int32_t kReduceL2KeepDimsMaxRank = 8;

struct ReduceL2KeepDimsParams {
  int32_t rank;
  int32_t num_axes;
  int64_t output_size;
  int64_t reduce_size;
  int64_t input_dims[kReduceL2KeepDimsMaxRank];
  int64_t input_strides[kReduceL2KeepDimsMaxRank];
  int64_t output_strides[kReduceL2KeepDimsMaxRank];
  int32_t axes[kReduceL2KeepDimsMaxRank];
  int32_t reduced_axes[kReduceL2KeepDimsMaxRank];
};

musaError_t LaunchReduceL2KeepDimsFloat(musaStream_t stream,
                                        const float* input_data,
                                        float* output_data,
                                        const ReduceL2KeepDimsParams& params);

musaError_t LaunchReduceL2KeepDimsHalf(musaStream_t stream,
                                       const void* input_data,
                                       void* output_data,
                                       const ReduceL2KeepDimsParams& params);

musaError_t LaunchReduceSumSquareFloat(musaStream_t stream,
                                       const float* input_data,
                                       float* output_data,
                                       const ReduceL2KeepDimsParams& params);

musaError_t LaunchReduceSumSquareHalf(musaStream_t stream,
                                      const void* input_data,
                                      void* output_data,
                                      const ReduceL2KeepDimsParams& params);

}  // namespace musa
}  // namespace onnxruntime
