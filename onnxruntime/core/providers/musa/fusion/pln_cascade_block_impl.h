// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <musa_runtime.h>

namespace onnxruntime {
namespace musa {

constexpr int kPlnCascadeBlockMaxDims = 8;
constexpr int kPlnCascadeBlockMaxSteps = 16;

struct PlnCascadeBlockShape {
  int rank;
  int dims[kPlnCascadeBlockMaxDims];
};

struct PlnCascadeBlockStrides {
  int values[kPlnCascadeBlockMaxDims];
};

struct PlnCascadeBlockFloatPtrs {
  const float* candidate_masks[kPlnCascadeBlockMaxSteps];
  const float* passthrough_masks[kPlnCascadeBlockMaxSteps];
  const float* add_values[kPlnCascadeBlockMaxSteps];
  const float* bias_values[kPlnCascadeBlockMaxSteps];
};

struct PlnCascadeBlockMeta {
  int num_steps;
  int output_last_dim;
  int add_value_counts[kPlnCascadeBlockMaxSteps];
  int bias_value_counts[kPlnCascadeBlockMaxSteps];
  PlnCascadeBlockStrides candidate_mask_strides[kPlnCascadeBlockMaxSteps];
  PlnCascadeBlockStrides passthrough_mask_strides[kPlnCascadeBlockMaxSteps];
};

musaError_t LaunchPlnCascadeBlockFloat(musaStream_t stream,
                                       const float* norm_out,
                                       PlnCascadeBlockFloatPtrs ptrs,
                                       PlnCascadeBlockMeta meta,
                                       float* output,
                                       PlnCascadeBlockShape shape,
                                       int total_elements);

}  // namespace musa
}  // namespace onnxruntime
