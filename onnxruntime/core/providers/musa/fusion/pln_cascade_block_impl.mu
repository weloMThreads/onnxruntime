// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/fusion/pln_cascade_block_impl.h"

#include <musa_runtime.h>

namespace onnxruntime {
namespace musa {
namespace {

template <int kStaticSteps, bool kUseDynamicSteps>
__global__ void PlnCascadeBlockKernel(const float* norm_out,
                                      PlnCascadeBlockFloatPtrs ptrs,
                                      PlnCascadeBlockMeta meta,
                                      float* output,
                                      PlnCascadeBlockShape shape,
                                      int total_elements) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) {
    return;
  }

  const int active_steps = kUseDynamicSteps ? meta.num_steps : kStaticSteps;
  if (active_steps <= 0) {
    output[idx] = norm_out[idx];
    return;
  }

  int candidate_mask_offsets[kStaticSteps];
  int passthrough_mask_offsets[kStaticSteps];
#pragma unroll
  for (int step = 0; step < kStaticSteps; ++step) {
    candidate_mask_offsets[step] = 0;
    passthrough_mask_offsets[step] = 0;
  }

  int remaining = idx;
  int channel_idx = 0;
  for (int dim = shape.rank - 1; dim >= 0; --dim) {
    const int coord = remaining % shape.dims[dim];
    remaining /= shape.dims[dim];
    if (dim == shape.rank - 1) {
      channel_idx = coord;
    }

#pragma unroll
    for (int step = 0; step < kStaticSteps; ++step) {
      if (!kUseDynamicSteps || step < active_steps) {
        candidate_mask_offsets[step] += coord * meta.candidate_mask_strides[step].values[dim];
        passthrough_mask_offsets[step] += coord * meta.passthrough_mask_strides[step].values[dim];
      }
    }
  }

  float value = norm_out[idx];

#pragma unroll
  for (int step = 0; step < kStaticSteps; ++step) {
    if (kUseDynamicSteps && step >= active_steps) {
      break;
    }

    const float* candidate_mask = ptrs.candidate_masks[step];
    const float* passthrough_mask = ptrs.passthrough_masks[step];
    const float* add_values = ptrs.add_values[step];
    const float* bias_values = ptrs.bias_values[step];
    if (candidate_mask == nullptr || passthrough_mask == nullptr ||
        add_values == nullptr || bias_values == nullptr) {
      continue;
    }

    const int add_offset = meta.add_value_counts[step] == 1 ? 0 : channel_idx;
    const int bias_offset = meta.bias_value_counts[step] == 1 ? 0 : channel_idx;
    const float candidate = value * add_values[add_offset] + bias_values[bias_offset];
    value = candidate_mask[candidate_mask_offsets[step]] * candidate +
            passthrough_mask[passthrough_mask_offsets[step]] * value;
  }

  output[idx] = value;
}

}  // namespace

musaError_t LaunchPlnCascadeBlockFloat(musaStream_t stream,
                                       const float* norm_out,
                                       PlnCascadeBlockFloatPtrs ptrs,
                                       PlnCascadeBlockMeta meta,
                                       float* output,
                                       PlnCascadeBlockShape shape,
                                       int total_elements) {
  if (total_elements <= 0) {
    return musaSuccess;
  }

  constexpr int kThreads = 256;
  const int blocks = (total_elements + kThreads - 1) / kThreads;

#define LAUNCH_PLN_BLOCK_CASE(STEPS)                                                \
  case STEPS:                                                                       \
    PlnCascadeBlockKernel<STEPS, false><<<blocks, kThreads, 0, stream>>>(           \
        norm_out, ptrs, meta, output, shape, total_elements);                       \
    break

  switch (meta.num_steps) {
    LAUNCH_PLN_BLOCK_CASE(1);
    LAUNCH_PLN_BLOCK_CASE(2);
    LAUNCH_PLN_BLOCK_CASE(3);
    LAUNCH_PLN_BLOCK_CASE(4);
    LAUNCH_PLN_BLOCK_CASE(5);
    LAUNCH_PLN_BLOCK_CASE(6);
    LAUNCH_PLN_BLOCK_CASE(7);
    LAUNCH_PLN_BLOCK_CASE(8);
    LAUNCH_PLN_BLOCK_CASE(9);
    LAUNCH_PLN_BLOCK_CASE(10);
    LAUNCH_PLN_BLOCK_CASE(11);
    LAUNCH_PLN_BLOCK_CASE(12);
    LAUNCH_PLN_BLOCK_CASE(13);
    LAUNCH_PLN_BLOCK_CASE(14);
    LAUNCH_PLN_BLOCK_CASE(15);
    LAUNCH_PLN_BLOCK_CASE(16);
    default:
      PlnCascadeBlockKernel<kPlnCascadeBlockMaxSteps, true>
          <<<blocks, kThreads, 0, stream>>>(norm_out, ptrs, meta, output, shape, total_elements);
      break;
  }

#undef LAUNCH_PLN_BLOCK_CASE

  return musaGetLastError();
}

}  // namespace musa
}  // namespace onnxruntime
