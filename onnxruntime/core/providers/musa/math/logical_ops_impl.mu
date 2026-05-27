// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/math/logical_ops_impl.h"

#include <musa_runtime.h>
#include <stdint.h>

namespace onnxruntime {
namespace musa {
namespace {

constexpr int kLogicalThreadsPerBlock = 256;
constexpr int kLogicalMaxBlocks = 4096;

__device__ __forceinline__ bool ApplyLogicalOp(bool lhs, bool rhs, LogicalOpType op_type) {
  switch (op_type) {
    case LogicalOpType::And:
      return lhs && rhs;
    case LogicalOpType::Or:
      return lhs || rhs;
    case LogicalOpType::Xor:
      return lhs != rhs;
  }
  return false;
}

__global__ void LogicalBinaryKernel(const bool* lhs,
                                    const bool* rhs,
                                    bool* output,
                                    LogicalBinaryParams params,
                                    LogicalOpType op_type) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t index = thread_id; index < params.total_elements; index += total_threads) {
    if (params.rank == 0) {
      output[index] = ApplyLogicalOp(lhs[0], rhs[0], op_type);
      continue;
    }

    int64_t lhs_index = 0;
    int64_t rhs_index = 0;
    int64_t remaining = index;
    for (int32_t dim = 0; dim < params.rank; ++dim) {
      const int64_t coord = remaining / params.output_strides[dim];
      remaining -= coord * params.output_strides[dim];
      lhs_index += coord * params.lhs_strides[dim];
      rhs_index += coord * params.rhs_strides[dim];
    }

    output[index] = ApplyLogicalOp(lhs[lhs_index], rhs[rhs_index], op_type);
  }
}

}  // namespace

void LaunchLogicalBinaryKernel(musaStream_t stream,
                               const bool* lhs,
                               const bool* rhs,
                               bool* output,
                               const LogicalBinaryParams& params,
                               LogicalOpType op_type) {
  if (params.total_elements == 0) {
    return;
  }

  int64_t blocks = (params.total_elements + kLogicalThreadsPerBlock - 1) / kLogicalThreadsPerBlock;
  if (blocks > kLogicalMaxBlocks) {
    blocks = kLogicalMaxBlocks;
  }

  LogicalBinaryKernel<<<static_cast<int>(blocks), kLogicalThreadsPerBlock, 0, stream>>>(
      lhs, rhs, output, params, op_type);
}

}  // namespace musa
}  // namespace onnxruntime
