// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <musa_runtime.h>
#include <stdint.h>

namespace onnxruntime {
namespace musa {

constexpr int32_t kLogicalBinaryMaxDims = 8;

enum class LogicalOpType : int32_t {
  And = 0,
  Or = 1,
  Xor = 2,
};

struct LogicalBinaryParams {
  int32_t rank;
  int64_t total_elements;
  int64_t output_strides[kLogicalBinaryMaxDims];
  int64_t lhs_strides[kLogicalBinaryMaxDims];
  int64_t rhs_strides[kLogicalBinaryMaxDims];
};

void LaunchLogicalBinaryKernel(musaStream_t stream,
                               const bool* lhs,
                               const bool* rhs,
                               bool* output,
                               const LogicalBinaryParams& params,
                               LogicalOpType op_type);

}  // namespace musa
}  // namespace onnxruntime
