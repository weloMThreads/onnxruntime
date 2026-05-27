// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/musa/math/logical_ops_impl.h"
#include "core/providers/musa/musa_kernel.h"

namespace onnxruntime {
namespace musa {

template <LogicalOpType Op>
class LogicalBinary final : public MusaKernel {
 public:
  explicit LogicalBinary(const OpKernelInfo& info) : MusaKernel(info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;
};

using LogicalAnd = LogicalBinary<LogicalOpType::And>;
using LogicalOr = LogicalBinary<LogicalOpType::Or>;
using LogicalXor = LogicalBinary<LogicalOpType::Xor>;

}  // namespace musa
}  // namespace onnxruntime
