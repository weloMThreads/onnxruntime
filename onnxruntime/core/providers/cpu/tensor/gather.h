// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"
#include "gatherbase.h"

namespace onnxruntime {

class Gather : public OpKernel, public GatherBase {
 public:
  Gather(const OpKernelInfo& info) : OpKernel(info), GatherBase(info) {}

  Status Compute(OpKernelContext* context) const override;
};

class GatherV2 : public OpKernel {
 public:
  explicit GatherV2(const OpKernelInfo& info) : OpKernel(info) {
    info.GetAttrOrDefault<int64_t>("batch_dims", &batch_dims_, 0);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t batch_dims_ = 0;
};
}  // namespace onnxruntime
