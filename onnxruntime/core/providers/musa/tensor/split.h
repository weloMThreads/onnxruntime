// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/musa_kernel.h"
#include "core/providers/cpu/tensor/split.h"

namespace onnxruntime {
namespace musa {

class SplitKernel : public MusaKernel, public SplitBase {
 public:
  SplitKernel(const OpKernelInfo& info, uint32_t opset) : MusaKernel(info), SplitBase(info, opset) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

// versions 2, 11 and 13
class Split_2_13 final : public SplitKernel {
 public:
  Split_2_13(const OpKernelInfo& info) : SplitKernel(info, /* opset */ 1) {}
};

class Split_18 final : public SplitKernel {
 public:
  Split_18(const OpKernelInfo& info) : SplitKernel(info, 18) {}
};

}  // namespace musa
}  // namespace onnxruntime
