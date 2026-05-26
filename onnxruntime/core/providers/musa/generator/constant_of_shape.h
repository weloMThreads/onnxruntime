// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/musa_kernel.h"
#include "core/providers/cpu/generator/constant_of_shape_base.h"
#include "core/providers/musa/musa_fwd.h"

namespace onnxruntime {
namespace musa {

class ConstantOfShape final : public ConstantOfShapeBase<>, public MusaKernel {
 public:
  explicit ConstantOfShape(const OpKernelInfo& info) : ConstantOfShapeBase(info), MusaKernel(info) {}

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ConstantOfShape);

  Status ComputeInternal(OpKernelContext* ctx) const override;
};

}  // namespace musa
}  // namespace onnxruntime
