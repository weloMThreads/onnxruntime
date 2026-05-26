// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/musa/musa_kernel.h"
#include "core/providers/musa/math/binary_elementwise_ops.h"

namespace onnxruntime {
namespace musa {

template <typename T>
class Mod final : public BinaryElementwise {
public:
    Mod(const OpKernelInfo& info) : BinaryElementwise(info) {
        int64_t fmod = info.GetAttrOrDefault<int64_t>("fmod", 0LL);
        fmod_ = fmod != 0;
    }

    Status ComputeInternal(OpKernelContext* context) const override;

private:
    bool fmod_{false};
};

}  // namespace musa
}  // namespace onnxruntime