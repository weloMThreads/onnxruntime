// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {
namespace contrib {
namespace musa {

template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

Status RegisterMusaContribKernels(KernelRegistry& kernel_registry);

}  // namespace musa
}  // namespace contrib
}  // namespace onnxruntime
