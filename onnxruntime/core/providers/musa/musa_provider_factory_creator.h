// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/providers/providers.h"

struct OrtMUSAProviderOptions;

namespace onnxruntime {
struct MusaProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory>
  Create(const OrtMUSAProviderOptions *provider_options);
};
} // namespace onnxruntime
