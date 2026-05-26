// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/musa_utils.h"
#include "core/providers/musa/musa_execution_provider.h"

namespace onnxruntime {
namespace musa {

MusaPreparation::MusaPreparation(const MusaExecutionProvider* ep)
    : input_a_ptr(nullptr),
      input_b_ptr(nullptr),
      input_c_ptr(nullptr),
      bias_ptr(nullptr),
      output_ptr(nullptr),
      output_size(0),
      processed_indices_ptr(nullptr),
      handle_ptr_(nullptr) {
  ORT_ENFORCE(ep != nullptr, "MusaExecutionProvider cannot be null");

  // Get PerThreadContext from EP (thread-safe, creates or reuses context)
  auto& context = ep->GetPerThreadContext();

  // Get Handle reference from context (NOT owned, do not delete)
  handle_ptr_ = &context.MudnnHandle();

  // Note: 'handle' unique_ptr remains nullptr in EP mode (not used)
}

}  // namespace musa
}  // namespace onnxruntime
