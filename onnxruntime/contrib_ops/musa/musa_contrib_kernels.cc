// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"

using namespace onnxruntime::common;

#define MUSA_MS_OP_TYPED_CLASS_NAME(ver, type, name) \
  ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSDomain, ver, type, name)

namespace onnxruntime {
namespace contrib {
namespace musa {

// Forward declarations
class MUSA_MS_OP_TYPED_CLASS_NAME(1, float, QuickGelu);
class MUSA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, QuickGelu);
class MUSA_MS_OP_TYPED_CLASS_NAME(1, float, FusedGemm);
class MUSA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, FusedGemm);

Status RegisterMusaContribKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<MUSA_MS_OP_TYPED_CLASS_NAME(1, float, QuickGelu)>,
      BuildKernelCreateInfo<MUSA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, QuickGelu)>,
      BuildKernelCreateInfo<MUSA_MS_OP_TYPED_CLASS_NAME(1, float, FusedGemm)>,
      BuildKernelCreateInfo<MUSA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, FusedGemm)>,
  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }

  return Status::OK();
}

}  // namespace musa
}  // namespace contrib
}  // namespace onnxruntime
