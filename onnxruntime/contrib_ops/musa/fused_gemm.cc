// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/math/gemm.h"

namespace onnxruntime {
namespace contrib {
namespace musa {

#define REGISTER_MUSA_FUSED_GEMM_TYPED(T)                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      FusedGemm, kMSDomain, 1, T, kMusaExecutionProvider,         \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      onnxruntime::musa::Gemm<T>);

REGISTER_MUSA_FUSED_GEMM_TYPED(float)
REGISTER_MUSA_FUSED_GEMM_TYPED(MLFloat16)

}  // namespace musa
}  // namespace contrib
}  // namespace onnxruntime
