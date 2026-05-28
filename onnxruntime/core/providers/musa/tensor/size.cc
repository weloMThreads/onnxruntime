// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/musa_fwd.h"

namespace onnxruntime {
namespace musa {

class MusaSize final : public OpKernel {
 public:
  explicit MusaSize(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* ctx) const override {
    const auto* input = ctx->Input<Tensor>(0);
    ORT_RETURN_IF_NOT(input != nullptr, "Size: input tensor is null");

    Tensor* output = ctx->Output(0, TensorShape{});
    ORT_RETURN_IF_NOT(output != nullptr, "Size: output tensor is null");

    *output->MutableData<int64_t>() = input->Shape().Size();
    return Status::OK();
  }
};

#define REGISTER_MUSA_SIZE_VERSIONED_KERNEL(startver, endver)                 \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                          \
      Size, kOnnxDomain, startver, endver, kMusaExecutionProvider,            \
      (*KernelDefBuilder::Create())                                           \
          .OutputMemoryType(OrtMemTypeCPUInput, 0)                            \
          .TypeConstraint("T", DataTypeImpl::AllTensorTypes())                \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),      \
      MusaSize);

#define REGISTER_MUSA_SIZE_KERNEL(ver)                                        \
  ONNX_OPERATOR_KERNEL_EX(                                                    \
      Size, kOnnxDomain, ver, kMusaExecutionProvider,                         \
      (*KernelDefBuilder::Create())                                           \
          .OutputMemoryType(OrtMemTypeCPUInput, 0)                            \
          .TypeConstraint("T", DataTypeImpl::AllTensorTypes())                \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),      \
      MusaSize);

REGISTER_MUSA_SIZE_VERSIONED_KERNEL(1, 12)
REGISTER_MUSA_SIZE_VERSIONED_KERNEL(13, 20)
REGISTER_MUSA_SIZE_VERSIONED_KERNEL(21, 22)
REGISTER_MUSA_SIZE_VERSIONED_KERNEL(23, 24)
REGISTER_MUSA_SIZE_KERNEL(25)

#undef REGISTER_MUSA_SIZE_KERNEL
#undef REGISTER_MUSA_SIZE_VERSIONED_KERNEL

}  // namespace musa
}  // namespace onnxruntime
