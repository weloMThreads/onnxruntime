// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <musa_runtime.h>
#include <mutex>
#include "core/framework/allocator.h"
#include "core/providers/musa/musa_utils.h"
#include "core/providers/musa/musa_stream_handle.h"

namespace onnxruntime {

// Forward declaration
class MusaExecutionProvider;
namespace musa {

class MusaKernel : public OpKernel {
 public:
  explicit MusaKernel(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* p_op_kernel_context) const override {
    auto s = ComputeInternal(p_op_kernel_context);

    if (s.IsOK()) {
      // TODO: Add MUSA-specific error checking if needed
      // Similar to CannKernel's aclGetRecentErrMsg() check
    }

    return s;
  }

  virtual Status
  ComputeInternal(OpKernelContext* p_op_kernel_context) const = 0;

  // Helper method to get MUSA stream
  inline musaStream_t Stream(OpKernelContext* ctx) const {
    auto* stream = ctx->GetComputeStream();
    return stream ? static_cast<musaStream_t>(stream->GetHandle()) : nullptr;
  }

  template <typename T>
  inline IAllocatorUniquePtr<T> GetScratchBuffer(size_t count_or_bytes,
                                                 onnxruntime::Stream* stream) const {
    if (count_or_bytes == 0) return nullptr;
    return IAllocator::MakeUniquePtr<T>(Info().GetAllocator(OrtMemType::OrtMemTypeDefault),
                                        count_or_bytes, false, stream);
  }

 protected:
  inline Status CopyTensor(const Tensor& src, Tensor& dst) const {
    return Info().GetDataTransferManager().CopyTensor(src, dst);
  }
};

}  // namespace musa
}  // namespace onnxruntime
