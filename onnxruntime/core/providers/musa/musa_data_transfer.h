// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/data_transfer.h"
#include "core/providers/shared_library/provider_api.h"
#include <musa_runtime.h>

namespace onnxruntime {

class MusaDataTransfer : public IDataTransfer {
 public:
  MusaDataTransfer();
  ~MusaDataTransfer();

  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;

  common::Status CopyTensor(const Tensor& src, Tensor& dst) const override;

  common::Status CopyTensorAsync(const Tensor& src, Tensor& dst, Stream& stream) const override;

 private:
  // Helper method to check if pointer is on device memory
  bool IsDevicePointer(const void* ptr) const;

  // Helper method to perform MUSA memory copy (synchronous)
  common::Status MusaMemcpyHelper(void* dst, const void* src, size_t bytes, musaMemcpyKind kind) const;

  // Helper method to perform MUSA memory copy (asynchronous)
  common::Status MusaMemcpyAsyncHelper(void* dst, const void* src, size_t bytes,
                                      musaMemcpyKind kind, musaStream_t stream) const;
};

}  // namespace onnxruntime
