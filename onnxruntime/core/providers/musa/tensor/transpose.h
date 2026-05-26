// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cpu/tensor/transpose.h"
#include "core/providers/musa/musa_kernel.h"
#include "core/providers/musa/musa_utils.h"
#include "gsl/gsl"

namespace onnxruntime {
namespace musa {

// Helper function to perform MUSA transpose using muDNN Permute operation
// This is a lower-level API that can be used by other ops (e.g., Conv for weight prepacking)
Status DoMusaTranspose(const void* input_data,
                       void* output_data,
                       const std::vector<int64_t>& input_shape,
                       const std::vector<size_t>& perm,
                       ::musa::dnn::Tensor::Type data_type,
                       musaStream_t stream,
                       int device_id);

template <typename T>
class Transpose final : public MusaKernel, public TransposeBase {
 public:
  Transpose(const OpKernelInfo& info) : MusaKernel(info), TransposeBase(info) {
  }

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  Status Prepare(OpKernelContext* ctx, MusaPreparation& prepare) const;
};

}  // namespace musa
}  // namespace onnxruntime
