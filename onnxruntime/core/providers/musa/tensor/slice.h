// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/musa_kernel.h"
#include "core/providers/musa/musa_utils.h"
#include "core/providers/cpu/tensor/slice_helper.h"
#include "core/providers/cpu/tensor/slice_compute_metadata.h"

namespace onnxruntime {
namespace musa {

// MUSA Slice implementation that doesn't inherit from SliceBase
// to avoid undefined symbol issues with dynamic library loading
template <bool dynamic>
class Slice : public MusaKernel {
  public:
  explicit Slice(const OpKernelInfo& info) : MusaKernel(info) {
    if (!dynamic) {
      // For static slice (versions 1-9), get attributes
      ORT_THROW_IF_ERROR(info.GetAttrs("starts", starts_));
      ORT_THROW_IF_ERROR(info.GetAttrs("ends", ends_));

      // axes attribute is optional
      if (info.GetAttrs("axes", axes_).IsOK()) {
        // axes provided
      } else {
        // axes will be set to default [0, 1, ..., len(starts)-1] in ComputeInternal
        axes_.clear();
      }
    }
  }

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  virtual const Tensor* GetSlicedOrUnslicedTensor(OpKernelContext* ctx) const;
  virtual Status FillInputVectors(OpKernelContext* ctx, TensorShapeVector& input_starts,
                                  TensorShapeVector& input_ends, TensorShapeVector& input_axes,
                                  TensorShapeVector& input_steps) const;

  template <typename T>
  Status Prepare(OpKernelContext* ctx, MusaPreparation& prepare,
                 SliceOp::PrepareForComputeMetadata& compute_metadata) const;

  // Attributes for static slice (versions 1-9)
  TensorShapeVector starts_;
  TensorShapeVector ends_;
  TensorShapeVector axes_;

  // Helper function to flatten output dimensions (replaces SliceBase::FlattenOutputDims)
  Status FlattenOutputDims(gsl::span<const int64_t> input_dimensions,
                          gsl::span<const int64_t> output_dims,
                          TensorShapeVector& starts, TensorShapeVector& ends, TensorShapeVector& steps,
                          TensorShapeVector*& p_flattened_input_dims, TensorShapeVector*& p_flattened_output_dims) const;

  // Helper function to prepare for compute (replaces SliceBase::PrepareForCompute)
  Status PrepareForCompute(gsl::span<const int64_t> raw_starts, gsl::span<const int64_t> raw_ends,
                          gsl::span<const int64_t> raw_axes, SliceOp::PrepareForComputeMetadata& compute_metadata) const;

  Status PrepareForCompute(gsl::span<const int64_t> raw_starts, gsl::span<const int64_t> raw_ends,
                          gsl::span<const int64_t> raw_axes, gsl::span<const int64_t> raw_steps,
                          SliceOp::PrepareForComputeMetadata& compute_metadata) const;

  // Helper function to fill vectors from input tensors (replaces SliceBase::FillVectorsFromInput)
  Status FillVectorsFromInput(const Tensor& start_tensor, const Tensor& ends_tensor,
                             const Tensor* axes_tensor, const Tensor* steps_tensor,
                             TensorShapeVector& input_starts, TensorShapeVector& input_ends,
                             TensorShapeVector& input_axes, TensorShapeVector& input_steps) const;

  // Get starts/ends/axes attributes for static slice
  [[nodiscard]] gsl::span<const int64_t> StartsAttribute() const { return gsl::make_span(starts_); }
  [[nodiscard]] gsl::span<const int64_t> EndsAttribute() const { return gsl::make_span(ends_); }
  [[nodiscard]] gsl::span<const int64_t> AxesAttribute() const { return gsl::make_span(axes_); }
};

}  // namespace musa
}  // namespace onnxruntime
