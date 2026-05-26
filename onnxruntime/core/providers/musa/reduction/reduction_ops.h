// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/musa_kernel.h"
#include "core/providers/musa/musa_utils.h"

namespace onnxruntime {
namespace musa {

template <bool allow_multi_axes>
class ReduceKernel : public MusaKernel {
 public:
  explicit ReduceKernel(const OpKernelInfo& info) : MusaKernel(info) {
    int64_t keepdims = 1;
    info.GetAttrOrDefault("keepdims", &keepdims, static_cast<int64_t>(1));
    keepdims_ = (keepdims == 1);

    int64_t noop_with_empty_axes = 0;
    info.GetAttrOrDefault("noop_with_empty_axes", &noop_with_empty_axes, static_cast<int64_t>(0));
    noop_with_empty_axes_ = (noop_with_empty_axes == 1);

    // Handle axis/axes attribute correctly based on allow_multi_axes
    if (allow_multi_axes) {
      // Multi-axis reduction: use "axes" attribute (plural)
      auto axes_vector = info.GetAttrsOrDefault<int64_t>("axes");
      axes_.assign(axes_vector.begin(), axes_vector.end());
    } else {
      // Single-axis reduction: use "axis" attribute (singular)
      auto v = info.GetAttrOrDefault<int64_t>("axis", 0);
      axes_.push_back(v);
    }
  }

  // Public getters for use by generic functions
  [[nodiscard]] bool GetKeepDims() const { return keepdims_; }
  [[nodiscard]] bool GetNoopWithEmptyAxes() const { return noop_with_empty_axes_; }
  [[nodiscard]] const TensorShapeVector& GetAxes() const { return axes_; }

  // Helper to compute output shape after reduction
  Status ComputeOutputShape(const TensorShape& input_shape,
                           const TensorShapeVector& axes,
                           bool keepdims,
                           TensorShape& output_shape) const;

  // Helper to setup axes for reduction
  Status PrepareAxesForReduction(const TensorShape& input_shape,
                                TensorShapeVector& processed_axes) const;

 protected:
  bool keepdims_;
  bool noop_with_empty_axes_;
  TensorShapeVector axes_;
};

template <typename T>
class ReduceMax final : public ReduceKernel<true> {
 public:
  explicit ReduceMax(const OpKernelInfo& info) : ReduceKernel<true>(info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;
};

template <typename T>
class ReduceMin final : public ReduceKernel<true> {
 public:
  explicit ReduceMin(const OpKernelInfo& info) : ReduceKernel<true>(info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;
};

template <typename T>
class ReduceSum final : public ReduceKernel<true> {
 public:
  explicit ReduceSum(const OpKernelInfo& info) : ReduceKernel<true>(info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;
};

template <typename T>
class ReduceMean final : public ReduceKernel<true> {
 public:
  explicit ReduceMean(const OpKernelInfo& info) : ReduceKernel<true>(info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;
};

template <typename T>
class ReduceProd final : public ReduceKernel<true> {
 public:
  explicit ReduceProd(const OpKernelInfo& info) : ReduceKernel<true>(info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;
};

template <typename T>
class ReduceL2 final : public ReduceKernel<true> {
 public:
  explicit ReduceL2(const OpKernelInfo& info) : ReduceKernel<true>(info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;
};

// ArgMax/ArgMin operations that reduce to indices
template <typename T>
class ArgMax final : public ReduceKernel<false> {
 public:
  explicit ArgMax(const OpKernelInfo& info) : ReduceKernel<false>(info) {
    // Get select_last_index attribute
    int64_t select_last_index = 0;
    if (info.GetAttr<int64_t>("select_last_index", &select_last_index).IsOK()) {
      select_last_index_ = (select_last_index != 0);
    } else {
      select_last_index_ = false;
    }
  }
  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  bool select_last_index_;
};

template <typename T>
class ArgMin final : public ReduceKernel<false> {
 public:
  explicit ArgMin(const OpKernelInfo& info) : ReduceKernel<false>(info) {
    // Get select_last_index attribute
    int64_t select_last_index = 0;
    if (info.GetAttr<int64_t>("select_last_index", &select_last_index).IsOK()) {
      select_last_index_ = (select_last_index != 0);
    } else {
      select_last_index_ = false;
    }
  }
  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  bool select_last_index_;
};

// Generic prepare function for all reduce operations
template <typename T>
Status PrepareReduceOperation(const ReduceKernel<true>* kernel,
                              OpKernelContext* ctx,
                              MusaPreparation& prepare);

// Specialized prepare function for ArgMax/ArgMin operations
template <typename T>
Status PrepareArgMaxMinOperation(const ReduceKernel<false>* kernel,
                                 OpKernelContext* ctx,
                                 MusaPreparation& prepare);

} // namespace musa
} // namespace onnxruntime
