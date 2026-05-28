// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/common/narrow.h"
#include "core/framework/op_kernel.h"
#include "core/common/float16.h"
#endif

#include <gsl/gsl>
#include <limits>
#include <type_traits>

namespace onnxruntime {

class Shape final : public OpKernel {
 public:
  Shape(const OpKernelInfo& info) : OpKernel(info) {
    info.GetAttrOrDefault<int64_t>("start", &start_index_, 0);

    if (start_index_ != 0) {
      // "start" is provided and is non-default (default is 0)
      needs_slicing_ = true;
    }

    if (info.GetAttr<int64_t>("end", &end_index_).IsOK()) {
      needs_slicing_ = true;
    }
  }

  // Takes a tensor as input and outputs an 1D int64 tensor
  // containing the shape of the input tensor.
  Status Compute(OpKernelContext* context) const override {
    const auto* input = context->Input<Tensor>(0);
    const TensorShape& input_shape = input->Shape();

    int64_t rank = gsl::narrow_cast<int64_t>(input_shape.NumDimensions());

    if (!needs_slicing_) {  // vanilla use of Shape (no slicing)
      Tensor* output = context->Output(0, {rank});
      input_shape.CopyDims(output->MutableData<int64_t>(), static_cast<size_t>(rank));
    } else {  // slicing is needed
      int64_t true_start = start_index_;
      int64_t true_end = end_index_;

      // Deal with negative(s) and clamp
      true_start = true_start < 0 ? true_start + rank : true_start;
      true_start = true_start < 0 ? 0 : ((true_start > rank) ? rank : true_start);

      true_end = true_end < 0 ? true_end + rank : true_end;
      true_end = true_end < 0 ? 0 : ((true_end > rank) ? rank : true_end);

      auto slice_length = true_end - true_start;
      Tensor* output = context->Output(0, {slice_length < 0 ? 0 : slice_length});

      if (slice_length > 0) {
        input_shape.CopyDims(output->MutableData<int64_t>(), onnxruntime::narrow<size_t>(true_start), onnxruntime::narrow<size_t>(slice_length));
      }
    }

    return Status::OK();
  }

 private:
  bool needs_slicing_ = false;
  int64_t start_index_ = 0;
  int64_t end_index_ = std::numeric_limits<int64_t>::max();
};

class ShapeN final : public OpKernel {
 public:
  explicit ShapeN(const OpKernelInfo& info) : OpKernel(info) {
    info.GetAttrOrDefault<int64_t>("out_type", &out_type_, ONNX_NAMESPACE::TensorProto_DataType_INT32);
  }

  Status Compute(OpKernelContext* context) const override {
    const size_t input_count = static_cast<size_t>(context->InputCount());
    const size_t output_count = static_cast<size_t>(context->OutputCount());
    ORT_RETURN_IF_NOT(input_count == output_count,
                      "ShapeN: input and output counts must match");

    for (size_t i = 0; i < input_count; ++i) {
      const Tensor* input = context->Input<Tensor>(static_cast<int>(i));
      ORT_RETURN_IF_NOT(input != nullptr, "ShapeN: input tensor is null");

      const TensorShape& input_shape = input->Shape();
      const int64_t rank = gsl::narrow_cast<int64_t>(input_shape.NumDimensions());
      switch (out_type_) {
        case ONNX_NAMESPACE::TensorProto_DataType_INT32:
          FillOutput<int32_t>(context, static_cast<int>(i), input_shape, rank);
          break;
        case ONNX_NAMESPACE::TensorProto_DataType_INT64:
          FillOutput<int64_t>(context, static_cast<int>(i), input_shape, rank);
          break;
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
          FillOutput<float>(context, static_cast<int>(i), input_shape, rank);
          break;
        case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
          FillOutput<double>(context, static_cast<int>(i), input_shape, rank);
          break;
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
          FillOutput<MLFloat16>(context, static_cast<int>(i), input_shape, rank);
          break;
        default:
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "ShapeN: unsupported out_type ", out_type_);
      }
    }

    return Status::OK();
  }

 private:
  template <typename T>
  static void FillOutput(OpKernelContext* context, int output_index,
                         const TensorShape& input_shape, int64_t rank) {
    Tensor* output = context->Output(output_index, TensorShape{rank});
    T* output_data = output->MutableData<T>();
    for (int64_t j = 0; j < rank; ++j) {
      output_data[j] = ConvertDim<T>(input_shape.GetDims()[onnxruntime::narrow<size_t>(j)]);
    }
  }

  template <typename T>
  static T ConvertDim(int64_t dim) {
    if constexpr (std::is_same<T, MLFloat16>::value) {
      return MLFloat16(static_cast<float>(dim));
    } else {
      return static_cast<T>(dim);
    }
  }

  int64_t out_type_;
};

}  // namespace onnxruntime
