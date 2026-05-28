// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/reduction/reduction_ops.h"
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/common.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/musa_execution_provider.h"
#include "core/providers/musa/reduction/reduce_l2_keepdims_impl.h"
#include "core/providers/musa/reduction/reduce_prod_int32_impl.h"
#include <algorithm>
#include <musa_runtime.h>
#include <mudnn.h>
#include <string>
#include <type_traits>
#include <vector>

using onnxruntime::common::Status;
namespace onnxruntime {
namespace musa {

namespace {

::musa::dnn::Tensor::Format GetTensorFormatForShape(const TensorShape& shape) {
  if (shape.NumDimensions() == 0 || shape.NumDimensions() == 1 || shape.NumDimensions() == 3) {
    return ::musa::dnn::Tensor::Format::NCW;
  }
  if (shape.NumDimensions() == 4) {
    return ::musa::dnn::Tensor::Format::NCHW;
  }
  if (shape.NumDimensions() == 5) {
    return ::musa::dnn::Tensor::Format::NCDHW;
  }
  return ::musa::dnn::Tensor::Format::NCHW;
}

Status SetupMusaTensorFromBuffer(::musa::dnn::Tensor& musa_tensor,
                                 const void* data_ptr,
                                 const TensorShape& shape,
                                 ::musa::dnn::Tensor::Type data_type) {
  auto status = musa_tensor.SetType(data_type);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to set MUSA tensor type, status: " +
                               std::to_string(static_cast<int>(status)));
  }

  if (data_ptr == nullptr && shape.Size() > 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Scratch tensor data pointer is null for non-empty tensor");
  }

  if (data_ptr != nullptr) {
    status = musa_tensor.SetAddr(data_ptr);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set MUSA tensor address, status: " +
                                 std::to_string(static_cast<int>(status)));
    }
  }

  status = musa_tensor.SetFormat(GetTensorFormatForShape(shape));
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to set scratch tensor format, status: " +
                               std::to_string(static_cast<int>(status)));
  }

  if (shape.NumDimensions() == 0) {
    std::vector<int64_t> scalar_dims = {1};
    std::vector<int64_t> scalar_strides = {1};
    status = musa_tensor.SetNdInfo(static_cast<int>(scalar_dims.size()),
                                   scalar_dims.data(),
                                   scalar_strides.data());
  } else {
    std::vector<int64_t> dims;
    dims.reserve(shape.NumDimensions());
    for (size_t i = 0; i < shape.NumDimensions(); ++i) {
      dims.push_back(shape[i]);
    }

    const auto strides = CalculateStrides(shape);
    status = musa_tensor.SetNdInfo(static_cast<int>(dims.size()), dims.data(), strides.data());
  }

  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to set scratch tensor shape info, status: " +
                               std::to_string(static_cast<int>(status)));
  }

  return Status::OK();
}

Status RunMusaBinaryOp(::musa::dnn::Handle& handle,
                       ::musa::dnn::Binary::Mode mode,
                       ::musa::dnn::Tensor& output_tensor,
                       const ::musa::dnn::Tensor& input_a_tensor,
                       const ::musa::dnn::Tensor& input_b_tensor) {
  try {
    ::musa::dnn::Binary binary_op;
    auto status = binary_op.SetMode(mode);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set mudnn Binary mode, status: " +
                                 std::to_string(static_cast<int>(status)));
    }

    status = binary_op.Run(handle, output_tensor, input_a_tensor, input_b_tensor);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "mudnn Binary operation failed, status: " +
                                 std::to_string(static_cast<int>(status)));
    }
  } catch (const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Exception in mudnn Binary operation: " + std::string(e.what()));
  }

  return Status::OK();
}

Status RunMusaUnaryOp(::musa::dnn::Handle& handle,
                      ::musa::dnn::Unary::Mode mode,
                      ::musa::dnn::Tensor& output_tensor,
                      const ::musa::dnn::Tensor& input_tensor) {
  try {
    ::musa::dnn::Unary unary_op;
    auto status = unary_op.SetMode(mode);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set mudnn Unary mode, status: " +
                                 std::to_string(static_cast<int>(status)));
    }

    status = unary_op.Run(handle, output_tensor, input_tensor);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "mudnn Unary operation failed, status: " +
                                 std::to_string(static_cast<int>(status)));
    }
  } catch (const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Exception in mudnn Unary operation: " + std::string(e.what()));
  }

  return Status::OK();
}

template <typename T>
Status RunMusaFill(::musa::dnn::Handle& handle,
                   ::musa::dnn::Tensor& output_tensor,
                   T value) {
  try {
    ::musa::dnn::Fill fill_op;
    ::musa::dnn::Status status;
    if constexpr (std::is_integral_v<T>) {
      status = fill_op.SetValue(static_cast<int64_t>(value));
    } else {
      status = fill_op.SetValue(static_cast<double>(value));
    }
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set mudnn Fill value, status: " +
                                 std::to_string(static_cast<int>(status)));
    }

    status = fill_op.Run(handle, output_tensor);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "mudnn Fill operation failed, status: " +
                                 std::to_string(static_cast<int>(status)));
    }
  } catch (const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Exception in mudnn Fill operation: " + std::string(e.what()));
  }

  return Status::OK();
}

int64_t ComputeReductionElementCount(const TensorShape& input_shape,
                                     const TensorShapeVector& axes) {
  int64_t count = 1;
  for (auto axis : axes) {
    count *= input_shape[static_cast<size_t>(axis)];
  }
  return count;
}

Status PrepareReduceL2Params(const TensorShape& input_shape,
                             const TensorShape& output_shape,
                             const TensorShapeVector& axes,
                             bool keepdims,
                             ReduceL2KeepDimsParams& params) {
  const size_t rank = input_shape.NumDimensions();
  if (rank > static_cast<size_t>(kReduceL2KeepDimsMaxRank)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "ReduceL2 MUSA kernel only supports rank <= ",
                           kReduceL2KeepDimsMaxRank);
  }

  params.rank = static_cast<int32_t>(rank);
  params.num_axes = static_cast<int32_t>(axes.size());
  params.output_size = 1;
  params.reduce_size = 1;

  bool reduced_axes[kReduceL2KeepDimsMaxRank] = {};
  for (size_t i = 0; i < axes.size(); ++i) {
    const int64_t axis = axes[i];
    if (axis < 0 || axis >= static_cast<int64_t>(rank)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid ReduceL2 axis: ", axis);
    }

    params.axes[i] = static_cast<int32_t>(axis);
    reduced_axes[axis] = true;
    params.reduced_axes[axis] = 1;
    params.reduce_size *= input_shape[static_cast<size_t>(axis)];
  }

  for (size_t dim = 0; dim < rank; ++dim) {
    params.input_dims[dim] = input_shape[dim];
  }

  params.output_size = output_shape.Size();
  std::vector<int64_t> output_strides(output_shape.NumDimensions(), 1);
  int64_t output_stride = 1;
  for (int64_t dim = static_cast<int64_t>(output_shape.NumDimensions()) - 1; dim >= 0; --dim) {
    output_strides[static_cast<size_t>(dim)] = output_stride;
    output_stride *= output_shape[static_cast<size_t>(dim)];
  }

  int64_t input_stride = 1;
  for (int64_t dim = static_cast<int64_t>(rank) - 1; dim >= 0; --dim) {
    params.input_strides[dim] = input_stride;
    input_stride *= params.input_dims[dim];
  }

  size_t output_dim_index = 0;
  for (size_t dim = 0; dim < rank; ++dim) {
    if (reduced_axes[dim]) {
      params.output_strides[dim] = 0;
      if (keepdims) {
        ++output_dim_index;
      }
      continue;
    }

    if (output_dim_index >= output_strides.size()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid ReduceL2 output rank mapping");
    }
    params.output_strides[dim] = output_strides[output_dim_index++];
  }

  return Status::OK();
}

Status PrepareReduceProdInt32Params(const TensorShape& input_shape,
                                    const TensorShape& output_shape,
                                    const TensorShapeVector& axes,
                                    bool keepdims,
                                    ReduceProdInt32Params& params) {
  const size_t rank = input_shape.NumDimensions();
  ORT_RETURN_IF_NOT(rank <= static_cast<size_t>(kReduceProdInt32MaxRank),
                    "ReduceProd int32 MUSA kernel only supports rank <= ",
                    kReduceProdInt32MaxRank);

  params.rank = static_cast<int32_t>(rank);
  params.input_size = input_shape.Size();
  params.output_size = output_shape.Size();
  std::fill_n(params.input_strides, kReduceProdInt32MaxRank, 0);
  std::fill_n(params.output_strides, kReduceProdInt32MaxRank, 0);

  bool reduced_axes[kReduceProdInt32MaxRank] = {};
  for (int64_t axis : axes) {
    if (axis < 0 || axis >= static_cast<int64_t>(rank)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid ReduceProd axis: ", axis);
    }
    reduced_axes[axis] = true;
  }

  int64_t input_stride = 1;
  for (int64_t dim = static_cast<int64_t>(rank) - 1; dim >= 0; --dim) {
    params.input_strides[dim] = input_stride;
    input_stride *= input_shape[static_cast<size_t>(dim)];
  }

  std::vector<int64_t> output_strides(output_shape.NumDimensions(), 1);
  int64_t output_stride = 1;
  for (int64_t dim = static_cast<int64_t>(output_shape.NumDimensions()) - 1; dim >= 0; --dim) {
    output_strides[static_cast<size_t>(dim)] = output_stride;
    output_stride *= output_shape[static_cast<size_t>(dim)];
  }

  size_t output_dim_index = 0;
  for (size_t dim = 0; dim < rank; ++dim) {
    if (reduced_axes[dim]) {
      params.output_strides[dim] = 0;
      if (keepdims) {
        ++output_dim_index;
      }
      continue;
    }

    if (output_dim_index >= output_strides.size()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid ReduceProd int32 output rank mapping");
    }
    params.output_strides[dim] = output_strides[output_dim_index++];
  }

  return Status::OK();
}

}  // namespace

template <bool allow_multi_axes>
Status ReduceKernel<allow_multi_axes>::ComputeOutputShape(const TensorShape& input_shape,
                                                         const TensorShapeVector& axes,
                                                         bool keepdims,
                                                         TensorShape& output_shape) const {
  size_t ndim = input_shape.NumDimensions();
  std::vector<int64_t> output_dims;

  for (size_t i = 0; i < ndim; ++i) {
    bool is_reduced_axis = false;
    for (auto axis : axes) {
      // Handle negative axis
      int64_t normalized_axis = axis < 0 ? axis + static_cast<int64_t>(ndim) : axis;
      if (normalized_axis == static_cast<int64_t>(i)) {
        is_reduced_axis = true;
        break;
      }
    }

    if (is_reduced_axis) {
      if (keepdims) {
        output_dims.push_back(1);
      }
    } else {
      output_dims.push_back(input_shape[i]);
    }
  }

  output_shape = TensorShape(output_dims);
  return Status::OK();
}

template <bool allow_multi_axes>
Status ReduceKernel<allow_multi_axes>::PrepareAxesForReduction(const TensorShape& input_shape,
                                                              TensorShapeVector& processed_axes) const {
  size_t ndim = input_shape.NumDimensions();
  processed_axes.clear();

  if (axes_.empty()) {
    // If no axes specified, reduce all axes
    for (size_t i = 0; i < ndim; ++i) {
      processed_axes.push_back(static_cast<int64_t>(i));
    }
  } else {
    // Normalize axes (handle negative values)
    for (auto axis : axes_) {
      int64_t normalized_axis = axis < 0 ? axis + static_cast<int64_t>(ndim) : axis;
      if (normalized_axis < 0 || normalized_axis >= static_cast<int64_t>(ndim)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Invalid axis: ", axis, " for tensor with ", ndim, " dimensions");
      }
      processed_axes.push_back(normalized_axis);
    }
  }

  // Sort and remove duplicates
  std::sort(processed_axes.begin(), processed_axes.end());
  processed_axes.erase(std::unique(processed_axes.begin(), processed_axes.end()), processed_axes.end());

  return Status::OK();
}

// Helper function to handle axes from both attribute and input (for opset 13+)
template <bool allow_multi_axes>
Status PrepareAxesForReductionFromContext(const ReduceKernel<allow_multi_axes>* kernel,
                                         OpKernelContext* ctx,
                                         const TensorShape& input_shape,
                                         TensorShapeVector& processed_axes) {
  size_t ndim = input_shape.NumDimensions();
  processed_axes.clear();

  TensorShapeVector input_axes;

  // Check if axes is provided as input (opset 13+) or attribute
  if (ctx->InputCount() == 2) {
    // axes is provided as input tensor
    const Tensor* axes_tensor = ctx->Input<Tensor>(1);
    if (axes_tensor != nullptr) {
      // Validate axes tensor shape
      if (axes_tensor->Shape().NumDimensions() != 1) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "An axes tensor must be a vector tensor");
      }

      auto nDims = static_cast<size_t>(axes_tensor->Shape()[0]);
      const auto* data = axes_tensor->Data<int64_t>();
      input_axes.insert(input_axes.begin(), data, data + nDims);
    }
    // else: axes_tensor is null, which means reduce all axes
  } else {
    // Use axes from attribute
    const auto& attr_axes = kernel->GetAxes();
    input_axes.assign(attr_axes.begin(), attr_axes.end());
  }

  if (input_axes.empty()) {
    // If no axes specified, check noop_with_empty_axes flag
    if (kernel->GetNoopWithEmptyAxes()) {
      // noop_with_empty_axes=true: empty axes means no reduction, return empty processed_axes
      // This will be handled by the noop case in the caller
      // Leave processed_axes empty
    } else {
      // noop_with_empty_axes=false: empty axes means reduce all axes
      for (size_t i = 0; i < ndim; ++i) {
        processed_axes.push_back(static_cast<int64_t>(i));
      }
    }
  } else {
    // Normalize axes (handle negative values)
    for (auto axis : input_axes) {
      int64_t normalized_axis = axis < 0 ? axis + static_cast<int64_t>(ndim) : axis;
      if (normalized_axis < 0 || normalized_axis >= static_cast<int64_t>(ndim)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Invalid axis: ", axis, " for tensor with ", ndim, " dimensions");
      }
      processed_axes.push_back(normalized_axis);
    }
  }

  // Sort and remove duplicates
  std::sort(processed_axes.begin(), processed_axes.end());
  processed_axes.erase(std::unique(processed_axes.begin(), processed_axes.end()), processed_axes.end());

  return Status::OK();
}

// Overload for ArgMax/ArgMin (allow_multi_axes = false)
template <>
Status PrepareAxesForReductionFromContext<false>(const ReduceKernel<false>* kernel,
                                                OpKernelContext* ctx,
                                                const TensorShape& input_shape,
                                                TensorShapeVector& processed_axes) {
  size_t ndim = input_shape.NumDimensions();
  processed_axes.clear();

  TensorShapeVector input_axes;

  // Check if axes is provided as input (opset 13+) or attribute
  if (ctx->InputCount() == 2) {
    // axes is provided as input tensor
    const Tensor* axes_tensor = ctx->Input<Tensor>(1);
    if (axes_tensor != nullptr) {
      // Validate axes tensor shape
      if (axes_tensor->Shape().NumDimensions() != 1) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "An axes tensor must be a vector tensor");
      }

      auto nDims = static_cast<size_t>(axes_tensor->Shape()[0]);
      const auto* data = axes_tensor->Data<int64_t>();
      input_axes.insert(input_axes.begin(), data, data + nDims);
    }
    // else: axes_tensor is null, which means reduce all axes
  } else {
    // Use axes from attribute
    const auto& attr_axes = kernel->GetAxes();
    input_axes.assign(attr_axes.begin(), attr_axes.end());
  }

  if (input_axes.empty()) {
    // If no axes specified, check noop_with_empty_axes flag
    if (kernel->GetNoopWithEmptyAxes()) {
      // noop_with_empty_axes=true: empty axes means no reduction, return empty processed_axes
      // This will be handled by the noop case in the caller
      // Leave processed_axes empty
    } else {
      // noop_with_empty_axes=false: empty axes means reduce all axes
      for (size_t i = 0; i < ndim; ++i) {
        processed_axes.push_back(static_cast<int64_t>(i));
      }
    }
  } else {
    // Normalize axes (handle negative values)
    for (auto axis : input_axes) {
      int64_t normalized_axis = axis < 0 ? axis + static_cast<int64_t>(ndim) : axis;
      if (normalized_axis < 0 || normalized_axis >= static_cast<int64_t>(ndim)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Invalid axis: ", axis, " for tensor with ", ndim, " dimensions");
      }
      processed_axes.push_back(normalized_axis);
    }
  }

  // Sort and remove duplicates
  std::sort(processed_axes.begin(), processed_axes.end());
  processed_axes.erase(std::unique(processed_axes.begin(), processed_axes.end()), processed_axes.end());

  return Status::OK();
}

// Generic MUSA device-based Reduce implementation using MusaPreparation and mudnn library
template <typename T>
Status SimpleMusaReduceOp(const MusaPreparation& prepare,
                          const MusaKernel* kernel,
                          OpKernelContext* ctx,
                          const TensorShapeVector& axes,
                          ::musa::dnn::Reduce::Mode reduce_mode,
                          const float* norm_ord = nullptr,
                          bool initialize_output = false) {
  // Handle empty tensors (zero elements) - no computation needed
  if (prepare.input_a_shape.Size() == 0 || prepare.output_size == 0) {
    return Status::OK();
  }

  // Get tensor data from prepared MUSA tensors
  const T* input_data = reinterpret_cast<const T*>(prepare.input_a_ptr);
  T* output_data = reinterpret_cast<T*>(prepare.output_ptr);

  // Validate prepared tensors
  if (input_data == nullptr || output_data == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid prepared tensor pointers");
  }

  if (prepare.inputTensors.empty() || prepare.outputTensors.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid prepared tensor configuration");
  }

  if (prepare.input_a_shape.Size() == prepare.output_size) {
    if (prepare.input_a_ptr != prepare.output_ptr) {
      musaError_t copy_status = musaMemcpyAsync(output_data,
                                                input_data,
                                                prepare.output_size * sizeof(T),
                                                musaMemcpyDeviceToDevice,
                                                kernel->Stream(ctx));
      if (copy_status != musaSuccess) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Failed to copy same-size Reduce output, status: ",
                               static_cast<int>(copy_status));
      }
    }
    return Status::OK();
  }

  if (initialize_output &&
      (reduce_mode == ::musa::dnn::Reduce::Mode::ADD ||
       reduce_mode == ::musa::dnn::Reduce::Mode::MEAN)) {
    musaError_t memset_status = musaMemsetAsync(output_data,
                                                0,
                                                prepare.output_size * sizeof(T),
                                                kernel->Stream(ctx));
    if (memset_status != musaSuccess) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to initialize Reduce output, status: ",
                             static_cast<int>(memset_status));
    }
  }

  // Use mudnn Reduce class for device computation
  try {
    // Create mudnn Reduce operation
    ::musa::dnn::Reduce reduce_op;

    // Set the operation mode
    auto status = reduce_op.SetMode(reduce_mode);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set mudnn Reduce mode, status: " +
                             std::to_string(static_cast<int>(status)));
    }

    if (norm_ord != nullptr) {
      status = reduce_op.SetNormOrd(*norm_ord);
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Failed to set mudnn Reduce norm order");
      }
    }

    // Convert axes to int array for mudnn
    std::vector<int> int_axes;
    for (auto axis : axes) {
      int_axes.push_back(static_cast<int>(axis));
    }

    // Set reduction dimensions
    status = reduce_op.SetDim(static_cast<int>(int_axes.size()), int_axes.data());
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set mudnn Reduce dimensions");
    }

    // Query workspace size
    size_t workspace_size = 0;
    status = reduce_op.GetWorkspaceSize(prepare.GetHandle(), workspace_size,
                                       const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]),
                                       prepare.inputTensors[0]);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to get mudnn Reduce workspace size");
    }

    std::vector<IAllocatorUniquePtr<void>> workspace_buffers;
    auto memory_allocator = [&](size_t size) -> ::musa::dnn::MemoryHandler {
      if (size == 0) {
        return {nullptr, [](void*) {}};
      }

      workspace_buffers.emplace_back(kernel->GetScratchBuffer<void>(size, ctx->GetComputeStream()));
      return {workspace_buffers.back().get(), [](void*) {}};
    };

    // Run the reduce operation
    status = reduce_op.Run(prepare.GetHandle(),
                          const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]),  // output tensor
                          prepare.inputTensors[0],   // input tensor
                          memory_allocator);

    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "mudnn Reduce operation failed, status: " +
                             std::to_string(static_cast<int>(status)));
    }


  } catch (const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Exception in mudnn Reduce operation: " +
                           std::string(e.what()));
  }

  return Status::OK();
}

// Generic prepare function for all reduce operations
template <typename T>
Status PrepareReduceOperation(const ReduceKernel<true>* kernel,
                              OpKernelContext* ctx,
                              MusaPreparation& prepare) {
  // 1. Get input tensor
  const Tensor* X = ctx->Input<Tensor>(0);
  if (!X) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid input tensor");
  }

  // 2. Process axes for reduction, handling both attribute and input cases
  TensorShapeVector processed_axes;
  ORT_RETURN_IF_ERROR(PrepareAxesForReductionFromContext(kernel, ctx, X->Shape(), processed_axes));

  // 3. Handle noop_with_empty_axes case
  if (processed_axes.empty() && kernel->GetNoopWithEmptyAxes()) {
    // No reduction needed, just copy input to output
    Tensor* Y = ctx->Output(0, X->Shape());
    if (!Y) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor");
    }

    // For noop case, we still need to set up pointers for potential copy operation
    prepare.input_a_ptr = X->DataRaw();
    prepare.output_ptr = Y->MutableDataRaw();
    prepare.output_size = Y->Shape().Size();
    prepare.input_a_shape = X->Shape();
    prepare.output_shape = Y->Shape();
    return Status::OK();
  }

  // 4. Compute output shape
  TensorShape output_shape;
  ORT_RETURN_IF_ERROR(kernel->ComputeOutputShape(X->Shape(), processed_axes, kernel->GetKeepDims(), output_shape));

  // 5. Create output tensor
  Tensor* Y = ctx->Output(0, output_shape);
  if (!Y) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor");
  }

  // 6. Store tensor pointers and shapes in preparation
  prepare.input_a_ptr = X->DataRaw();
  prepare.output_ptr = Y->MutableDataRaw();
  prepare.output_size = Y->Shape().Size();
  prepare.input_a_shape = X->Shape();
  prepare.output_shape = Y->Shape();

  // For tensors with zero elements, data pointers might be null, which is valid
  if ((prepare.input_a_shape.Size() > 0 && prepare.input_a_ptr == nullptr) ||
      (prepare.output_size > 0 && prepare.output_ptr == nullptr)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid tensor data pointers");
  }

  // 7. Get MUSA data type
  const auto musaType = GetMusaDataType<T>();

  // 8. Prepare MUSA tensors - using ORT_TRY/ORT_CATCH like other MusaEP operations
  ORT_TRY {
    // Set up MUSA stream for asynchronous execution
    auto stream = kernel->Stream(ctx);
    if (prepare.handle) {
      if (stream) {
        auto status = prepare.handle->SetStream(stream);
        if (status != ::musa::dnn::Status::SUCCESS) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                 "Failed to set MUSA stream, status: " +
                                     std::to_string(static_cast<int>(status)));
        }
      } else {
        // Use default stream for backward compatibility
        LOGS_DEFAULT(WARNING) << "No stream provided, using default MUSA stream";
      }
    }

    // Initialize tensors vectors
    prepare.inputTensors.resize(1);
    prepare.outputTensors.resize(1);

    // Setup input tensor
    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.inputTensors[0], X, musaType, &prepare));

    // Setup output tensor
    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.outputTensors[0], Y, musaType, &prepare));
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  return Status::OK();
}

template <typename T>
Status CopyReduceInputToOutput(const Tensor* X,
                               Tensor* Y,
                               const MusaKernel* kernel,
                               OpKernelContext* ctx) {
  if (X->SizeInBytes() == 0) {
    return Status::OK();
  }

  musaError_t status = musaMemcpyAsync(Y->MutableDataRaw(), X->DataRaw(),
                                       X->SizeInBytes(), musaMemcpyDeviceToDevice,
                                       kernel->Stream(ctx));
  if (status != musaSuccess) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to copy tensor data");
  }
  return Status::OK();
}

template <typename T>
Status SetupReduceTensorFromBuffer(::musa::dnn::Tensor& tensor,
                                   const void* buffer,
                                   const TensorShape& shape) {
  return SetupMusaTensorFromBuffer(tensor, buffer, shape, GetMusaDataType<T>());
}

template <typename T>
Status RunMusaReduceStaged(MusaPreparation& prepare,
                           const ReduceKernel<true>* reduce_kernel,
                           const MusaKernel* kernel,
                           OpKernelContext* ctx,
                           const void* input_buffer,
                           const TensorShape& input_shape,
                           void* output_buffer,
                           const TensorShape& output_shape,
                           const TensorShapeVector& axes,
                           ::musa::dnn::Reduce::Mode reduce_mode) {
  if (input_shape.Size() == 0 || output_shape.Size() == 0) {
    return Status::OK();
  }

  if (axes.size() <= 1) {
    prepare.input_a_ptr = input_buffer;
    prepare.input_a_shape = input_shape;
    prepare.output_ptr = output_buffer;
    prepare.output_size = output_shape.Size();
    prepare.output_shape = output_shape;
    prepare.inputTensors.resize(1);
    prepare.outputTensors.resize(1);
    ORT_RETURN_IF_ERROR(SetupReduceTensorFromBuffer<T>(prepare.inputTensors[0], input_buffer, input_shape));
    ORT_RETURN_IF_ERROR(SetupReduceTensorFromBuffer<T>(prepare.outputTensors[0], output_buffer, output_shape));
    return SimpleMusaReduceOp<T>(prepare, kernel, ctx, axes, reduce_mode, nullptr, true);
  }

  TensorShape current_shape = input_shape;
  const void* current_buffer = input_buffer;
  ::musa::dnn::Tensor current_tensor;
  ORT_RETURN_IF_ERROR(SetupReduceTensorFromBuffer<T>(current_tensor, current_buffer, current_shape));

  TensorShapeVector sorted_axes = axes;
  std::sort(sorted_axes.begin(), sorted_axes.end(), [](int64_t lhs, int64_t rhs) { return lhs > rhs; });

  std::vector<IAllocatorUniquePtr<T>> reduce_buffers;
  reduce_buffers.reserve(sorted_axes.size() > 0 ? sorted_axes.size() - 1 : 0);

  for (size_t i = 0; i < sorted_axes.size(); ++i) {
    const TensorShapeVector single_axis{sorted_axes[i]};
    TensorShape step_output_shape;
    ORT_RETURN_IF_ERROR(
        reduce_kernel->ComputeOutputShape(current_shape, single_axis, reduce_kernel->GetKeepDims(), step_output_shape));

    void* step_output_buffer = nullptr;
    if (i + 1 == sorted_axes.size()) {
      step_output_buffer = output_buffer;
    } else {
      reduce_buffers.emplace_back(kernel->GetScratchBuffer<T>(step_output_shape.Size(), ctx->GetComputeStream()));
      if (step_output_shape.Size() > 0 && !reduce_buffers.back()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to allocate staged Reduce scratch buffer");
      }
      step_output_buffer = reduce_buffers.back().get();
    }

    ::musa::dnn::Tensor step_output_tensor;
    ORT_RETURN_IF_ERROR(SetupReduceTensorFromBuffer<T>(step_output_tensor, step_output_buffer, step_output_shape));

    prepare.input_a_ptr = current_buffer;
    prepare.input_a_shape = current_shape;
    prepare.output_ptr = step_output_buffer;
    prepare.output_size = step_output_shape.Size();
    prepare.output_shape = step_output_shape;
    prepare.inputTensors.resize(1);
    prepare.outputTensors.resize(1);
    prepare.inputTensors[0] = current_tensor;
    prepare.outputTensors[0] = step_output_tensor;

    ORT_RETURN_IF_ERROR(SimpleMusaReduceOp<T>(prepare, kernel, ctx, single_axis, reduce_mode, nullptr, true));

    current_buffer = step_output_buffer;
    current_shape = step_output_shape;
    current_tensor = step_output_tensor;
  }

  return Status::OK();
}

template <typename T>
Status RunReduceSumTyped(const ReduceKernel<true>* reduce_kernel,
                         const MusaKernel* kernel,
                         OpKernelContext* ctx,
                         MusaPreparation& prepare,
                         const Tensor* X,
                         Tensor* Y,
                         const TensorShapeVector& axes) {
  if (X->Shape().Size() == 0) {
    if (Y->Shape().Size() > 0) {
      musaError_t status = musaMemsetAsync(Y->MutableDataRaw(), 0, Y->SizeInBytes(), kernel->Stream(ctx));
      if (status != musaSuccess) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to zero empty ReduceSum output tensor");
      }
    }
    return Status::OK();
  }

  ORT_RETURN_IF_ERROR(RunMusaReduceStaged<T>(prepare, reduce_kernel, kernel, ctx,
                                             X->DataRaw(), X->Shape(),
                                             Y->MutableDataRaw(), Y->Shape(),
                                             axes, ::musa::dnn::Reduce::Mode::MEAN));

  const int64_t divisor = ComputeReductionElementCount(X->Shape(), axes);
  if (divisor <= 1 || Y->Shape().Size() == 0) {
    return Status::OK();
  }

  auto divisor_buffer = kernel->GetScratchBuffer<T>(Y->Shape().Size(), ctx->GetComputeStream());
  if (!divisor_buffer) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to allocate ReduceSum multiplier scratch buffer");
  }

  ::musa::dnn::Tensor divisor_tensor;
  ORT_RETURN_IF_ERROR(SetupReduceTensorFromBuffer<T>(divisor_tensor, divisor_buffer.get(), Y->Shape()));
  ORT_RETURN_IF_ERROR(RunMusaFill<T>(prepare.GetHandle(), divisor_tensor, static_cast<T>(divisor)));
  return RunMusaBinaryOp(prepare.GetHandle(),
                         ::musa::dnn::Binary::Mode::MUL,
                         prepare.outputTensors[0],
                         prepare.outputTensors[0],
                         divisor_tensor);
}

template <>
Status RunReduceSumTyped<MLFloat16>(const ReduceKernel<true>* reduce_kernel,
                                    const MusaKernel* kernel,
                                    OpKernelContext* ctx,
                                    MusaPreparation& prepare,
                                    const Tensor* X,
                                    Tensor* Y,
                                    const TensorShapeVector& axes) {
  if (X->Shape().Size() == 0) {
    if (Y->Shape().Size() > 0) {
      musaError_t status = musaMemsetAsync(Y->MutableDataRaw(), 0, Y->SizeInBytes(), kernel->Stream(ctx));
      if (status != musaSuccess) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to zero empty fp16 ReduceSum output tensor");
      }
    }
    return Status::OK();
  }

  ORT_RETURN_IF_ERROR(RunMusaReduceStaged<MLFloat16>(prepare, reduce_kernel, kernel, ctx,
                                                     X->DataRaw(), X->Shape(),
                                                     Y->MutableDataRaw(), Y->Shape(),
                                                     axes, ::musa::dnn::Reduce::Mode::MEAN));
  const int64_t divisor = ComputeReductionElementCount(X->Shape(), axes);
  if (divisor > 1 && Y->Shape().Size() > 0) {
    auto divisor_buffer = kernel->GetScratchBuffer<MLFloat16>(Y->Shape().Size(), ctx->GetComputeStream());
    if (!divisor_buffer) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to allocate fp16 ReduceSum multiplier scratch buffer");
    }

    ::musa::dnn::Tensor divisor_tensor;
    ORT_RETURN_IF_ERROR(SetupReduceTensorFromBuffer<MLFloat16>(divisor_tensor, divisor_buffer.get(), Y->Shape()));
    ORT_RETURN_IF_ERROR(RunMusaFill<MLFloat16>(prepare.GetHandle(), divisor_tensor, MLFloat16(static_cast<float>(divisor))));
    return RunMusaBinaryOp(prepare.GetHandle(),
                           ::musa::dnn::Binary::Mode::MUL,
                           prepare.outputTensors[0],
                           prepare.outputTensors[0],
                           divisor_tensor);
  }

  return Status::OK();
}

template <typename T>
Status RunReduceMeanTyped(const ReduceKernel<true>* reduce_kernel,
                          const MusaKernel* kernel,
                          OpKernelContext* ctx,
                          MusaPreparation& prepare,
                          const Tensor* X,
                          Tensor* Y,
                          const TensorShapeVector& axes) {
  ORT_RETURN_IF_ERROR(RunReduceSumTyped<T>(reduce_kernel, kernel, ctx, prepare, X, Y, axes));
  if (X->Shape().Size() == 0 || Y->Shape().Size() == 0) {
    return Status::OK();
  }

  const int64_t divisor = ComputeReductionElementCount(X->Shape(), axes);
  if (divisor <= 1) {
    return Status::OK();
  }

  auto divisor_buffer = kernel->GetScratchBuffer<T>(Y->Shape().Size(), ctx->GetComputeStream());
  if (!divisor_buffer) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to allocate ReduceMean divisor scratch buffer");
  }

  ::musa::dnn::Tensor divisor_tensor;
  ORT_RETURN_IF_ERROR(SetupReduceTensorFromBuffer<T>(divisor_tensor, divisor_buffer.get(), Y->Shape()));
  ORT_RETURN_IF_ERROR(RunMusaFill<T>(prepare.GetHandle(), divisor_tensor, static_cast<T>(divisor)));
  return RunMusaBinaryOp(prepare.GetHandle(),
                         ::musa::dnn::Binary::Mode::DIV,
                         prepare.outputTensors[0],
                         prepare.outputTensors[0],
                         divisor_tensor);
}

template <>
Status RunReduceMeanTyped<MLFloat16>(const ReduceKernel<true>* reduce_kernel,
                                     const MusaKernel* kernel,
                                     OpKernelContext* ctx,
                                     MusaPreparation& prepare,
                                     const Tensor* X,
                                     Tensor* Y,
                                     const TensorShapeVector& axes) {
  if (X->Shape().Size() == 0) {
    if (Y->Shape().Size() > 0) {
      musaError_t status = musaMemsetAsync(Y->MutableDataRaw(), 0, Y->SizeInBytes(), kernel->Stream(ctx));
      if (status != musaSuccess) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to zero empty fp16 ReduceMean output tensor");
      }
    }
    return Status::OK();
  }

  auto input_float = kernel->GetScratchBuffer<float>(X->Shape().Size(), ctx->GetComputeStream());
  auto sum_float = kernel->GetScratchBuffer<float>(Y->Shape().Size(), ctx->GetComputeStream());
  auto mean_float = kernel->GetScratchBuffer<float>(Y->Shape().Size(), ctx->GetComputeStream());
  if ((X->Shape().Size() > 0 && !input_float) || (Y->Shape().Size() > 0 && (!sum_float || !mean_float))) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to allocate fp16 ReduceMean float scratch buffers");
  }

  ::musa::dnn::Tensor final_output_tensor = prepare.outputTensors[0];
  ::musa::dnn::Tensor input_float_tensor;
  ORT_RETURN_IF_ERROR(SetupReduceTensorFromBuffer<float>(input_float_tensor, input_float.get(), X->Shape()));
  ORT_RETURN_IF_ERROR(RunMusaUnaryOp(prepare.GetHandle(),
                                     ::musa::dnn::Unary::Mode::CAST,
                                     input_float_tensor,
                                     prepare.inputTensors[0]));

  ORT_RETURN_IF_ERROR(RunMusaReduceStaged<float>(prepare, reduce_kernel, kernel, ctx,
                                                 input_float.get(), X->Shape(),
                                                 sum_float.get(), Y->Shape(),
                                                 axes, ::musa::dnn::Reduce::Mode::ADD));

  const int64_t divisor = ComputeReductionElementCount(X->Shape(), axes);
  ::musa::dnn::Tensor mean_float_tensor;
  if (divisor <= 1) {
    ORT_RETURN_IF_ERROR(SetupReduceTensorFromBuffer<float>(mean_float_tensor, sum_float.get(), Y->Shape()));
  } else {
    auto divisor_buffer = kernel->GetScratchBuffer<float>(Y->Shape().Size(), ctx->GetComputeStream());
    if (!divisor_buffer) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to allocate fp16 ReduceMean divisor scratch buffer");
    }

    ::musa::dnn::Tensor sum_float_tensor;
    ::musa::dnn::Tensor divisor_tensor;
    ORT_RETURN_IF_ERROR(SetupReduceTensorFromBuffer<float>(sum_float_tensor, sum_float.get(), Y->Shape()));
    ORT_RETURN_IF_ERROR(SetupReduceTensorFromBuffer<float>(divisor_tensor, divisor_buffer.get(), Y->Shape()));
    ORT_RETURN_IF_ERROR(SetupReduceTensorFromBuffer<float>(mean_float_tensor, mean_float.get(), Y->Shape()));
    ORT_RETURN_IF_ERROR(RunMusaFill<float>(prepare.GetHandle(), divisor_tensor, static_cast<float>(divisor)));
    ORT_RETURN_IF_ERROR(RunMusaBinaryOp(prepare.GetHandle(),
                                        ::musa::dnn::Binary::Mode::DIV,
                                        mean_float_tensor,
                                        sum_float_tensor,
                                        divisor_tensor));
  }

  return RunMusaUnaryOp(prepare.GetHandle(),
                        ::musa::dnn::Unary::Mode::CAST,
                        final_output_tensor,
                        mean_float_tensor);
}


template <typename T>
Status RunReduceMaxHostFallback(const ReduceKernel<true>* reduce_kernel,
                                const MusaKernel* kernel,
                                OpKernelContext* ctx,
                                const Tensor* X,
                                Tensor* Y,
                                const TensorShapeVector& axes) {
  const int64_t input_size = X->Shape().Size();
  const int64_t output_size = Y->Shape().Size();
  if (input_size == 0 || output_size == 0) {
    return Status::OK();
  }

  auto stream = kernel->Stream(ctx);
  std::vector<T> host_input(static_cast<size_t>(input_size));
  std::vector<T> host_output(static_cast<size_t>(output_size));
  std::vector<uint8_t> initialized(static_cast<size_t>(output_size), 0);

  musaError_t copy_in = musaMemcpyAsync(host_input.data(), X->DataRaw(),
                                        host_input.size() * sizeof(T),
                                        musaMemcpyDeviceToHost, stream);
  if (copy_in != musaSuccess) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "ReduceMax host fallback input copy failed, status: ",
                           static_cast<int>(copy_in));
  }
  musaError_t sync_in = musaStreamSynchronize(stream);
  if (sync_in != musaSuccess) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "ReduceMax host fallback input sync failed, status: ",
                           static_cast<int>(sync_in));
  }

  const auto& input_shape = X->Shape();
  const auto& output_shape = Y->Shape();
  const size_t rank = input_shape.NumDimensions();

  std::vector<uint8_t> reduced_axes(rank, 0);
  for (int64_t axis : axes) {
    reduced_axes[static_cast<size_t>(axis)] = 1;
  }

  std::vector<int64_t> input_strides(rank, 1);
  int64_t input_stride = 1;
  for (int64_t dim = static_cast<int64_t>(rank) - 1; dim >= 0; --dim) {
    input_strides[static_cast<size_t>(dim)] = input_stride;
    input_stride *= input_shape[static_cast<size_t>(dim)];
  }

  std::vector<int64_t> output_strides(output_shape.NumDimensions(), 1);
  int64_t output_stride = 1;
  for (int64_t dim = static_cast<int64_t>(output_shape.NumDimensions()) - 1; dim >= 0; --dim) {
    output_strides[static_cast<size_t>(dim)] = output_stride;
    output_stride *= output_shape[static_cast<size_t>(dim)];
  }

  for (int64_t linear = 0; linear < input_size; ++linear) {
    int64_t output_offset = 0;
    size_t output_dim = 0;

    for (size_t dim = 0; dim < rank; ++dim) {
      const int64_t index = (linear / input_strides[dim]) % input_shape[dim];
      if (reduced_axes[dim]) {
        if (reduce_kernel->GetKeepDims()) {
          ++output_dim;
        }
        continue;
      }

      output_offset += index * output_strides[output_dim++];
    }

    auto out_index = static_cast<size_t>(output_offset);
    const T value = host_input[static_cast<size_t>(linear)];
    if (!initialized[out_index] || static_cast<float>(value) > static_cast<float>(host_output[out_index])) {
      host_output[out_index] = value;
      initialized[out_index] = 1;
    }
  }

  musaError_t copy_out = musaMemcpyAsync(Y->MutableDataRaw(), host_output.data(),
                                         host_output.size() * sizeof(T),
                                         musaMemcpyHostToDevice, stream);
  if (copy_out != musaSuccess) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "ReduceMax host fallback output copy failed, status: ",
                           static_cast<int>(copy_out));
  }
  musaError_t sync_out = musaStreamSynchronize(stream);
  if (sync_out != musaSuccess) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "ReduceMax host fallback output sync failed, status: ",
                           static_cast<int>(sync_out));
  }

  return Status::OK();
}

template <typename T>
Status ReduceMax<T>::ComputeInternal(OpKernelContext* ctx) const {
  // Prepare MUSA operation using generic function
  const auto* ep = static_cast<const MusaExecutionProvider*>(
      Info().GetExecutionProvider());
  MusaPreparation prepare(ep);
  ORT_RETURN_IF_ERROR(PrepareReduceOperation<T>(this, ctx, prepare));

  // Process axes for reduction
  TensorShapeVector processed_axes;
  const Tensor* X = ctx->Input<Tensor>(0);
  ORT_RETURN_IF_ERROR(PrepareAxesForReductionFromContext(this, ctx, X->Shape(), processed_axes));

  // Handle noop_with_empty_axes case
  if (processed_axes.empty() && GetNoopWithEmptyAxes()) {
    // No reduction needed, copy input to output using memcpy
    Tensor* Y = ctx->Output(0, X->Shape());
    if (X->SizeInBytes() > 0) {
      musaStream_t stream = Stream(ctx);
      musaError_t status = musaMemcpyAsync(Y->MutableDataRaw(), X->DataRaw(),
                                           X->SizeInBytes(), musaMemcpyDeviceToDevice, stream);
      if (status != musaSuccess) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to copy tensor data");
      }
    }
    return Status::OK();
  }

  Tensor* Y = ctx->Output(0, prepare.output_shape);
  ORT_RETURN_IF_NOT(Y != nullptr, "ReduceMax output tensor is null");

  // Correctness-first path: current muDNN ReduceMax returns wrong values for model3-style
  // rank4 axis reductions, so compute on host and copy the result back to MUSA memory.
  ORT_RETURN_IF_ERROR(RunReduceMaxHostFallback<T>(this, this, ctx, X, Y, processed_axes));

  return Status::OK();
}

template <typename T>
Status ReduceMin<T>::ComputeInternal(OpKernelContext* ctx) const {
  // Prepare MUSA operation using generic function
  const auto* ep = static_cast<const MusaExecutionProvider*>(
      Info().GetExecutionProvider());
  MusaPreparation prepare(ep);
  ORT_RETURN_IF_ERROR(PrepareReduceOperation<T>(this, ctx, prepare));

  // Process axes for reduction
  TensorShapeVector processed_axes;
  const Tensor* X = ctx->Input<Tensor>(0);
  ORT_RETURN_IF_ERROR(PrepareAxesForReductionFromContext(this, ctx, X->Shape(), processed_axes));

  // Handle noop_with_empty_axes case
  if (processed_axes.empty() && GetNoopWithEmptyAxes()) {
    // No reduction needed, copy input to output using memcpy
    Tensor* Y = ctx->Output(0, X->Shape());
    if (X->SizeInBytes() > 0) {
      musaStream_t stream = Stream(ctx);
      musaError_t status = musaMemcpyAsync(Y->MutableDataRaw(), X->DataRaw(),
                                           X->SizeInBytes(), musaMemcpyDeviceToDevice, stream);
      if (status != musaSuccess) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to copy tensor data");
      }
    }
    return Status::OK();
  }

  // Call MUSA device ReduceMin operation using prepared data
  ORT_RETURN_IF_ERROR(SimpleMusaReduceOp<T>(prepare, this, ctx, processed_axes, ::musa::dnn::Reduce::Mode::MIN));

  return Status::OK();
}

template <typename T>
Status ReduceSum<T>::ComputeInternal(OpKernelContext* ctx) const {
  const auto* ep = static_cast<const MusaExecutionProvider*>(
      Info().GetExecutionProvider());
  MusaPreparation prepare(ep);
  ORT_RETURN_IF_ERROR(PrepareReduceOperation<T>(this, ctx, prepare));

  TensorShapeVector processed_axes;
  const Tensor* X = ctx->Input<Tensor>(0);
  ORT_RETURN_IF_ERROR(PrepareAxesForReductionFromContext(this, ctx, X->Shape(), processed_axes));

  if (processed_axes.empty() && GetNoopWithEmptyAxes()) {
    Tensor* Y = ctx->Output(0, X->Shape());
    return CopyReduceInputToOutput<T>(X, Y, this, ctx);
  }

  Tensor* Y = ctx->Output(0, prepare.output_shape);
  return RunReduceSumTyped<T>(this, this, ctx, prepare, X, Y, processed_axes);
}

template <typename T>
Status ReduceMean<T>::ComputeInternal(OpKernelContext* ctx) const {
  const auto* ep = static_cast<const MusaExecutionProvider*>(
      Info().GetExecutionProvider());
  MusaPreparation prepare(ep);
  ORT_RETURN_IF_ERROR(PrepareReduceOperation<T>(this, ctx, prepare));

  TensorShapeVector processed_axes;
  const Tensor* X = ctx->Input<Tensor>(0);
  ORT_RETURN_IF_ERROR(PrepareAxesForReductionFromContext(this, ctx, X->Shape(), processed_axes));

  if (processed_axes.empty() && GetNoopWithEmptyAxes()) {
    Tensor* Y = ctx->Output(0, X->Shape());
    return CopyReduceInputToOutput<T>(X, Y, this, ctx);
  }

  Tensor* Y = ctx->Output(0, prepare.output_shape);
  return RunReduceMeanTyped<T>(this, this, ctx, prepare, X, Y, processed_axes);
}

template <typename T>
Status ReduceProd<T>::ComputeInternal(OpKernelContext* ctx) const {
  // Prepare MUSA operation using generic function
  const auto* ep = static_cast<const MusaExecutionProvider*>(
      Info().GetExecutionProvider());
  MusaPreparation prepare(ep);
  ORT_RETURN_IF_ERROR(PrepareReduceOperation<T>(this, ctx, prepare));

  // Process axes for reduction
  TensorShapeVector processed_axes;
  const Tensor* X = ctx->Input<Tensor>(0);
  ORT_RETURN_IF_ERROR(PrepareAxesForReductionFromContext(this, ctx, X->Shape(), processed_axes));

  // Handle noop_with_empty_axes case
  if (processed_axes.empty() && GetNoopWithEmptyAxes()) {
    // No reduction needed, copy input to output using memcpy
    Tensor* Y = ctx->Output(0, X->Shape());
    if (X->SizeInBytes() > 0) {
      musaStream_t stream = Stream(ctx);
      musaError_t status = musaMemcpyAsync(Y->MutableDataRaw(), X->DataRaw(),
                                           X->SizeInBytes(), musaMemcpyDeviceToDevice, stream);
      if (status != musaSuccess) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to copy tensor data");
      }
    }
    return Status::OK();
  }

  if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, float>) {
    ReduceProdInt32Params params{};
    ORT_RETURN_IF_ERROR(PrepareReduceProdInt32Params(X->Shape(),
                                                     prepare.output_shape,
                                                     processed_axes,
                                                     GetKeepDims(),
                                                     params));

    musaError_t status = musaSuccess;
    if constexpr (std::is_same_v<T, int32_t>) {
      status = LaunchFillInt32Kernel(Stream(ctx),
                                     static_cast<int32_t*>(prepare.output_ptr),
                                     params.output_size,
                                     1);
    } else {
      status = LaunchFillFloatKernel(Stream(ctx),
                                     static_cast<float*>(prepare.output_ptr),
                                     params.output_size,
                                     1.0f);
    }

    if (status != musaSuccess) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "ReduceProd fill kernel failed, status: ",
                             static_cast<int>(status));
    }

    if constexpr (std::is_same_v<T, int32_t>) {
      status = LaunchReduceProdInt32Kernel(Stream(ctx),
                                           static_cast<const int32_t*>(prepare.input_a_ptr),
                                           static_cast<int32_t*>(prepare.output_ptr),
                                           params);
    } else {
      status = LaunchReduceProdFloatKernel(Stream(ctx),
                                           static_cast<const float*>(prepare.input_a_ptr),
                                           static_cast<float*>(prepare.output_ptr),
                                           params);
    }

    if (status != musaSuccess) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "ReduceProd kernel failed, status: ",
                             static_cast<int>(status));
    }
    return Status::OK();
  }

  // Call MUSA device ReduceProd operation using prepared data
  ORT_RETURN_IF_ERROR(SimpleMusaReduceOp<T>(prepare, this, ctx, processed_axes, ::musa::dnn::Reduce::Mode::PROD));

  return Status::OK();
}

template <typename T>
Status ReduceL2<T>::ComputeInternal(OpKernelContext* ctx) const {
  const auto* ep = static_cast<const MusaExecutionProvider*>(
      Info().GetExecutionProvider());
  MusaPreparation prepare(ep);
  ORT_RETURN_IF_ERROR(PrepareReduceOperation<T>(this, ctx, prepare));

  TensorShapeVector processed_axes;
  const Tensor* X = ctx->Input<Tensor>(0);
  ORT_RETURN_IF_ERROR(PrepareAxesForReductionFromContext(this, ctx, X->Shape(), processed_axes));

  if (processed_axes.empty() && GetNoopWithEmptyAxes()) {
    Tensor* Y = ctx->Output(0, X->Shape());
    if (X->SizeInBytes() > 0) {
      musaStream_t stream = Stream(ctx);
      musaError_t status = musaMemcpyAsync(Y->MutableDataRaw(), X->DataRaw(),
                                           X->SizeInBytes(), musaMemcpyDeviceToDevice, stream);
      if (status != musaSuccess) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to copy tensor data");
      }
    }
    return Status::OK();
  }

  if (X->Shape().Size() == 0) {
    if (prepare.output_size > 0) {
      musaError_t status = musaMemsetAsync(prepare.output_ptr, 0, prepare.output_size * sizeof(T), Stream(ctx));
      if (status != musaSuccess) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to zero ReduceL2 output tensor");
      }
    }
    return Status::OK();
  }

  if (auto stream = Stream(ctx)) {
    auto status = prepare.GetHandle().SetStream(stream);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set ReduceL2 MUSA stream, status: " +
                                 std::to_string(static_cast<int>(status)));
    }
  }

  if (X->Shape().NumDimensions() <= static_cast<size_t>(kReduceL2KeepDimsMaxRank)) {
    ReduceL2KeepDimsParams l2_params{};
    ORT_RETURN_IF_ERROR(PrepareReduceL2Params(X->Shape(),
                                              prepare.output_shape,
                                              processed_axes,
                                              GetKeepDims(),
                                              l2_params));

    musaError_t status = musaSuccess;
    if constexpr (std::is_same<T, MLFloat16>::value) {
      status = LaunchReduceL2KeepDimsHalf(Stream(ctx), X->DataRaw(), prepare.output_ptr, l2_params);
    } else {
      status = LaunchReduceL2KeepDimsFloat(Stream(ctx),
                                           static_cast<const float*>(X->DataRaw()),
                                           static_cast<float*>(prepare.output_ptr),
                                           l2_params);
    }

    if (status != musaSuccess) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "ReduceL2 MUSA kernel launch failed, status: ",
                             static_cast<int>(status));
    }
    return Status::OK();
  }

  const auto musa_type = GetMusaDataType<T>();
  auto squared_buffer = GetScratchBuffer<T>(X->Shape().Size(), ctx->GetComputeStream());
  if (!squared_buffer) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to allocate ReduceL2 squared scratch buffer");
  }

  ::musa::dnn::Tensor squared_tensor;
  ORT_RETURN_IF_ERROR(SetupMusaTensorFromBuffer(squared_tensor,
                                                squared_buffer.get(),
                                                X->Shape(),
                                                musa_type));

  ORT_RETURN_IF_ERROR(RunMusaBinaryOp(prepare.GetHandle(),
                                      ::musa::dnn::Binary::Mode::MUL,
                                      squared_tensor,
                                      prepare.inputTensors[0],
                                      prepare.inputTensors[0]));

  ::musa::dnn::Tensor final_output_tensor = prepare.outputTensors[0];
  prepare.input_a_ptr = squared_buffer.get();
  prepare.input_a_shape = X->Shape();
  prepare.inputTensors[0] = squared_tensor;
  ::musa::dnn::Tensor reduced_sum_tensor;
  std::vector<IAllocatorUniquePtr<T>> reduce_buffers;
  IAllocatorUniquePtr<T> reduced_buffer;
  const bool full_reduce_to_scalar = !GetKeepDims() &&
                                     processed_axes.size() == X->Shape().NumDimensions();

  if (full_reduce_to_scalar) {
    const TensorShape flattened_shape(TensorShapeVector{static_cast<int64_t>(X->Shape().Size())});
    ::musa::dnn::Tensor flattened_tensor;
    ORT_RETURN_IF_ERROR(SetupMusaTensorFromBuffer(flattened_tensor,
                                                  squared_buffer.get(),
                                                  flattened_shape,
                                                  musa_type));

    reduce_buffers.emplace_back(GetScratchBuffer<T>(prepare.output_size, ctx->GetComputeStream()));
    if (prepare.output_size > 0 && !reduce_buffers.back()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to allocate ReduceL2 full-reduce scratch buffer");
    }

    ::musa::dnn::Tensor scalar_sum_tensor;
    ORT_RETURN_IF_ERROR(SetupMusaTensorFromBuffer(scalar_sum_tensor,
                                                  reduce_buffers.back().get(),
                                                  prepare.output_shape,
                                                  musa_type));

    prepare.input_a_ptr = squared_buffer.get();
    prepare.input_a_shape = flattened_shape;
    prepare.inputTensors[0] = flattened_tensor;
    prepare.output_ptr = reduce_buffers.back().get();
    prepare.output_size = prepare.output_shape.Size();
    prepare.outputTensors[0] = scalar_sum_tensor;

    ORT_RETURN_IF_ERROR(SimpleMusaReduceOp<T>(prepare, this, ctx, TensorShapeVector{0},
                                              ::musa::dnn::Reduce::Mode::ADD));

    reduced_sum_tensor = scalar_sum_tensor;
  } else if (processed_axes.size() > 1) {
    TensorShape current_shape = X->Shape();
    ::musa::dnn::Tensor current_tensor = squared_tensor;
    const void* current_buffer = squared_buffer.get();

    TensorShapeVector sorted_axes = processed_axes;
    std::sort(sorted_axes.begin(), sorted_axes.end(),
              [](int64_t lhs, int64_t rhs) { return lhs > rhs; });

    reduce_buffers.reserve(sorted_axes.size());

    for (size_t i = 0; i < sorted_axes.size(); ++i) {
      const TensorShapeVector single_axis{sorted_axes[i]};
      TensorShape step_output_shape;
      ORT_RETURN_IF_ERROR(ComputeOutputShape(current_shape, single_axis, GetKeepDims(), step_output_shape));

      if (current_shape[static_cast<size_t>(sorted_axes[i])] == 1) {
        ::musa::dnn::Tensor reshaped_tensor;
        ORT_RETURN_IF_ERROR(SetupMusaTensorFromBuffer(reshaped_tensor,
                                                      current_buffer,
                                                      step_output_shape,
                                                      musa_type));
        current_shape = step_output_shape;
        current_tensor = reshaped_tensor;
        continue;
      }

      reduce_buffers.emplace_back(GetScratchBuffer<T>(step_output_shape.Size(), ctx->GetComputeStream()));
      if (step_output_shape.Size() > 0 && !reduce_buffers.back()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to allocate ReduceL2 sequential reduce scratch buffer");
      }

      ::musa::dnn::Tensor step_output_tensor;
      ORT_RETURN_IF_ERROR(SetupMusaTensorFromBuffer(step_output_tensor,
                                                    reduce_buffers.back().get(),
                                                    step_output_shape,
                                                    musa_type));

      prepare.input_a_ptr = current_buffer;
      prepare.input_a_shape = current_shape;
      prepare.inputTensors[0] = current_tensor;
      prepare.output_ptr = reduce_buffers.back().get();
      prepare.output_size = step_output_shape.Size();
      prepare.output_shape = step_output_shape;
      prepare.outputTensors[0] = step_output_tensor;

      ORT_RETURN_IF_ERROR(SimpleMusaReduceOp<T>(prepare, this, ctx, single_axis,
                                                ::musa::dnn::Reduce::Mode::ADD));

      current_buffer = reduce_buffers.back().get();
      current_shape = step_output_shape;
      current_tensor = step_output_tensor;
    }

    reduced_sum_tensor = current_tensor;
  } else {
    ORT_RETURN_IF_ERROR(SimpleMusaReduceOp<T>(prepare, this, ctx, processed_axes,
                                              ::musa::dnn::Reduce::Mode::ADD));

    reduced_buffer = GetScratchBuffer<T>(prepare.output_size, ctx->GetComputeStream());
    if (prepare.output_size > 0 && !reduced_buffer) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to allocate ReduceL2 reduced scratch buffer");
    }

    musaError_t copy_status = musaMemcpyAsync(reduced_buffer.get(),
                                              prepare.output_ptr,
                                              prepare.output_size * sizeof(T),
                                              musaMemcpyDeviceToDevice,
                                              Stream(ctx));
    if (copy_status != musaSuccess) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to copy ReduceL2 reduce result to scratch buffer");
    }

    ORT_RETURN_IF_ERROR(SetupMusaTensorFromBuffer(reduced_sum_tensor,
                                                  reduced_buffer.get(),
                                                  prepare.output_shape,
                                                  musa_type));
  }

  ORT_RETURN_IF_ERROR(RunMusaUnaryOp(prepare.GetHandle(),
                                     ::musa::dnn::Unary::Mode::SQRT,
                                     final_output_tensor,
                                     reduced_sum_tensor));

  return Status::OK();
}

// MUSA device-based ArgMax/ArgMin implementation using MusaPreparation and mudnn library
template <typename T>
Status SimpleMusaArgMaxMinOp(const MusaPreparation& prepare,
                             const MusaKernel* kernel,
                             OpKernelContext* ctx,
                             const TensorShapeVector& axes,
                             ::musa::dnn::Reduce::Mode reduce_mode,
                             bool /* select_last_index */) {
  // Handle empty tensors (zero elements) - no computation needed
  if (prepare.input_a_shape.Size() == 0 || prepare.output_size == 0) {
    return Status::OK();
  }

  // Get tensor data from prepared MUSA tensors
  const auto* input_data = reinterpret_cast<const T*>(prepare.input_a_ptr);
  auto* output_data = reinterpret_cast<int64_t*>(prepare.output_ptr);

  // Validate prepared tensors
  if (input_data == nullptr || output_data == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid prepared tensor pointers");
  }

  if (prepare.inputTensors.empty() || prepare.outputTensors.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid prepared tensor configuration");
  }

  // Use mudnn Reduce class for device computation with RunIndices
  try {
    // Create mudnn Reduce operation
    ::musa::dnn::Reduce reduce_op;

    // Set the operation mode (MAX or MIN)
    auto status = reduce_op.SetMode(reduce_mode);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set mudnn Reduce mode, status: " +
                             std::to_string(static_cast<int>(status)));
    }

    // Convert axes to int array for mudnn
    std::vector<int> int_axes;
    for (auto axis : axes) {
      int_axes.push_back(static_cast<int>(axis));
    }

    // Check muDNN limitation: RunIndices only supports single axis reduction
    if (int_axes.size() != 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "muDNN Reduce::RunIndices only supports single axis reduction (ndim == 1). "
                             "Current request has ", int_axes.size(), " axes. "
                             "This is a limitation of the current muDNN implementation. "
                             "Please report this to MUSA engineering team for multi-axis ArgMax/ArgMin support.");
    }

    // Set reduction dimensions
    status = reduce_op.SetDim(static_cast<int>(int_axes.size()), int_axes.data());
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set mudnn Reduce dimensions");
    }

    // Query workspace size for RunIndices
    size_t workspace_size = 0;
    status = reduce_op.GetWorkspaceSize(prepare.GetHandle(), workspace_size,
                                       const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]),
                                       prepare.inputTensors[0]);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to get mudnn Reduce workspace size");
    }

    std::vector<IAllocatorUniquePtr<void>> workspace_buffers;
    auto memory_allocator = [&](size_t size) -> ::musa::dnn::MemoryHandler {
      if (size == 0) {
        return {nullptr, [](void*) {}};
      }

      workspace_buffers.emplace_back(kernel->GetScratchBuffer<void>(size, ctx->GetComputeStream()));
      return {workspace_buffers.back().get(), [](void*) {}};
    };

    // Run the reduce operation with RunIndices to get indices
    status = reduce_op.RunIndices(prepare.GetHandle(),
                                 const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]),  // output indices tensor
                                 prepare.inputTensors[0],   // input tensor
                                 memory_allocator);

    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "mudnn Reduce RunIndices operation failed, status: " +
                             std::to_string(static_cast<int>(status)));
    }


  } catch (const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Exception in mudnn Reduce RunIndices operation: " +
                           std::string(e.what()));
  }

  return Status::OK();
}

// Specialized prepare function for ArgMax/ArgMin operations
template <typename T>
Status PrepareArgMaxMinOperation(const ReduceKernel<false>* kernel,
                                 OpKernelContext* ctx,
                                 MusaPreparation& prepare) {
  // 1. Get input tensor
  const Tensor* X = ctx->Input<Tensor>(0);
  if (!X) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid input tensor");
  }

  // 2. Process axes for reduction
  TensorShapeVector processed_axes;
  ORT_RETURN_IF_ERROR(kernel->PrepareAxesForReduction(X->Shape(), processed_axes));

  // 3. Handle noop_with_empty_axes case
  if (processed_axes.empty() && kernel->GetNoopWithEmptyAxes()) {
    // No reduction needed, just copy input to output
    // For ArgMax/ArgMin, this means returning indices that match the input shape
    Tensor* Y = ctx->Output(0, X->Shape());
    if (!Y) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor");
    }

    // For noop case, we still need to set up pointers
    prepare.input_a_ptr = X->DataRaw();
    prepare.output_ptr = Y->MutableDataRaw();
    prepare.output_size = Y->Shape().Size();
    prepare.input_a_shape = X->Shape();
    prepare.output_shape = Y->Shape();
    return Status::OK();
  }

  // 4. Compute output shape
  TensorShape output_shape;
  ORT_RETURN_IF_ERROR(kernel->ComputeOutputShape(X->Shape(), processed_axes, kernel->GetKeepDims(), output_shape));

  // 5. Create output tensor with int64 type (indices)
  Tensor* Y = ctx->Output(0, output_shape);
  if (!Y) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor");
  }

  // 6. Store tensor pointers and shapes in preparation
  prepare.input_a_ptr = X->DataRaw();
  prepare.output_ptr = Y->MutableDataRaw();
  prepare.output_size = Y->Shape().Size();
  prepare.input_a_shape = X->Shape();
  prepare.output_shape = Y->Shape();

  // For tensors with zero elements, data pointers might be null, which is valid
  if ((prepare.input_a_shape.Size() > 0 && prepare.input_a_ptr == nullptr) ||
      (prepare.output_size > 0 && prepare.output_ptr == nullptr)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid tensor data pointers");
  }

  // 7. Get MUSA data types - input uses T, output uses int64
  const auto input_musa_type = GetMusaDataType<T>();
  const auto output_musa_type = GetMusaDataType<int64_t>();

  // 8. Prepare MUSA tensors - using ORT_TRY/ORT_CATCH like other MusaEP operations
  ORT_TRY {
    // Set up MUSA stream for asynchronous execution
    auto stream = kernel->Stream(ctx);
    if (prepare.handle) {
      if (stream) {
        auto status = prepare.handle->SetStream(stream);
        if (status != ::musa::dnn::Status::SUCCESS) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                 "Failed to set MUSA stream, status: " +
                                     std::to_string(static_cast<int>(status)));
        }
      } else {
        // Use default stream for backward compatibility
        LOGS_DEFAULT(WARNING) << "No stream provided, using default MUSA stream";
      }
    }

    // Initialize tensors vectors
    prepare.inputTensors.resize(1);
    prepare.outputTensors.resize(1);

    // Setup input tensor (using input data type T)
    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.inputTensors[0], X, input_musa_type, &prepare));

    // Setup output tensor (using int64 for indices)
    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.outputTensors[0], Y, output_musa_type, &prepare));
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  return Status::OK();
}

template <typename T>
Status ArgMax<T>::ComputeInternal(OpKernelContext* ctx) const {
  // Check for unsupported select_last_index feature
  if (select_last_index_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "ArgMax with select_last_index=1 is not supported on MUSA EP. "
                           "muDNN's Reduce::RunIndices API does not support selecting the last index "
                           "when duplicate maximum values are encountered. "
                           "This is a limitation of the current muDNN implementation. "
                           "Please consider using CPU EP for this specific case or "
                           "report this limitation to MUSA engineering team for future support.");
  }

  // Prepare MUSA operation using specialized function for ArgMax/ArgMin
  const auto* ep = static_cast<const MusaExecutionProvider*>(
      Info().GetExecutionProvider());
  MusaPreparation prepare(ep);
  ORT_RETURN_IF_ERROR(PrepareArgMaxMinOperation<T>(this, ctx, prepare));

  // Process axes for reduction
  TensorShapeVector processed_axes;
  const Tensor* X = ctx->Input<Tensor>(0);
  ORT_RETURN_IF_ERROR(PrepareAxesForReductionFromContext(this, ctx, X->Shape(), processed_axes));

  // Handle noop_with_empty_axes case
  if (processed_axes.empty() && GetNoopWithEmptyAxes()) {
        // For ArgMax noop case, initialize output indices to sequential values
    Tensor* Y = ctx->Output(0, X->Shape());
    auto* output_data = reinterpret_cast<int64_t*>(Y->MutableDataRaw());
    size_t total_elements = Y->Shape().Size();

    // Initialize indices from 0 to total_elements-1
    for (size_t i = 0; i < total_elements; ++i) {
      output_data[i] = static_cast<int64_t>(i);
    }
    return Status::OK();
  }

  // Call MUSA device ArgMax operation using prepared data
  ORT_RETURN_IF_ERROR(SimpleMusaArgMaxMinOp<T>(prepare, this, ctx, processed_axes,
                                               ::musa::dnn::Reduce::Mode::MAX,
                                               select_last_index_));

  return Status::OK();
}

template <typename T>
Status ArgMin<T>::ComputeInternal(OpKernelContext* ctx) const {
  // Check for unsupported select_last_index feature
  if (select_last_index_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "ArgMin with select_last_index=1 is not supported on MUSA EP. "
                           "muDNN's Reduce::RunIndices API does not support selecting the last index "
                           "when duplicate minimum values are encountered. "
                           "This is a limitation of the current muDNN implementation. "
                           "Please consider using CPU EP for this specific case or "
                           "report this limitation to MUSA engineering team for future support.");
  }

  // Prepare MUSA operation using specialized function for ArgMax/ArgMin
  const auto* ep = static_cast<const MusaExecutionProvider*>(
      Info().GetExecutionProvider());
  MusaPreparation prepare(ep);
  ORT_RETURN_IF_ERROR(PrepareArgMaxMinOperation<T>(this, ctx, prepare));

  // Process axes for reduction
  TensorShapeVector processed_axes;
  const Tensor* X = ctx->Input<Tensor>(0);
  ORT_RETURN_IF_ERROR(PrepareAxesForReductionFromContext(this, ctx, X->Shape(), processed_axes));

  // Handle noop_with_empty_axes case
  if (processed_axes.empty() && GetNoopWithEmptyAxes()) {
        // For ArgMin noop case, initialize output indices to sequential values
    Tensor* Y = ctx->Output(0, X->Shape());
    auto* output_data = reinterpret_cast<int64_t*>(Y->MutableDataRaw());
    size_t total_elements = Y->Shape().Size();

    // Initialize indices from 0 to total_elements-1
    for (size_t i = 0; i < total_elements; ++i) {
      output_data[i] = static_cast<int64_t>(i);
    }
    return Status::OK();
  }

  // Call MUSA device ArgMin operation using prepared data
  ORT_RETURN_IF_ERROR(SimpleMusaArgMaxMinOp<T>(prepare, this, ctx, processed_axes,
                                               ::musa::dnn::Reduce::Mode::MIN,
                                               select_last_index_));

  return Status::OK();
}

// Macro for registering typed compute function with MUSA implementation
#define REGISTER_MUSA_REDUCE_TYPED_COMPUTE(x, T)                               \
  template Status x<T>::ComputeInternal(OpKernelContext* ctx) const;

// Macro for registering typed kernel
#define REGISTER_MUSA_REDUCE_TYPED_KERNEL_AXES_INPUT(x, ver, T)                \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                               \
      x, kOnnxDomain, ver, T, kMusaExecutionProvider,                          \
      (*KernelDefBuilder::Create())                                            \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())               \
          .InputMemoryType(OrtMemTypeCPUInput, 1),                             \
      x<T>);

#define REGISTER_MUSA_REDUCE_TYPED_KERNEL_NO_CPU_INPUT(x, ver, T)              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                               \
      x, kOnnxDomain, ver, T, kMusaExecutionProvider,                          \
      (*KernelDefBuilder::Create())                                            \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),              \
      x<T>);

// Macro for registering versioned typed kernel
#define REGISTER_MUSA_REDUCE_VERSIONED_TYPED_KERNEL_NO_CPU_INPUT(x, startver, endver, T) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                     \
      x, kOnnxDomain, startver, endver, T, kMusaExecutionProvider,             \
      (*KernelDefBuilder::Create())                                            \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),              \
      x<T>);

// Combined macro for both kernel and compute registration
#define REGISTER_MUSA_REDUCE_TYPED_AXES_INPUT(name, ver, T)                   \
  REGISTER_MUSA_REDUCE_TYPED_KERNEL_AXES_INPUT(name, ver, T)                  \
  REGISTER_MUSA_REDUCE_TYPED_COMPUTE(name, T)

#define REGISTER_MUSA_REDUCE_TYPED_NO_CPU_INPUT(name, ver, T)                 \
  REGISTER_MUSA_REDUCE_TYPED_KERNEL_NO_CPU_INPUT(name, ver, T)                \
  REGISTER_MUSA_REDUCE_TYPED_COMPUTE(name, T)

// Register operations for different types
#define REGISTER_MUSA_REDUCE_HFD_AXES_INPUT(name, ver)                        \
  REGISTER_MUSA_REDUCE_TYPED_AXES_INPUT(name, ver, MLFloat16)                 \
  REGISTER_MUSA_REDUCE_TYPED_AXES_INPUT(name, ver, float)

#define REGISTER_MUSA_REDUCE_HFD_NO_CPU_INPUT(name, ver)                      \
  REGISTER_MUSA_REDUCE_TYPED_NO_CPU_INPUT(name, ver, MLFloat16)               \
  REGISTER_MUSA_REDUCE_TYPED_NO_CPU_INPUT(name, ver, float)

// Register versioned operations (kernel only, for HFD types)
#define REGISTER_MUSA_REDUCE_VERSIONED_HFD_NO_CPU_INPUT(name, startver, endver)   \
  REGISTER_MUSA_REDUCE_VERSIONED_TYPED_KERNEL_NO_CPU_INPUT(name, startver, endver, MLFloat16) \
  REGISTER_MUSA_REDUCE_VERSIONED_TYPED_KERNEL_NO_CPU_INPUT(name, startver, endver, float)

// Register ReduceMax operations following ONNX operator versions
REGISTER_MUSA_REDUCE_VERSIONED_HFD_NO_CPU_INPUT(ReduceMax, 1, 17)
REGISTER_MUSA_REDUCE_HFD_AXES_INPUT(ReduceMax, 18)

// Register ReduceMin operations following ONNX operator versions
REGISTER_MUSA_REDUCE_VERSIONED_HFD_NO_CPU_INPUT(ReduceMin, 1, 17)
REGISTER_MUSA_REDUCE_HFD_AXES_INPUT(ReduceMin, 18)

// Register ReduceSum operations following ONNX operator versions
REGISTER_MUSA_REDUCE_VERSIONED_HFD_NO_CPU_INPUT(ReduceSum, 1, 12)
REGISTER_MUSA_REDUCE_HFD_AXES_INPUT(ReduceSum, 13)

// Register ReduceMean operations following ONNX operator versions
REGISTER_MUSA_REDUCE_VERSIONED_HFD_NO_CPU_INPUT(ReduceMean, 1, 12)
REGISTER_MUSA_REDUCE_HFD_NO_CPU_INPUT(ReduceMean, 13)

// Register ReduceProd operations following ONNX operator versions
REGISTER_MUSA_REDUCE_VERSIONED_HFD_NO_CPU_INPUT(ReduceProd, 1, 17)
REGISTER_MUSA_REDUCE_HFD_AXES_INPUT(ReduceProd, 18)
REGISTER_MUSA_REDUCE_VERSIONED_TYPED_KERNEL_NO_CPU_INPUT(ReduceProd, 1, 17, int32_t)
REGISTER_MUSA_REDUCE_VERSIONED_TYPED_KERNEL_NO_CPU_INPUT(ReduceProd, 1, 17, int64_t)
REGISTER_MUSA_REDUCE_TYPED_KERNEL_AXES_INPUT(ReduceProd, 18, int32_t)
REGISTER_MUSA_REDUCE_TYPED_KERNEL_AXES_INPUT(ReduceProd, 18, int64_t)
REGISTER_MUSA_REDUCE_TYPED_COMPUTE(ReduceProd, int32_t)
REGISTER_MUSA_REDUCE_TYPED_COMPUTE(ReduceProd, int64_t)

// Register ReduceL2 operations following ONNX operator versions
REGISTER_MUSA_REDUCE_VERSIONED_HFD_NO_CPU_INPUT(ReduceL2, 1, 17)
REGISTER_MUSA_REDUCE_HFD_AXES_INPUT(ReduceL2, 18)

// Macro for registering ArgMax/ArgMin typed kernel (output is always int64)
#define REGISTER_MUSA_ARGMAXMIN_TYPED_KERNEL(x, ver, T)                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                               \
      x, kOnnxDomain, ver, T, kMusaExecutionProvider,                          \
      (*KernelDefBuilder::Create())                                            \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),              \
      x<T>);

// Macro for registering versioned ArgMax/ArgMin typed kernel
#define REGISTER_MUSA_ARGMAXMIN_VERSIONED_TYPED_KERNEL(x, startver, endver, T) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                     \
      x, kOnnxDomain, startver, endver, T, kMusaExecutionProvider,             \
      (*KernelDefBuilder::Create())                                            \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),              \
      x<T>);

// Macro for registering ArgMax/ArgMin typed compute function
#define REGISTER_MUSA_ARGMAXMIN_TYPED_COMPUTE(x, T)                           \
  template Status x<T>::ComputeInternal(OpKernelContext* ctx) const;

// Combined macro for both kernel and compute registration for ArgMax/ArgMin
#define REGISTER_MUSA_ARGMAXMIN_TYPED(name, ver, T)                           \
  REGISTER_MUSA_ARGMAXMIN_TYPED_KERNEL(name, ver, T)                          \
  REGISTER_MUSA_ARGMAXMIN_TYPED_COMPUTE(name, T)

// Register ArgMax/ArgMin operations for HFD types (MLFloat16, float)
#define REGISTER_MUSA_ARGMAXMIN_HFD(name, ver)                               \
  REGISTER_MUSA_ARGMAXMIN_TYPED(name, ver, MLFloat16)                         \
  REGISTER_MUSA_ARGMAXMIN_TYPED(name, ver, float)

// Register ArgMax/ArgMin operations for integer types
#define REGISTER_MUSA_ARGMAXMIN_INT(name, ver)                               \
  REGISTER_MUSA_ARGMAXMIN_TYPED(name, ver, int32_t)                           \
  REGISTER_MUSA_ARGMAXMIN_TYPED(name, ver, int64_t)                           \
  REGISTER_MUSA_ARGMAXMIN_TYPED(name, ver, int8_t)                            \
  REGISTER_MUSA_ARGMAXMIN_TYPED(name, ver, uint8_t)

// Register ArgMax/ArgMin operations for all supported types
#define REGISTER_MUSA_ARGMAXMIN_ALL(name, ver)                               \
  REGISTER_MUSA_ARGMAXMIN_HFD(name, ver)                                     \
  REGISTER_MUSA_ARGMAXMIN_INT(name, ver)

// Register versioned ArgMax/ArgMin operations (kernel only, for HFD types)
#define REGISTER_MUSA_ARGMAXMIN_VERSIONED_HFD(name, startver, endver)        \
  REGISTER_MUSA_ARGMAXMIN_VERSIONED_TYPED_KERNEL(name, startver, endver, MLFloat16) \
  REGISTER_MUSA_ARGMAXMIN_VERSIONED_TYPED_KERNEL(name, startver, endver, float)

// Register versioned ArgMax/ArgMin operations (kernel only, for integer types)
#define REGISTER_MUSA_ARGMAXMIN_VERSIONED_INT(name, startver, endver)        \
  REGISTER_MUSA_ARGMAXMIN_VERSIONED_TYPED_KERNEL(name, startver, endver, int32_t) \
  REGISTER_MUSA_ARGMAXMIN_VERSIONED_TYPED_KERNEL(name, startver, endver, int64_t) \
  REGISTER_MUSA_ARGMAXMIN_VERSIONED_TYPED_KERNEL(name, startver, endver, int8_t)  \
  REGISTER_MUSA_ARGMAXMIN_VERSIONED_TYPED_KERNEL(name, startver, endver, uint8_t)

// Register versioned ArgMax/ArgMin operations (kernel only, for all types)
#define REGISTER_MUSA_ARGMAXMIN_VERSIONED_ALL(name, startver, endver)        \
  REGISTER_MUSA_ARGMAXMIN_VERSIONED_HFD(name, startver, endver)              \
  REGISTER_MUSA_ARGMAXMIN_VERSIONED_INT(name, startver, endver)

// Register ArgMax operations following ONNX operator versions
REGISTER_MUSA_ARGMAXMIN_VERSIONED_ALL(ArgMax, 1, 10)
REGISTER_MUSA_ARGMAXMIN_VERSIONED_ALL(ArgMax, 11, 12)
REGISTER_MUSA_ARGMAXMIN_ALL(ArgMax, 13)

// Register ArgMin operations following ONNX operator versions
REGISTER_MUSA_ARGMAXMIN_VERSIONED_ALL(ArgMin, 1, 10)
REGISTER_MUSA_ARGMAXMIN_VERSIONED_ALL(ArgMin, 11, 12)
REGISTER_MUSA_ARGMAXMIN_ALL(ArgMin, 13)

// Template instantiations are handled by the registration macros

} // namespace musa
} // namespace onnxruntime
