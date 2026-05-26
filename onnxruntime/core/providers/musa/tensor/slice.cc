// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/tensor/slice.h"
#include "core/providers/musa/tensor/slice_kernel.h"
#include "core/providers/common.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/musa_execution_provider.h"
#include "core/providers/cpu/tensor/slice_helper.h"
#include <algorithm>
#include <functional>
#include <musa_runtime.h>
#include <mudnn.h>
#include <string>
#include <vector>

using onnxruntime::common::Status;
namespace onnxruntime {
namespace musa {

/*
  NOTE: This MUSA Slice implementation does not inherit from SliceBase to avoid
  undefined symbol issues when loading the MUSA provider as a dynamic library.
  Instead, we directly use the inline helper functions from slice_helper.h and
  implement the required functionality locally within the MUSA provider.

  This approach trades some code duplication for stability and avoids complex
  symbol linking issues between the CPU and MUSA providers.
*/

// Helper function to copy data from input tensors (replaces SliceBase::FillVectorsFromInput)
template <bool dynamic>
Status Slice<dynamic>::FillVectorsFromInput(const Tensor& start_tensor, const Tensor& ends_tensor,
                                            const Tensor* axes_tensor, const Tensor* steps_tensor,
                                            TensorShapeVector& input_starts, TensorShapeVector& input_ends,
                                            TensorShapeVector& input_axes, TensorShapeVector& input_steps) const {
  // Check tensor data types and use Data<T>() instead of DataAsSpan to avoid symbol linking issues
  if (start_tensor.DataType() == DataTypeImpl::GetType<int32_t>()) {
    const int32_t* start_data = start_tensor.Data<int32_t>();
    const int32_t* ends_data = ends_tensor.Data<int32_t>();

    size_t start_size = start_tensor.Shape().Size();
    size_t ends_size = ends_tensor.Shape().Size();

    input_starts.assign(start_data, start_data + start_size);
    input_ends.assign(ends_data, ends_data + ends_size);

    if (axes_tensor != nullptr) {
      const int32_t* axes_data = axes_tensor->Data<int32_t>();
      size_t axes_size = axes_tensor->Shape().Size();
      input_axes.assign(axes_data, axes_data + axes_size);
    }

    if (steps_tensor != nullptr) {
      const int32_t* steps_data = steps_tensor->Data<int32_t>();
      size_t steps_size = steps_tensor->Shape().Size();
      input_steps.assign(steps_data, steps_data + steps_size);
    }
  } else if (start_tensor.DataType() == DataTypeImpl::GetType<int64_t>()) {
    const int64_t* start_data = start_tensor.Data<int64_t>();
    const int64_t* ends_data = ends_tensor.Data<int64_t>();

    size_t start_size = start_tensor.Shape().Size();
    size_t ends_size = ends_tensor.Shape().Size();

    input_starts.assign(start_data, start_data + start_size);
    input_ends.assign(ends_data, ends_data + ends_size);

    if (axes_tensor != nullptr) {
      const int64_t* axes_data = axes_tensor->Data<int64_t>();
      size_t axes_size = axes_tensor->Shape().Size();
      input_axes.assign(axes_data, axes_data + axes_size);
    }

    if (steps_tensor != nullptr) {
      const int64_t* steps_data = steps_tensor->Data<int64_t>();
      size_t steps_size = steps_tensor->Shape().Size();
      input_steps.assign(steps_data, steps_data + steps_size);
    }
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported tensor data type for slice indices");
  }

  return Status::OK();
}

// Helper function to flatten output dimensions (replaces SliceBase::FlattenOutputDims)
template <bool dynamic>
Status Slice<dynamic>::FlattenOutputDims(gsl::span<const int64_t> input_dimensions,
                                         gsl::span<const int64_t> output_dims,
                                         TensorShapeVector& starts, TensorShapeVector& ends, TensorShapeVector& steps,
                                         TensorShapeVector*& p_flattened_input_dims, TensorShapeVector*& p_flattened_output_dims) const {
  size_t cur = 0;
  size_t nxt = 0;
  while (true) {
    // Skip all leading slicing dims.
    while (nxt < starts.size() && (steps[nxt] != 1 || input_dimensions[nxt] != output_dims[nxt])) {
      p_flattened_input_dims->emplace_back(input_dimensions[nxt]);
      p_flattened_output_dims->emplace_back(output_dims[nxt]);
      starts[cur] = starts[nxt];
      ends[cur] = ends[nxt];
      steps[cur] = steps[nxt];
      ++cur;
      ++nxt;
    }
    if (nxt == starts.size()) {
      break;
    }
    // Coalesce contiguous non-slicing dims.
    int64_t running_size = 1;
    while (nxt < starts.size() && steps[nxt] == 1 && input_dimensions[nxt] == output_dims[nxt]) {
      running_size *= input_dimensions[nxt];
      ++nxt;
    }
    if (running_size > 1) {
      p_flattened_input_dims->emplace_back(running_size);
      p_flattened_output_dims->emplace_back(running_size);
      starts[cur] = 0LL;
      ends[cur] = running_size;
      steps[cur] = 1LL;
      ++cur;
    }
  }

  // No actual slice dim, and all dims are size 1.
  if (cur == 0) {
    p_flattened_input_dims->emplace_back(1LL);
    p_flattened_output_dims->emplace_back(1LL);
    starts[cur] = 0LL;
    ends[cur] = 1LL;
    steps[cur] = 1LL;
    ++cur;
  }

  if (p_flattened_output_dims->size() == output_dims.size()) {
    p_flattened_input_dims->clear();
    p_flattened_output_dims->clear();
    p_flattened_input_dims = nullptr;
    p_flattened_output_dims = nullptr;
  } else {
    starts.resize(cur);
    ends.resize(cur);
    steps.resize(cur);
  }

  return Status::OK();
}

// Helper function to determine if stride-based view optimization can be used
// This expands the fast path beyond simple contiguous cases
bool CanUseStridedView(const TensorShapeVector& starts,
                       const TensorShapeVector& ends,
                       const TensorShapeVector& steps,
                       const gsl::span<const int64_t>& input_dims) {
  // Condition 1: All steps must be positive (negative steps not supported yet)
  if (!steps.empty() &&
      std::any_of(steps.begin(), steps.end(), [](int64_t s) { return s <= 0; })) {
    return false;
  }

  // Condition 2: Dimensions cannot exceed 8 (muDNN API limitation)
  if (input_dims.size() > 8) {
    return false;
  }

  // Condition 3: For slices with step=1, even with non-zero starts or incomplete ends
  // we can still use optimization by adjusting pointer offset and size
  bool all_steps_one = steps.empty() ||
                       std::all_of(steps.begin(), steps.end(), [](int64_t s) { return s == 1; });

  if (all_steps_one) {
    // Step=1 cases can be optimized through address offset
    return true;
  }

  // Condition 4: Step>1 but the innermost dimension is contiguous
  // This allows for 2D/3D memory copy optimizations
  if (!steps.empty() && steps.back() == 1) {
    // Innermost dimension is contiguous, can use batch copy
    return true;
  }

  // Other cases return false for now, will be extended in Phase 2
  return false;
}

// Helper function to prepare for compute (replaces SliceBase::PrepareForCompute)
template <bool dynamic>
Status Slice<dynamic>::PrepareForCompute(gsl::span<const int64_t> raw_starts, gsl::span<const int64_t> raw_ends,
                                         gsl::span<const int64_t> raw_axes, SliceOp::PrepareForComputeMetadata& compute_metadata) const {
  ORT_RETURN_IF_ERROR(SliceOp::PrepareForComputeHelper(raw_starts, raw_ends, raw_axes, compute_metadata));
  ORT_RETURN_IF_ERROR(FlattenOutputDims(compute_metadata.input_dimensions_, compute_metadata.output_dims_, compute_metadata.starts_,
                                        compute_metadata.ends_, compute_metadata.steps_, compute_metadata.p_flattened_input_dims_,
                                        compute_metadata.p_flattened_output_dims_));
  return Status::OK();
}

template <bool dynamic>
Status Slice<dynamic>::PrepareForCompute(gsl::span<const int64_t> raw_starts, gsl::span<const int64_t> raw_ends,
                                         gsl::span<const int64_t> raw_axes, gsl::span<const int64_t> raw_steps,
                                         SliceOp::PrepareForComputeMetadata& compute_metadata) const {
  ORT_RETURN_IF_ERROR(SliceOp::PrepareForComputeHelper(raw_starts, raw_ends, raw_axes, raw_steps, compute_metadata));
  ORT_RETURN_IF_ERROR(FlattenOutputDims(compute_metadata.input_dimensions_, compute_metadata.output_dims_, compute_metadata.starts_,
                                        compute_metadata.ends_, compute_metadata.steps_, compute_metadata.p_flattened_input_dims_,
                                        compute_metadata.p_flattened_output_dims_));
  return Status::OK();
}

// MUSA device-based implementation using stride-based slice approach
template <typename T>
Status SimpleMusaSliceOp(const MusaPreparation& prepare,
                         SliceOp::PrepareForComputeMetadata& compute_metadata,
                         musaStream_t stream) {
  // Get tensor data from prepared MUSA tensors
  const T* input_data = reinterpret_cast<const T*>(prepare.input_a_ptr);
  T* output_data = reinterpret_cast<T*>(prepare.output_ptr);

  // Validate prepared tensors - allow null pointers for empty tensors
  if ((prepare.input_a_shape.Size() > 0 && !input_data) ||
      (prepare.output_size > 0 && !output_data)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid prepared tensor pointers");
  }

  if (prepare.inputTensors.empty() || prepare.outputTensors.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid prepared tensor configuration");
  }

  // Handle empty output tensor case
  if (prepare.output_size == 0) {
    // For empty output, no computation needed
    return Status::OK();
  }

  // Use stride-based approach to implement slice operation
  try {
    // Determine which dimensions to use (original vs flattened)
    const auto& input_shape = prepare.input_a_shape;
    bool use_flattened = (compute_metadata.p_flattened_input_dims_ != nullptr &&
                          compute_metadata.p_flattened_output_dims_ != nullptr);

    // Get the actual dimensions to use for stride calculation
    gsl::span<const int64_t> actual_input_dims;
    gsl::span<const int64_t> actual_output_dims;

    if (use_flattened) {
      actual_input_dims = gsl::span<const int64_t>(*compute_metadata.p_flattened_input_dims_);
      actual_output_dims = gsl::span<const int64_t>(*compute_metadata.p_flattened_output_dims_);
    } else {
      actual_input_dims = gsl::span<const int64_t>(input_shape.GetDims());
      actual_output_dims = gsl::span<const int64_t>(compute_metadata.output_dims_);
    }

    // Calculate input strides based on actual dimensions
    std::vector<int64_t> input_strides(actual_input_dims.size());
    int64_t stride = 1;
    for (int64_t i = static_cast<int64_t>(actual_input_dims.size()) - 1; i >= 0; --i) {
      input_strides[i] = stride;
      stride *= actual_input_dims[i];
    }

    // Calculate the memory offset based on slice starts
    int64_t offset_elements = 0;
    for (size_t i = 0; i < compute_metadata.starts_.size(); ++i) {
      offset_elements += compute_metadata.starts_[i] * input_strides[i];
    }

    // Calculate actual slice parameters for each dimension using actual dimensions
    std::vector<int64_t> slice_sizes(actual_input_dims.size());
    std::vector<int64_t> slice_strides(actual_input_dims.size());

    // Initialize with actual dimensions
    for (size_t i = 0; i < actual_input_dims.size(); ++i) {
      slice_sizes[i] = actual_input_dims[i];
      slice_strides[i] = input_strides[i];
    }

    // Apply slice parameters using actual dimensions
    for (size_t i = 0; i < compute_metadata.starts_.size() && i < actual_input_dims.size(); ++i) {
      size_t axis = i;
      int64_t start = compute_metadata.starts_[i];
      int64_t end = compute_metadata.ends_[i];
      int64_t step = (i < compute_metadata.steps_.size()) ? compute_metadata.steps_[i] : 1;

      // Calculate the output size for this dimension
      slice_sizes[axis] = (end - start + step - 1) / step;
      slice_strides[axis] = input_strides[axis] * step;
    }

    // Set up input tensor with offset address
    ::musa::dnn::Tensor input_tensor = prepare.inputTensors[0];
    ::musa::dnn::Tensor output_tensor = prepare.outputTensors[0];

    // Configure output tensor with slice dimensions and strides
    auto output_status = output_tensor.SetNdInfo(static_cast<int64_t>(slice_sizes.size()),
                                                 slice_sizes.data(),
                                                 slice_strides.data());
    if (output_status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to configure output tensor dimensions, status: " +
                                 std::to_string(static_cast<int>(output_status)));
    }

    // Set the offset address for input tensor
    const T* sliced_input_data = input_data + offset_elements;
    auto input_status = input_tensor.SetAddr(sliced_input_data);
    if (input_status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set input tensor address, status: " +
                                 std::to_string(static_cast<int>(input_status)));
    }

    // Now perform the actual copy operation
    size_t output_bytes = prepare.output_size * sizeof(T);

    // For simple contiguous slice, we can use direct memory copy
    // Note: Even if all steps are 1, we need strided copy if slice parameters
    // result in non-contiguous output (e.g., non-zero starts or non-sequential axes)
    bool all_starts_zero = std::all_of(compute_metadata.starts_.begin(), compute_metadata.starts_.end(),
                                       [](int64_t start) { return start == 0; });
    bool all_ends_match_dims = true;
    for (size_t i = 0; i < compute_metadata.ends_.size() && i < actual_input_dims.size(); ++i) {
      if (compute_metadata.ends_[i] != actual_input_dims[i]) {
        all_ends_match_dims = false;
        break;
      }
    }

    // Determine which optimization path to use
    bool is_simple_contiguous = (compute_metadata.steps_.empty() ||
                                 std::all_of(compute_metadata.steps_.begin(), compute_metadata.steps_.end(),
                                             [](int64_t step) { return step == 1; })) &&
                                all_starts_zero && all_ends_match_dims;

    bool can_use_strided_view = CanUseStridedView(
        compute_metadata.starts_,
        compute_metadata.ends_,
        compute_metadata.steps_,
        actual_input_dims);

    if (is_simple_contiguous) {
      // Path 1: Simple contiguous slice - direct copy (preserve original logic)
      auto musa_status = musaMemcpyAsync(output_data, sliced_input_data, output_bytes,
                                         musaMemcpyDeviceToDevice, stream);
      if (musa_status != musaSuccess) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "MUSA slice memory copy failed, status: " +
                                   std::to_string(static_cast<int>(musa_status)));
      }

    } else if (can_use_strided_view) {
      // Path 2: Extended optimization - handle step=1 non-contiguous cases
      // Utilize the already configured tensor settings (lines 300-316)

      // For 2D cases with contiguous inner dimension, use optimized 2D copy
      if (actual_input_dims.size() == 2 &&
          (compute_metadata.steps_.empty() || compute_metadata.steps_[1] == 1)) {
        // Use 2D optimized copy
        size_t width_bytes = actual_output_dims[1] * sizeof(T);
        size_t height = actual_output_dims[0];
        size_t src_pitch = actual_input_dims[1] * sizeof(T);
        size_t dst_pitch = actual_output_dims[1] * sizeof(T);

        auto musa_status = musaMemcpy2DAsync(
            output_data, dst_pitch,
            sliced_input_data, src_pitch,
            width_bytes, height,
            musaMemcpyDeviceToDevice, stream);

        if (musa_status != musaSuccess) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                 "MUSA 2D slice copy failed, status: " +
                                     std::to_string(static_cast<int>(musa_status)));
        }
      } else if (actual_input_dims.size() == 3 &&
                 (compute_metadata.steps_.empty() || compute_metadata.steps_[2] == 1)) {
        // Path 2B: 3D optimized copy using musaMemcpy3DAsync
        // For 3D tensor layout [depth, height, width]
        musaMemcpy3DParms copy_params = {};
        copy_params.srcArray = nullptr;
        copy_params.srcPos = {0, 0, 0};
        copy_params.dstArray = nullptr;
        copy_params.dstPos = {0, 0, 0};

        // Configure source parameters
        copy_params.srcPtr.ptr = (void*)sliced_input_data;
        copy_params.srcPtr.pitch = actual_input_dims[2] * sizeof(T);  // bytes per row (width * sizeof(T))
        copy_params.srcPtr.xsize = actual_input_dims[2] * sizeof(T);  // logical width in bytes
        copy_params.srcPtr.ysize = actual_input_dims[1];              // logical height

        // Configure destination parameters
        copy_params.dstPtr.ptr = (void*)output_data;
        copy_params.dstPtr.pitch = actual_output_dims[2] * sizeof(T);  // bytes per row (width * sizeof(T))
        copy_params.dstPtr.xsize = actual_output_dims[2] * sizeof(T);  // logical width in bytes
        copy_params.dstPtr.ysize = actual_output_dims[1];              // logical height

        // Configure copy extent - use output dimensions for the actual copy size
        copy_params.extent.width = actual_output_dims[2] * sizeof(T);  // width in bytes to copy
        copy_params.extent.height = actual_output_dims[1];             // height (rows) to copy
        copy_params.extent.depth = actual_output_dims[0];              // depth (layers) to copy
        copy_params.kind = musaMemcpyDeviceToDevice;

        auto musa_status = musaMemcpy3DAsync(&copy_params, stream);
        if (musa_status != musaSuccess) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                 "MUSA 3D slice copy failed, status: " +
                                     std::to_string(static_cast<int>(musa_status)));
        }
      } else {
        // For dimensions > 3D or complex cases, fallback to recursive copy for now
        // TODO: Phase 2 will implement optimized multi-dimensional copy
        goto fallback_recursive_copy;
      }

    } else {
    fallback_recursive_copy:
      // Path 3: Fallback to recursive copy for complex cases (preserve original logic)
      // This handles step>1 and other complex slicing patterns

      // Use actual dimensions (may be flattened) for the copy operation
      const size_t rank = actual_input_dims.size();

      SliceKernelParams params{};  // Initialize all fields to zero
      params.ndim = static_cast<int32_t>(rank);
      params.total_elements = static_cast<int64_t>(prepare.output_size);

      // Fill kernel parameters
      for (size_t i = 0; i < rank && i < kMaxSliceDimensions; ++i) {
        params.input_dims[i] = actual_input_dims[i];
        params.output_dims[i] = actual_output_dims[i];
        params.input_strides[i] = input_strides[i];
        params.starts[i] = compute_metadata.starts_[i];
        params.steps[i] = compute_metadata.steps_[i];
      }

      // Launch GPU kernel (templated by data type)
      LaunchSliceKernel<T>(input_data, output_data, params, stream);

      // Check for kernel errors
      musaError_t kernel_error = musaGetLastError();
      if (kernel_error != musaSuccess) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Slice GPU kernel failed: " +
                                   std::string(musaGetErrorString(kernel_error)));
      }
    }

  } catch (const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Exception in MUSA slice operation: " +
                               std::string(e.what()));
  }

  return Status::OK();
}

template <bool dynamic>
template <typename T>
Status Slice<dynamic>::Prepare(OpKernelContext* ctx, MusaPreparation& prepare,
                               SliceOp::PrepareForComputeMetadata& compute_metadata) const {
  // Get input tensor
  const Tensor* input_tensor = GetSlicedOrUnslicedTensor(ctx);
  if (!input_tensor) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input tensor");
  }

  // Compute output shape
  TensorShape output_shape(compute_metadata.output_dims_);

  // Create output tensor
  Tensor* output_tensor = ctx->Output(0, output_shape);
  if (!output_tensor) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor");
  }

  // Store tensor pointers and shapes in preparation
  prepare.input_a_ptr = input_tensor->DataRaw();
  prepare.output_ptr = output_tensor->MutableDataRaw();
  prepare.output_size = output_shape.Size();
  prepare.input_a_shape = input_tensor->Shape();
  prepare.output_shape = output_shape;

  // Validate tensor pointers - allow null pointers for empty tensors
  if ((prepare.input_a_shape.Size() > 0 && !prepare.input_a_ptr) ||
      (prepare.output_size > 0 && !prepare.output_ptr)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid tensor data pointers");
  }

  // Get MUSA data type
  const auto musaType = GetMusaDataType<T>();

  // Prepare MUSA tensors - using ORT_TRY/ORT_CATCH like other MusaEP operations
  ORT_TRY {
    // Set up MUSA stream for asynchronous execution
    // In EP mode, stream is already set in PerThreadContext constructor
    // In legacy mode, we need to set stream here
    auto* stream = Stream(ctx);
    if (prepare.handle && stream) {
      auto musa_status = prepare.handle->SetStream(stream);
      if (musa_status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Failed to set MUSA stream, status: " +
                                   std::to_string(static_cast<int>(musa_status)));
      }
    }

    // Initialize tensors vectors
    prepare.inputTensors.resize(1);
    prepare.outputTensors.resize(1);

    // Setup input tensor
    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.inputTensors[0], input_tensor, musaType, &prepare));

    // Setup output tensor
    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.outputTensors[0], output_tensor, musaType, &prepare));
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  return Status::OK();
}

template <bool dynamic>
Status Slice<dynamic>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* input_tensor = GetSlicedOrUnslicedTensor(ctx);
  ORT_ENFORCE(nullptr != input_tensor);
  const auto& input_shape = input_tensor->Shape();
  const auto input_dimensions = input_shape.GetDims();
  if (input_dimensions.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Cannot slice scalars");
  }

  SliceOp::PrepareForComputeMetadata compute_metadata(input_dimensions);
  // Prepare MUSA operation - use EP mode for thread-safe Handle management
  const auto* ep = static_cast<const MusaExecutionProvider*>(
      Info().GetExecutionProvider());
  MusaPreparation prepare(ep);

  if (dynamic) {
    TensorShapeVector input_starts;
    TensorShapeVector input_ends;
    TensorShapeVector input_axes;
    TensorShapeVector input_steps;

    ORT_RETURN_IF_ERROR(FillInputVectors(ctx, input_starts, input_ends, input_axes, input_steps));
    ORT_RETURN_IF_ERROR(PrepareForCompute(input_starts, input_ends, input_axes, input_steps, compute_metadata));
  } else {
    ORT_RETURN_IF_ERROR(PrepareForCompute(StartsAttribute(), EndsAttribute(), AxesAttribute(), compute_metadata));
  }

  // FlattenOutputDims is already called inside PrepareForCompute, no need to call again

  // Dispatch to different data types
  Status status = Status::OK();
  const auto& data_type = input_tensor->DataType();
  auto* stream = Stream(ctx);

  if (data_type == DataTypeImpl::GetType<float>()) {
    ORT_RETURN_IF_ERROR(Prepare<float>(ctx, prepare, compute_metadata));
    ORT_RETURN_IF_ERROR(SimpleMusaSliceOp<float>(prepare, compute_metadata, stream));
  } else if (data_type == DataTypeImpl::GetType<MLFloat16>()) {
    ORT_RETURN_IF_ERROR(Prepare<MLFloat16>(ctx, prepare, compute_metadata));
    ORT_RETURN_IF_ERROR(SimpleMusaSliceOp<MLFloat16>(prepare, compute_metadata, stream));
  } else if (data_type == DataTypeImpl::GetType<int32_t>()) {
    ORT_RETURN_IF_ERROR(Prepare<int32_t>(ctx, prepare, compute_metadata));
    ORT_RETURN_IF_ERROR(SimpleMusaSliceOp<int32_t>(prepare, compute_metadata, stream));
  } else if (data_type == DataTypeImpl::GetType<int64_t>()) {
    ORT_RETURN_IF_ERROR(Prepare<int64_t>(ctx, prepare, compute_metadata));
    ORT_RETURN_IF_ERROR(SimpleMusaSliceOp<int64_t>(prepare, compute_metadata, stream));
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported data type for MUSA Slice: ", data_type);
  }

  return Status::OK();
}

template <bool dynamic>
const Tensor* Slice<dynamic>::GetSlicedOrUnslicedTensor(OpKernelContext* ctx) const {
  return ctx->Input<Tensor>(0);
}

template <bool dynamic>
Status Slice<dynamic>::FillInputVectors(OpKernelContext* ctx, TensorShapeVector& input_starts,
                                        TensorShapeVector& input_ends, TensorShapeVector& input_axes,
                                        TensorShapeVector& input_steps) const {
  return FillVectorsFromInput(*ctx->Input<Tensor>(1), *ctx->Input<Tensor>(2),
                              ctx->Input<Tensor>(3), ctx->Input<Tensor>(4),
                              input_starts, input_ends, input_axes, input_steps);
}

// Macro for registering typed compute function with MUSA implementation
#define REGISTER_MUSA_SLICE_TYPED_COMPUTE(dynamic_flag) \
  template Status Slice<dynamic_flag>::ComputeInternal(OpKernelContext* ctx) const;

// Macro for registering typed kernel (v13+, with Tind constraint)
#define REGISTER_MUSA_SLICE_TYPED_KERNEL(ver, dynamic_flag, T)                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      Slice, kOnnxDomain, ver, T, kMusaExecutionProvider,                       \
      (*KernelDefBuilder::Create())                                             \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                               \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                               \
          .InputMemoryType(OrtMemTypeCPUInput, 3)                               \
          .InputMemoryType(OrtMemTypeCPUInput, 4)                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                \
          .TypeConstraint("Tind", std::vector<MLDataType>{                      \
                                      DataTypeImpl::GetTensorType<int32_t>(),   \
                                      DataTypeImpl::GetTensorType<int64_t>()}), \
      Slice<dynamic_flag>);

// Macro for registering versioned typed kernel (v1-9, no Tind constraint)
#define REGISTER_MUSA_SLICE_VERSIONED_TYPED_KERNEL_V1_9(startver, endver, dynamic_flag, T) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      Slice, kOnnxDomain, startver, endver, T, kMusaExecutionProvider,                     \
      (*KernelDefBuilder::Create())                                                        \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),                          \
      Slice<dynamic_flag>);

// Macro for registering versioned typed kernel (v10+, with Tind constraint)
#define REGISTER_MUSA_SLICE_VERSIONED_TYPED_KERNEL_V10_PLUS(startver, endver, dynamic_flag, T) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                     \
      Slice, kOnnxDomain, startver, endver, T, kMusaExecutionProvider,                         \
      (*KernelDefBuilder::Create())                                                            \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                                              \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                                              \
          .InputMemoryType(OrtMemTypeCPUInput, 3)                                              \
          .InputMemoryType(OrtMemTypeCPUInput, 4)                                              \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                               \
          .TypeConstraint("Tind", std::vector<MLDataType>{                                     \
                                      DataTypeImpl::GetTensorType<int32_t>(),                  \
                                      DataTypeImpl::GetTensorType<int64_t>()}),                \
      Slice<dynamic_flag>);

// Register Slice operations for supported data types and versions

// Slice operations (version 1-9) - non-dynamic, no Tind constraint
REGISTER_MUSA_SLICE_VERSIONED_TYPED_KERNEL_V1_9(1, 9, false, float)
REGISTER_MUSA_SLICE_VERSIONED_TYPED_KERNEL_V1_9(1, 9, false, MLFloat16)
REGISTER_MUSA_SLICE_VERSIONED_TYPED_KERNEL_V1_9(1, 9, false, int32_t)
REGISTER_MUSA_SLICE_VERSIONED_TYPED_KERNEL_V1_9(1, 9, false, int64_t)

// Slice operations (version 10) - dynamic, with Tind constraint
REGISTER_MUSA_SLICE_VERSIONED_TYPED_KERNEL_V10_PLUS(10, 10, true, float)
REGISTER_MUSA_SLICE_VERSIONED_TYPED_KERNEL_V10_PLUS(10, 10, true, MLFloat16)
REGISTER_MUSA_SLICE_VERSIONED_TYPED_KERNEL_V10_PLUS(10, 10, true, int32_t)
REGISTER_MUSA_SLICE_VERSIONED_TYPED_KERNEL_V10_PLUS(10, 10, true, int64_t)

// Slice operations (version 11-12) - dynamic, with Tind constraint
REGISTER_MUSA_SLICE_VERSIONED_TYPED_KERNEL_V10_PLUS(11, 12, true, float)
REGISTER_MUSA_SLICE_VERSIONED_TYPED_KERNEL_V10_PLUS(11, 12, true, MLFloat16)
REGISTER_MUSA_SLICE_VERSIONED_TYPED_KERNEL_V10_PLUS(11, 12, true, int32_t)
REGISTER_MUSA_SLICE_VERSIONED_TYPED_KERNEL_V10_PLUS(11, 12, true, int64_t)

// Slice operations (version 13+) - dynamic
REGISTER_MUSA_SLICE_TYPED_KERNEL(13, true, float)
REGISTER_MUSA_SLICE_TYPED_KERNEL(13, true, MLFloat16)
REGISTER_MUSA_SLICE_TYPED_KERNEL(13, true, int32_t)
REGISTER_MUSA_SLICE_TYPED_KERNEL(13, true, int64_t)

// Register compute implementations
REGISTER_MUSA_SLICE_TYPED_COMPUTE(false)
REGISTER_MUSA_SLICE_TYPED_COMPUTE(true)

}  // namespace musa
}  // namespace onnxruntime
