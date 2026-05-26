// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/common.h"
#include "onehot.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/musa_execution_provider.h"
#include <musa_runtime.h>
#include <mudnn.h>

using onnxruntime::common::Status;
namespace onnxruntime {
namespace musa {

// Helper function to get data type for mudnn tensor
template <typename T>
::musa::dnn::Tensor::Type GetMusaDataType() {
  if constexpr (std::is_same_v<T, int8_t>) {
    return ::musa::dnn::Tensor::Type::INT8;
  } else if constexpr (std::is_same_v<T, int16_t>) {
    return ::musa::dnn::Tensor::Type::INT16;
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return ::musa::dnn::Tensor::Type::INT32;
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return ::musa::dnn::Tensor::Type::INT64;
  } else if constexpr (std::is_same_v<T, MLFloat16>) {
    return ::musa::dnn::Tensor::Type::HALF;
  } else if constexpr (std::is_same_v<T, float>) {
    return ::musa::dnn::Tensor::Type::FLOAT;
  } else if constexpr (std::is_same_v<T, double>) {
    return ::musa::dnn::Tensor::Type::DOUBLE;
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    return ::musa::dnn::Tensor::Type::UINT8;
  } else if constexpr (std::is_same_v<T, bool>) {
    return ::musa::dnn::Tensor::Type::BOOL;
  } else {
    return ::musa::dnn::Tensor::Type::FLOAT; // Default
  }
}

// MUSA device implementation using custom kernel approach (similar to torch_musa)
template <typename in_type, typename out_type, typename depth_type>
Status SimpleMusaOneHotOp(const MusaPreparation& prepare, 
                         int64_t axis, 
                         int64_t depth_val, 
                         out_type on_value, 
                         out_type off_value) {
  const in_type* indices_data = reinterpret_cast<const in_type*>(prepare.input_a_ptr);
  out_type* output_data = reinterpret_cast<out_type*>(prepare.output_ptr);
  
  // Handle zero-dimension case - pointers can be null for empty tensors
  if (prepare.input_a_shape.Size() == 0 || prepare.output_shape.Size() == 0) {
    return Status::OK();  // Empty tensor case - nothing to do
  }
  
  if (!indices_data || !output_data) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid tensor pointers");
  }

  // Get the prepared tensors
  if (prepare.inputTensors.empty() || prepare.outputTensors.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid prepared tensor configuration");
  }

  try {
    // First initialize output tensor with off_value using mudnn Fill
    ::musa::dnn::Fill fill_op;
      
    // Set fill value
    auto status = fill_op.SetValue(static_cast<double>(off_value));
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set fill value, status: " +
                             std::to_string(static_cast<int>(status)));
    }

    // Fill output tensor with off_value
    status = fill_op.Run(prepare.GetHandle(),
                         const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]));
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to fill output tensor, status: " +
                             std::to_string(static_cast<int>(status)));
    }

    // For now, implement a simplified version using Scatter approach
    // This is a basic implementation - for better performance, custom MUSA kernel could be added later
      
    // Calculate dimensions for scatter operation
    const auto& input_shape = prepare.input_a_shape;
    const auto& output_shape = prepare.output_shape;
      
    int64_t prefix_dim_size = 1;
    int64_t suffix_dim_size = 1;
      
    // Calculate prefix and suffix dimensions
    for (int64_t i = 0; i < axis; ++i) {
      prefix_dim_size *= input_shape[i];
    }
      
    for (size_t i = axis + 1; i < output_shape.NumDimensions(); ++i) {
      suffix_dim_size *= output_shape[i];
    }
      
    // Use a simple approach: iterate through indices and set on_value
    // Note: This is a CPU-based fallback for the initial implementation
    // A more efficient GPU kernel implementation can be added later
      
    // For now, copy to CPU, process, and copy back
    // This is not optimal but provides a working implementation
      
    auto indices_size = prepare.input_a_shape.Size();
    std::vector<in_type> cpu_indices(indices_size);
    std::vector<out_type> cpu_output(prepare.output_size);
      
    // Copy indices from GPU to CPU
    auto musa_status = musaMemcpy(cpu_indices.data(), indices_data,
                                  indices_size * sizeof(in_type),
                                  musaMemcpyDeviceToHost);
    if (musa_status != musaSuccess) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to copy indices to CPU");
    }
      
    // Initialize CPU output with off_value (mudnn Fill might not work properly)
    std::fill(cpu_output.begin(), cpu_output.end(), off_value);
      
    // Process OneHot on CPU
    for (int64_t i = 0; i < indices_size; ++i) {
      in_type idx = cpu_indices[i];
        
      // Handle negative indices
      if (idx < 0) {
        idx += static_cast<in_type>(depth_val);
      }
        
      // Check bounds
      if (idx >= 0 && idx < static_cast<in_type>(depth_val)) {
        int64_t prefix_idx = i / suffix_dim_size;
        int64_t suffix_idx = i % suffix_dim_size;
        int64_t output_idx = prefix_idx * depth_val * suffix_dim_size +
                            static_cast<int64_t>(idx) * suffix_dim_size + suffix_idx;
          
        if (output_idx < static_cast<int64_t>(prepare.output_size)) {
          cpu_output[output_idx] = on_value;
        }
      }
    }
      
    // Copy result back to GPU
    musa_status = musaMemcpy(output_data, cpu_output.data(),
                             prepare.output_size * sizeof(out_type),
                             musaMemcpyHostToDevice);
    if (musa_status != musaSuccess) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to copy output back to GPU");
    }
  } catch (const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, 
                           "Exception in MUSA OneHot operation: " + std::string(e.what()));
  }

  return Status::OK();
}

template <typename in_type, typename out_type, typename depth_type>
Status OneHot<in_type, out_type, depth_type>::Prepare(OpKernelContext* ctx, MusaPreparation& prepare) const {
  // Get input tensors
  const Tensor* indices_tensor = ctx->Input<Tensor>(0);
  const Tensor* depth_tensor = ctx->Input<Tensor>(1);
  const Tensor* values_tensor = ctx->Input<Tensor>(2);

  if (!indices_tensor || !depth_tensor || !values_tensor) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input tensors");
  }

  // Get depth value (should be a scalar on CPU)
  if (depth_tensor->Shape().Size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Depth must be a scalar");
  }
  
  // Get values (should have 2 elements: [off_value, on_value])
  if (values_tensor->Shape().Size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Values must contain exactly 2 elements");
  }

  // Get depth value
  depth_type depth_val = *depth_tensor->Data<depth_type>();
  if (depth_val < 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Depth must be non-negative");
  }

  // Calculate output shape
  const auto& input_shape = indices_tensor->Shape();
  std::vector<int64_t> output_dims;
  
  int64_t actual_axis = axis_;
  if (actual_axis < 0) {
    actual_axis += input_shape.NumDimensions() + 1;
  }
  
  if (actual_axis < 0 || actual_axis > static_cast<int64_t>(input_shape.NumDimensions())) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid axis");
  }

  // Build output shape by inserting depth dimension at specified axis
  for (int64_t i = 0; i < static_cast<int64_t>(input_shape.NumDimensions()); ++i) {
    if (i == actual_axis) {
      output_dims.push_back(static_cast<int64_t>(depth_val));
    }
    output_dims.push_back(input_shape[i]);
  }
  
  if (actual_axis == static_cast<int64_t>(input_shape.NumDimensions())) {
    output_dims.push_back(static_cast<int64_t>(depth_val));
  }

  TensorShape output_shape(output_dims);
  Tensor* output_tensor = ctx->Output(0, output_shape);
  if (!output_tensor) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor");
  }

  // Store tensor information in preparation
  prepare.input_a_ptr = indices_tensor->DataRaw();
  prepare.input_b_ptr = depth_tensor->DataRaw();
  prepare.input_c_ptr = values_tensor->DataRaw();
  prepare.output_ptr = output_tensor->MutableDataRaw();
  prepare.output_size = output_tensor->Shape().Size();
  prepare.input_a_shape = indices_tensor->Shape();
  prepare.input_b_shape = depth_tensor->Shape();
  prepare.input_c_shape = values_tensor->Shape();
  prepare.output_shape = output_tensor->Shape();

  // Get MUSA data types
  const auto indices_musa_type = GetMusaDataType<in_type>();
  const auto output_musa_type = GetMusaDataType<out_type>();

  // Set up MUSA stream
  ORT_TRY {
    auto* stream = Stream(ctx);
    if (prepare.handle) {
      if (stream) {
        auto musa_status = prepare.handle->SetStream(stream);
        if (musa_status != ::musa::dnn::Status::SUCCESS) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                 "Failed to set MUSA stream, status: " +
                                 std::to_string(static_cast<int>(musa_status)));
        }
      }
    }

    // Initialize tensor vectors
    prepare.inputTensors.resize(1);  // Only indices tensor needs MUSA tensor setup
    prepare.outputTensors.resize(1);

    // Setup indices tensor
    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.inputTensors[0], indices_tensor, indices_musa_type, &prepare));

    // Setup output tensor
    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.outputTensors[0], output_tensor, output_musa_type, &prepare));
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  return Status::OK();
}

template <typename in_type, typename out_type, typename depth_type>
Status OneHot<in_type, out_type, depth_type>::ComputeInternal(OpKernelContext* ctx) const {
  // Get values tensor for on/off values
  const Tensor* values_tensor = ctx->Input<Tensor>(2);
  const Tensor* depth_tensor = ctx->Input<Tensor>(1);
  
  if (!values_tensor || !depth_tensor) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Missing input tensors");
  }

  // Get depth and values
  depth_type depth_val = *depth_tensor->Data<depth_type>();
  const out_type* values_data = values_tensor->Data<out_type>();
  out_type off_value = values_data[0];
  out_type on_value = values_data[1];

  // Prepare MUSA operation
  const auto* ep = static_cast<const MusaExecutionProvider*>(
      Info().GetExecutionProvider());
  MusaPreparation prepare(ep);
  ORT_RETURN_IF_ERROR(Prepare(ctx, prepare));

  // Calculate actual axis
  const auto& input_shape = ctx->Input<Tensor>(0)->Shape();
  int64_t actual_axis = axis_;
  if (actual_axis < 0) {
    actual_axis += input_shape.NumDimensions() + 1;
  }

  // Call MUSA device OneHot operation
  Status status = SimpleMusaOneHotOp<in_type, out_type, depth_type>(
      prepare, actual_axis, static_cast<int64_t>(depth_val), on_value, off_value);
  ORT_RETURN_IF_ERROR(status);

  return Status::OK();
}

// Macro for registering typed compute function
#define REGISTER_MUSA_ONEHOT_TYPED_COMPUTE(in_type, out_type, depth_type) \
  template Status OneHot<in_type, out_type, depth_type>::ComputeInternal(OpKernelContext* ctx) const;

// Macro for registering typed kernel
#define REGISTER_MUSA_ONEHOT_TYPED_KERNEL(ver, in_type, out_type, depth_type) \
  ONNX_OPERATOR_TYPED_KERNEL_EX( \
      OneHot, kOnnxDomain, ver, in_type##_##out_type##_##depth_type, kMusaExecutionProvider, \
      (*KernelDefBuilder::Create()) \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<in_type>()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<depth_type>()) \
          .TypeConstraint("T3", DataTypeImpl::GetTensorType<out_type>()) \
          .InputMemoryType(OrtMemTypeCPUInput, 1) \
          .InputMemoryType(OrtMemTypeCPUInput, 2), \
      OneHot<in_type, out_type, depth_type>);

// Macro for versioned typed kernel
#define REGISTER_MUSA_ONEHOT_VERSIONED_TYPED_KERNEL(startver, endver, in_type, out_type, depth_type) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX( \
      OneHot, kOnnxDomain, startver, endver, in_type##_##out_type##_##depth_type, kMusaExecutionProvider, \
      (*KernelDefBuilder::Create()) \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<in_type>()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<depth_type>()) \
          .TypeConstraint("T3", DataTypeImpl::GetTensorType<out_type>()) \
          .InputMemoryType(OrtMemTypeCPUInput, 1) \
          .InputMemoryType(OrtMemTypeCPUInput, 2), \
      OneHot<in_type, out_type, depth_type>);

// Combined macro for both kernel and compute registration
#define REGISTER_MUSA_ONEHOT_TYPED(ver, in_type, out_type, depth_type) \
  REGISTER_MUSA_ONEHOT_TYPED_KERNEL(ver, in_type, out_type, depth_type) \
  REGISTER_MUSA_ONEHOT_TYPED_COMPUTE(in_type, out_type, depth_type)

// Macro for versioned typed registration
#define REGISTER_MUSA_ONEHOT_VERSIONED_TYPED(startver, endver, in_type, out_type, depth_type) \
  REGISTER_MUSA_ONEHOT_VERSIONED_TYPED_KERNEL(startver, endver, in_type, out_type, depth_type)

// Register OneHot operations for supported data types
// Note: Following MUSA documentation and avoiding double/bfloat16 as mentioned

// OneHot operations (version 9-10)
REGISTER_MUSA_ONEHOT_VERSIONED_TYPED(9, 10, int32_t, float, int64_t)
REGISTER_MUSA_ONEHOT_VERSIONED_TYPED(9, 10, int32_t, int32_t, int64_t)
REGISTER_MUSA_ONEHOT_VERSIONED_TYPED(9, 10, int32_t, int64_t, int64_t)
REGISTER_MUSA_ONEHOT_VERSIONED_TYPED(9, 10, int64_t, float, int64_t)
REGISTER_MUSA_ONEHOT_VERSIONED_TYPED(9, 10, int64_t, int32_t, int64_t)
REGISTER_MUSA_ONEHOT_VERSIONED_TYPED(9, 10, int64_t, int64_t, int64_t)

// OneHot operations (version 11+)
REGISTER_MUSA_ONEHOT_TYPED(11, int32_t, float, int64_t)
REGISTER_MUSA_ONEHOT_TYPED(11, int32_t, int32_t, int64_t)
REGISTER_MUSA_ONEHOT_TYPED(11, int32_t, int64_t, int64_t)
REGISTER_MUSA_ONEHOT_TYPED(11, int32_t, MLFloat16, int64_t)
REGISTER_MUSA_ONEHOT_TYPED(11, int32_t, uint8_t, int64_t)
REGISTER_MUSA_ONEHOT_TYPED(11, int32_t, int8_t, int64_t)
REGISTER_MUSA_ONEHOT_TYPED(11, int64_t, float, int64_t)
REGISTER_MUSA_ONEHOT_TYPED(11, int64_t, int32_t, int64_t)
REGISTER_MUSA_ONEHOT_TYPED(11, int64_t, int64_t, int64_t)
REGISTER_MUSA_ONEHOT_TYPED(11, int64_t, MLFloat16, int64_t)
REGISTER_MUSA_ONEHOT_TYPED(11, int64_t, uint8_t, int64_t)
REGISTER_MUSA_ONEHOT_TYPED(11, int64_t, int8_t, int64_t)

} // namespace musa
} // namespace onnxruntime
