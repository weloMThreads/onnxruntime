// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "instance_norm.h"
#include "core/providers/cpu/nn/instance_norm_helper.h"
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/common.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/musa_execution_provider.h"
#include <musa_runtime.h>
#include <mudnn.h>

using onnxruntime::common::Status;
namespace onnxruntime {
namespace musa {

// MUSA device-based InstanceNormalization implementation using GroupNorm
template <typename T>
Status SimpleMusaInstanceNormOp(const MusaPreparation& prepare,
                                double epsilon,
                                int64_t num_channels) {
  // Get tensor data from prepared MUSA tensors
  const T* input_data = reinterpret_cast<const T*>(prepare.input_a_ptr);
  const T* scale_data = reinterpret_cast<const T*>(prepare.input_b_ptr);
  const T* bias_data = reinterpret_cast<const T*>(prepare.bias_ptr);
  T* output_data = reinterpret_cast<T*>(prepare.output_ptr);

  // Validate prepared tensors
  if (!input_data || !scale_data || !bias_data || !output_data) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid prepared tensor pointers for InstanceNormalization");
  }

  if (prepare.inputTensors.size() < 3 || prepare.outputTensors.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid prepared tensor configuration for InstanceNormalization");
  }

  // Use mudnn GroupNorm to implement InstanceNormalization
  // InstanceNorm = GroupNorm with groups = num_channels
  try {
    // Create mudnn GroupNorm operation
    ::musa::dnn::GroupNorm group_norm_op;

    // Set epsilon parameter
    auto status = group_norm_op.SetEpsilon(epsilon);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set GroupNorm epsilon for InstanceNormalization");
    }

    // Set groups = num_channels (this makes GroupNorm equivalent to InstanceNorm)
    status = group_norm_op.SetGroup(static_cast<int>(num_channels));
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set GroupNorm groups for InstanceNormalization");
    }

    // Set axis to 1 (channel dimension)
    status = group_norm_op.SetAxis(1);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set GroupNorm axis for InstanceNormalization");
    }

    // For GroupNorm we need mean and variance output tensors, but for InstanceNorm we don't use them
    // Create temporary tensors for mean and variance (they won't be used in the final output)
    ::musa::dnn::Tensor temp_mean, temp_invvar;
    
    // Set up temporary tensors with the same type as input
    const auto musa_type = GetMusaDataType<T>();
    status = temp_mean.SetType(musa_type);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set temp_mean tensor type");
    }
    
    status = temp_invvar.SetType(musa_type);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set temp_invvar tensor type");
    }

    // Run the GroupNorm operation (which implements InstanceNorm when groups=channels)
    status = group_norm_op.Run(prepare.GetHandle(),
                               const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]),  // output tensor
                               temp_mean,                                                   // mean (temporary)
                               temp_invvar,                                                 // inverse variance (temporary)
                               prepare.inputTensors[0],                                     // input tensor
                               prepare.inputTensors[1],                                     // scale/gamma tensor
                               prepare.inputTensors[2]);                                    // bias/beta tensor

    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "mudnn GroupNorm operation failed for InstanceNormalization, status: " +
                             std::to_string(static_cast<int>(status)));
    }

  } catch (const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Exception in mudnn GroupNorm operation for InstanceNormalization: " +
                           std::string(e.what()));
  }

  return Status::OK();
}

template <typename T>
InstanceNormalization<T>::InstanceNormalization(const OpKernelInfo& op_kernel_info)
    : MusaKernel(op_kernel_info) {
  float tmp_epsilon;
  ORT_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &tmp_epsilon).IsOK());
  epsilon_ = static_cast<double>(tmp_epsilon);
}

template <typename T>
Status InstanceNormalization<T>::Prepare(OpKernelContext* ctx, MusaPreparation& prepare) const {
  // 1. Get input tensors
  const Tensor* X = ctx->Input<Tensor>(0);
  const Tensor* scale = ctx->Input<Tensor>(1);
  const Tensor* bias = ctx->Input<Tensor>(2);
  
  if (X == nullptr || scale == nullptr || bias == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "InstanceNormalization requires 3 input tensors");
  }

  // 2. Validate inputs using the helper
  ORT_RETURN_IF_ERROR(InstanceNormHelper::ValidateInputs(X, scale, bias));

  // 3. Check if we need to handle 3D tensor reshape
  const auto& input_shape = X->Shape();
  bool is_3d_tensor = (input_shape.NumDimensions() == 3);
  
  TensorShape output_shape = X->Shape(); // Original shape for output
  TensorShape reshaped_input_shape = input_shape;
  
  if (is_3d_tensor) {
    // Reshape 3D tensor [N,C,W] to 4D tensor [N,C,1,W] for mudNN GroupNorm
    std::vector<int64_t> new_dims = {input_shape[0], input_shape[1], 1, input_shape[2]};
    reshaped_input_shape = TensorShape(new_dims);
  }

  // 4. Create output tensor with original shape (not reshaped)
  Tensor* Y = ctx->Output(0, output_shape);
  if (Y == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor");
  }

  // 5. Store tensor pointers and shapes in preparation
  prepare.input_a_ptr = X->DataRaw();
  prepare.input_b_ptr = scale->DataRaw();
  prepare.bias_ptr = bias->DataRaw();
  prepare.output_ptr = Y->MutableDataRaw();
  prepare.output_size = Y->Shape().Size();
  prepare.input_a_shape = reshaped_input_shape; // Use reshaped shape for computation
  prepare.output_shape = reshaped_input_shape;  // mudNN expects reshaped output shape

  if (prepare.input_a_ptr == nullptr || prepare.input_b_ptr == nullptr || 
      prepare.bias_ptr == nullptr || prepare.output_ptr == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid tensor data pointers");
  }

  // 5. Get MUSA data type
  const auto musaType = GetMusaDataType<T>();

  // 6. Prepare MUSA tensors - using ORT_TRY/ORT_CATCH like other MusaEP operations
  ORT_TRY {
    // Set up MUSA stream for asynchronous execution
    auto* stream = Stream(ctx);
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

    // Initialize tensors vectors (input, scale, bias, output)
    prepare.inputTensors.resize(3);
    prepare.outputTensors.resize(1);

    // Setup input tensor
    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.inputTensors[0], X, musaType, &prepare));

    // Setup scale tensor
    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.inputTensors[1], scale, musaType, &prepare));

    // Setup bias tensor
    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.inputTensors[2], bias, musaType, &prepare));

    // Setup output tensor
    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.outputTensors[0], Y, musaType, &prepare));
    
    // For 3D tensors, reshape mudnn tensors to 4D format for GroupNorm compatibility
    if (is_3d_tensor) {
      // Reshape input tensor from [N,C,W] to [N,C,1,W]
      std::vector<int64_t> reshaped_dims = {input_shape[0], input_shape[1], 1, input_shape[2]};
      
      // Reshape input tensor dimensions
      auto status = prepare.inputTensors[0].SetNdInfo(static_cast<int>(reshaped_dims.size()), reshaped_dims.data());
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to reshape input tensor for 3D InstanceNorm");
      }
      
      // Set format to NCHW for GroupNorm compatibility (not NCW)
      status = prepare.inputTensors[0].SetFormat(::musa::dnn::Tensor::Format::NCHW);
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set NCHW format for input tensor");
      }
      
      // Reshape output tensor similarly
      status = prepare.outputTensors[0].SetNdInfo(static_cast<int>(reshaped_dims.size()), reshaped_dims.data());
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to reshape output tensor for 3D InstanceNorm");
      }
      
      status = prepare.outputTensors[0].SetFormat(::musa::dnn::Tensor::Format::NCHW);
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set NCHW format for output tensor");
      }
    }
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  return Status::OK();
}

template <typename T>
Status InstanceNormalization<T>::ComputeInternal(OpKernelContext* ctx) const {
  // Get input tensors first to check dimensions
  const Tensor* X = ctx->Input<Tensor>(0);
  if (X == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input tensor is null");
  }

  const auto& input_shape = X->Shape();
  if (input_shape.NumDimensions() < 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "InstanceNormalization requires at least 2D input");
  }
  
  const int64_t num_channels = input_shape[1];

  // Check if this is a 3D tensor (NCW format) - mudnn GroupNorm doesn't support it
  // We'll handle 3D tensors by reshaping them to 4D (adding a dummy dimension)
  bool is_3d_tensor = (input_shape.NumDimensions() == 3);
  
  // Handle 3D tensors by reshaping to 4D format for mudNN GroupNorm compatibility

  // Handle 4D and higher dimensions normally
  const auto* ep = static_cast<const MusaExecutionProvider*>(
      Info().GetExecutionProvider());
  MusaPreparation prepare(ep);
  ORT_RETURN_IF_ERROR(this->Prepare(ctx, prepare));
  ORT_RETURN_IF_ERROR(SimpleMusaInstanceNormOp<T>(prepare, epsilon_, num_channels));
  return Status::OK();
}

// Macro for registering typed kernel
#define REGISTER_MUSA_INSTANCENORM_TYPED_KERNEL(ver, T)                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      InstanceNormalization,                                                    \
      kOnnxDomain,                                                              \
      ver,                                                                      \
      T,                                                                        \
      kMusaExecutionProvider,                                                   \
      (*KernelDefBuilder::Create())                                             \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),               \
      InstanceNormalization<T>);

// Macro for registering versioned typed kernel
#define REGISTER_MUSA_INSTANCENORM_VERSIONED_TYPED_KERNEL(startver, endver, T) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                     \
      InstanceNormalization,                                                    \
      kOnnxDomain,                                                              \
      startver,                                                                 \
      endver,                                                                   \
      T,                                                                        \
      kMusaExecutionProvider,                                                   \
      (*KernelDefBuilder::Create())                                             \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),               \
      InstanceNormalization<T>);

// Register InstanceNormalization for different ONNX versions
// ONNX v1-5: Basic InstanceNormalization
REGISTER_MUSA_INSTANCENORM_VERSIONED_TYPED_KERNEL(1, 5, float)
REGISTER_MUSA_INSTANCENORM_VERSIONED_TYPED_KERNEL(1, 5, MLFloat16)

// ONNX v6+: Extended InstanceNormalization (current)
REGISTER_MUSA_INSTANCENORM_TYPED_KERNEL(6, float)
REGISTER_MUSA_INSTANCENORM_TYPED_KERNEL(6, MLFloat16)

}  // namespace musa
}  // namespace onnxruntime
