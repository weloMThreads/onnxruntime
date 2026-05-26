// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/common.h"
#include "core/providers/musa/math/sum.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/musa_execution_provider.h"
#include <musa_runtime.h>
#include <mudnn.h>
#include <string>
#include <vector>
#include <algorithm>

using onnxruntime::common::Status;
namespace onnxruntime {
namespace musa {

// MUSA device-based implementation for binary sum (equivalent to Add)
template <typename T>
Status SimpleMusaBinarySumOp(const MusaPreparation& prepare) {
  // Get tensor data from prepared MUSA tensors
  const T* input_a = reinterpret_cast<const T*>(prepare.input_a_ptr);
  const T* input_b = reinterpret_cast<const T*>(prepare.input_b_ptr);
  T* output = reinterpret_cast<T*>(prepare.output_ptr);

  // Validate prepared tensors
  if (!input_a || !input_b || !output) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid prepared tensor pointers");
  }

  if (prepare.inputTensors.size() < 2 || prepare.outputTensors.size() < 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid prepared tensor configuration");
  }

  // Use mudnn Binary class for device computation with broadcasting support
  try {
    // Create mudnn Binary operation
    ::musa::dnn::Binary binary_op;

    // Set the operation mode to ADD
    auto status = binary_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set mudnn Binary mode to ADD");
    }

    // Run the binary operation directly on device with automatic broadcasting
    status = binary_op.Run(prepare.GetHandle(),
                          const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]),  // output tensor
                          prepare.inputTensors[0],   // input A tensor
                          prepare.inputTensors[1]);  // input B tensor

    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "mudnn Binary Sum operation failed, status: " +
                             std::to_string(static_cast<int>(status)));
    }

  } catch (const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Exception in mudnn Binary Sum operation: " +
                           std::string(e.what()));
  }

  return Status::OK();
}

// MUSA device-based implementation for variadic sum (multiple inputs)
template <typename T>
Status SimpleMusaVariadicSumOp(const MusaPreparation& prepare, size_t num_inputs) {
  // For multiple inputs, we perform sequential binary additions
  // temp = input[0] + input[1], temp = temp + input[2], etc.
  
  if (num_inputs < 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Variadic sum requires at least 2 inputs");
  }

  if (prepare.inputTensors.size() < num_inputs || prepare.outputTensors.size() < 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid prepared tensor configuration");
  }

  try {
    // Create mudnn Binary operation
    ::musa::dnn::Binary binary_op;
    auto status = binary_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set mudnn Binary mode to ADD");
    }

    // For 2 inputs, directly compute to output
    if (num_inputs == 2) {
      status = binary_op.Run(prepare.GetHandle(),
                            const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]),
                            prepare.inputTensors[0],
                            prepare.inputTensors[1]);
      
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "mudnn Binary Sum operation failed for 2 inputs, status: " +
                               std::to_string(static_cast<int>(status)));
      }
    } else {
      // For more than 2 inputs, we need intermediate tensors
      // Note: This is a simplified approach. In production, we might want to use 
      // workspace memory for intermediate results
      
      // First, add input[0] + input[1] -> output
      status = binary_op.Run(prepare.GetHandle(),
                            const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]),
                            prepare.inputTensors[0],
                            prepare.inputTensors[1]);
      
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "mudnn Binary Sum operation failed for first two inputs, status: " +
                               std::to_string(static_cast<int>(status)));
      }

      // Then iteratively add remaining inputs: output = output + input[i]
      for (size_t i = 2; i < num_inputs; i++) {
        status = binary_op.Run(prepare.GetHandle(),
                              const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]),
                              prepare.outputTensors[0],   // current accumulated sum
                              prepare.inputTensors[i]);   // next input
        
        if (status != ::musa::dnn::Status::SUCCESS) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                 "mudnn Binary Sum operation failed for input " + 
                                 std::to_string(i) + ", status: " +
                                 std::to_string(static_cast<int>(status)));
        }
      }
    }

  } catch (const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Exception in mudnn Variadic Sum operation: " +
                           std::string(e.what()));
  }

  return Status::OK();
}

template <typename T>
Status Sum<T>::Prepare(OpKernelContext* ctx, MusaPreparation& prepare) const {
  // Get number of inputs
  const auto num_inputs = ctx->InputCount();
  if (num_inputs < 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Sum requires at least 1 input");
  }

  // Get all input tensors
  std::vector<const Tensor*> input_tensors;
  input_tensors.reserve(num_inputs);
  
  for (int i = 0; i < num_inputs; i++) {
    const Tensor* input = ctx->Input<Tensor>(i);
    if (!input) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Invalid input tensor at index " + std::to_string(i));
    }
    input_tensors.push_back(input);
  }

  // Handle single input case - just copy
  if (num_inputs == 1) {
    Tensor* output = ctx->Output(0, input_tensors[0]->Shape());
    if (!output) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor");
    }
    
    // Copy data directly
    auto status = musaMemcpyAsync(output->MutableDataRaw(),
                                 input_tensors[0]->DataRaw(),
                                 input_tensors[0]->SizeInBytes(),
                                 musaMemcpyDeviceToDevice,
                                 static_cast<musaStream_t>(ctx->GetComputeStream()->GetHandle()));
    if (status != musaSuccess) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to copy single input to output");
    }
    return Status::OK();
  }

  // Compute output shape through broadcasting
  TensorShape output_shape = input_tensors[0]->Shape();
  for (size_t i = 1; i < input_tensors.size(); i++) {
    TensorShape temp_shape;
    ORT_RETURN_IF_ERROR(ComputeBroadcastOutputShape(Node().Name(), 
                                                   output_shape,
                                                   input_tensors[i]->Shape(), 
                                                   temp_shape));
    output_shape = temp_shape;
  }

  // Create output tensor
  Tensor* output = ctx->Output(0, output_shape);
  if (!output) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor");
  }

  // Store basic tensor info for compatibility with existing helper functions
  prepare.output_ptr = output->MutableDataRaw();
  prepare.output_size = output_shape.Size();
  prepare.output_shape = output_shape;
  
  // For binary case, also set input_a_ptr and input_b_ptr for compatibility
  if (num_inputs == 2) {
    prepare.input_a_ptr = input_tensors[0]->DataRaw();
    prepare.input_b_ptr = input_tensors[1]->DataRaw();
    prepare.input_a_shape = input_tensors[0]->Shape();
    prepare.input_b_shape = input_tensors[1]->Shape();
  }

  // Setup MUSA tensors
  ORT_TRY {
    // Get MUSA stream
    auto* ort_stream = ctx->GetComputeStream();
    musaStream_t stream = nullptr;
    if (ort_stream) {
      stream = static_cast<musaStream_t>(ort_stream->GetHandle());
    }

    if (prepare.handle && stream) {
      auto status = prepare.handle->SetStream(stream);
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set MUSA stream");
      }
    }

    prepare.inputTensors.resize(num_inputs);
    prepare.outputTensors.resize(1);

    // Setup input tensors
    const auto musa_type = GetMusaDataType<T>();
    for (size_t i = 0; i < input_tensors.size(); i++) {
      ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.inputTensors[i], input_tensors[i], musa_type, &prepare));
    }
    
    // Setup output tensor
    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.outputTensors[0], output, musa_type, &prepare));
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  return Status::OK();
}

template <typename T>
Status Sum<T>::ComputeBinarySum(OpKernelContext* ctx, MusaPreparation& prepare) const {
  // Use binary sum for 2-input case
  return SimpleMusaBinarySumOp<T>(prepare);
}

template <typename T>
Status Sum<T>::ComputeVariadicSum(OpKernelContext* ctx, MusaPreparation& prepare) const {
  // Use variadic sum for multi-input case
  const auto num_inputs = ctx->InputCount();
  return SimpleMusaVariadicSumOp<T>(prepare, num_inputs);
}

// Macro for registering typed compute function
#define REGISTER_MUSA_SUM_TYPED_COMPUTE(T)                                   \
  template <>                                                                \
  Status Sum<T>::ComputeInternal(OpKernelContext* ctx) const {               \
    const auto num_inputs = ctx->InputCount();                               \
    if (num_inputs < 1) {                                                    \
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,                 \
                             "Sum requires at least 1 input");              \
    }                                                                        \
                                                                             \
    const auto* ep = static_cast<const MusaExecutionProvider*>(             \
        Info().GetExecutionProvider());                                      \
    MusaPreparation prepare(ep);                                            \
    ORT_RETURN_IF_ERROR(Prepare(ctx, prepare));                             \
                                                                             \
    if (num_inputs == 1) {                                                   \
      return Status::OK();                                                   \
    } else if (num_inputs == 2) {                                            \
      return ComputeBinarySum(ctx, prepare);                                 \
    } else {                                                                 \
      return ComputeVariadicSum(ctx, prepare);                               \
    }                                                                        \
  }

// Add debug log to verify this is being compiled
#pragma message "Compiling Sum operator registration macros"

// Sum operator registration using versioned macros to match EP declarations  
#define REGISTER_MUSA_SUM_TYPED_KERNEL_VERSIONED(start_ver, end_ver, T)           \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                        \
      Sum, kOnnxDomain, start_ver, end_ver, T, kMusaExecutionProvider,            \
      (*KernelDefBuilder::Create())                                                \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),                  \
      Sum<T>);

#define REGISTER_MUSA_SUM_TYPED_KERNEL_SINGLE(ver, T)                            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                  \
      Sum, kOnnxDomain, ver, T, kMusaExecutionProvider,                           \
      (*KernelDefBuilder::Create())                                                \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),                  \
      Sum<T>);

// Register compute functions for different types
REGISTER_MUSA_SUM_TYPED_COMPUTE(int32_t)
REGISTER_MUSA_SUM_TYPED_COMPUTE(int64_t)  
REGISTER_MUSA_SUM_TYPED_COMPUTE(MLFloat16)
REGISTER_MUSA_SUM_TYPED_COMPUTE(float)
// Note: double and bfloat16 are not supported in MusaEP as per requirements

// Register Sum operations for supported data types using versioned registration
// Following Sum opset evolution: 6-7, 8-12, 13+ (matching EP declarations)

// ONNX opset 6-7: Basic Sum operation
REGISTER_MUSA_SUM_TYPED_KERNEL_VERSIONED(6, 7, int32_t)
REGISTER_MUSA_SUM_TYPED_KERNEL_VERSIONED(6, 7, int64_t)
REGISTER_MUSA_SUM_TYPED_KERNEL_VERSIONED(6, 7, MLFloat16)
REGISTER_MUSA_SUM_TYPED_KERNEL_VERSIONED(6, 7, float)

// ONNX opset 8-12: Sum operation with broadcasting support
REGISTER_MUSA_SUM_TYPED_KERNEL_VERSIONED(8, 12, int32_t)
REGISTER_MUSA_SUM_TYPED_KERNEL_VERSIONED(8, 12, int64_t)
REGISTER_MUSA_SUM_TYPED_KERNEL_VERSIONED(8, 12, MLFloat16)
REGISTER_MUSA_SUM_TYPED_KERNEL_VERSIONED(8, 12, float)

// ONNX opset 13+: Modern Sum implementation  
REGISTER_MUSA_SUM_TYPED_KERNEL_SINGLE(13, int32_t)
REGISTER_MUSA_SUM_TYPED_KERNEL_SINGLE(13, int64_t)
REGISTER_MUSA_SUM_TYPED_KERNEL_SINGLE(13, MLFloat16)
REGISTER_MUSA_SUM_TYPED_KERNEL_SINGLE(13, float)

}  // namespace musa
}  // namespace onnxruntime