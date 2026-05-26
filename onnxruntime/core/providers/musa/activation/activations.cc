// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/activation/activations.h"
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/common.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/musa_execution_provider.h"
#include <musa_runtime.h>
#include <mudnn.h>

using onnxruntime::common::Status;
namespace onnxruntime {
namespace musa {

// Helper function to get the correct mudnn Unary mode for different activation operations
static ::musa::dnn::Unary::Mode GetActivationMode(const std::string &op_name) {
  if (op_name == "Relu") {
    return ::musa::dnn::Unary::Mode::RELU;
  }
  if (op_name == "Tanh") {
    return ::musa::dnn::Unary::Mode::TANH;
  }
  if (op_name == "Sigmoid") {
    return ::musa::dnn::Unary::Mode::SIGMOID;
  }
  if (op_name == "Log") {
    return ::musa::dnn::Unary::Mode::LOG;
  }
  if (op_name == "Softplus") {
    return ::musa::dnn::Unary::Mode::SOFTPLUS;
  }
  // This should not happen as we validate in the calling function
  return ::musa::dnn::Unary::Mode::RELU;
}

// MUSA device-based activation implementation using MusaPreparation and mudnn library
template <typename T>
Status SimpleMusaActivationOp(const MusaPreparation &prepare,
                               const std::string &op_name) {
  // Support Relu, Tanh, Sigmoid, Log operations (LeakyRelu has its own specialized function)
  if (op_name != "Relu" && op_name != "Tanh" && op_name != "Sigmoid" && op_name != "Log" && op_name != "Softplus") {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Unsupported activation operation: " + op_name);
  }

  // Get tensor data from prepared MUSA tensors
  const T *input_data = reinterpret_cast<const T *>(prepare.input_a_ptr);
  T *output_data = reinterpret_cast<T *>(prepare.output_ptr);

  // Validate prepared tensors
  if (!input_data || !output_data) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid prepared tensor pointers");
  }

  if (prepare.inputTensors.empty() || prepare.outputTensors.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid prepared tensor configuration");
  }

  // Use mudnn Unary class for device computation
  try {
    // Create mudnn Unary operation
    ::musa::dnn::Unary unary_op;

    // Set the operation mode based on the activation type
    auto mode = GetActivationMode(op_name);
    auto status = unary_op.SetMode(mode);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set mudnn Unary mode to " + op_name);
    }

    // Softplus requires alpha (beta) and beta (threshold) parameters
    // ONNX Softplus is parameterless: Softplus(x) = log(1 + exp(x))
    // This corresponds to PyTorch defaults: beta=1.0, threshold=20.0
    // In mudnn: SetAlpha sets beta, SetBeta sets threshold
    if (op_name == "Softplus") {
      status = unary_op.SetAlpha(1.0);  // beta parameter
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Failed to set mudnn Unary alpha for Softplus");
      }
      status = unary_op.SetBeta(20.0);  // threshold parameter
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Failed to set mudnn Unary beta for Softplus");
      }
    }

    // Run the unary operation directly on device
    status = unary_op.Run(prepare.GetHandle(),
                          const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]),  // output tensor
                          prepare.inputTensors[0]);  // input tensor

    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "mudnn Unary " + op_name + " operation failed, status: " +
                             std::to_string(static_cast<int>(status)));
    }

  } catch (const std::exception &e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Exception in mudnn Unary operation: " +
                           std::string(e.what()));
  }

  return Status::OK();
}

// Specialized LeakyRelu implementation with alpha parameter
template <typename T>
Status LeakyReluMusaActivationOp(const MusaPreparation &prepare,
                                  float alpha) {
  // Get tensor data from prepared MUSA tensors
  const T *input_data = reinterpret_cast<const T *>(prepare.input_a_ptr);
  T *output_data = reinterpret_cast<T *>(prepare.output_ptr);

  // Validate prepared tensors
  if (!input_data || !output_data) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid prepared tensor pointers");
  }

  if (prepare.inputTensors.empty() || prepare.outputTensors.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid prepared tensor configuration");
  }

  // Use mudnn Unary class for device computation
  try {
    // Create mudnn Unary operation
    ::musa::dnn::Unary unary_op;

    // Set the operation mode to LeakyRelu
    auto status = unary_op.SetMode(::musa::dnn::Unary::Mode::LEAKY_RELU);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set mudnn Unary mode to LeakyRelu");
    }

    // Set the alpha parameter for LeakyRelu
    status = unary_op.SetAlpha(static_cast<double>(alpha));
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set mudnn Unary alpha value for LeakyRelu");
    }

    // Run the unary operation directly on device
    status = unary_op.Run(prepare.GetHandle(),
                          const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]),  // output tensor
                          prepare.inputTensors[0]);  // input tensor

    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "mudnn Unary LeakyRelu operation failed, status: " +
                             std::to_string(static_cast<int>(status)));
    }


  } catch (const std::exception &e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Exception in mudnn Unary LeakyRelu operation: " +
                           std::string(e.what()));
  }

  return Status::OK();
}

template <typename T>
Status Activations<T>::Prepare(OpKernelContext* ctx, MusaPreparation& prepare) const {
  // 1. Get input tensor
  const Tensor* X = ctx->Input<Tensor>(0);
  if (X == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid input tensor");
  }

  // 2. Create output tensor with same shape as input
  Tensor* Y = ctx->Output(0, X->Shape());
  if (Y == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor");
  }

  // 3. Store tensor pointers and shapes in preparation
  prepare.input_a_ptr = X->DataRaw();
  prepare.output_ptr = Y->MutableDataRaw();
  prepare.output_size = Y->Shape().Size();
  prepare.input_a_shape = X->Shape();
  prepare.output_shape = Y->Shape();

  if (prepare.input_a_ptr == nullptr || prepare.output_ptr == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid tensor data pointers");
  }

  // 4. Get MUSA data type
  const auto musaType = GetMusaDataType<T>();

  // 5. Prepare MUSA tensors - using ORT_TRY/ORT_CATCH like other MusaEP operations
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

    // Initialize tensors vectors
    prepare.inputTensors.resize(1);
    prepare.outputTensors.resize(1);

    // Setup input tensor
    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.inputTensors[0], X, musaType, &prepare));

    // Setup output tensor
    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.outputTensors[0], Y, musaType, &prepare));
  }
  ORT_CATCH(const std::exception &e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  return Status::OK();
}

// Macro for registering typed compute function with MUSA implementation
#define REGISTER_MUSA_ACTIVATION_TYPED_COMPUTE(x, T)                           \
  template <> Status x<T>::ComputeInternal(OpKernelContext *ctx) const {       \
    /* Get EP and prepare MUSA operation */                                    \
    const auto* ep = static_cast<const MusaExecutionProvider*>(                \
        Info().GetExecutionProvider());                                         \
    MusaPreparation prepare(ep);                                               \
    ORT_RETURN_IF_ERROR(this->Prepare(ctx, prepare));                          \
                                                                               \
    /* Call MUSA device activation operation using prepared data */            \
    ORT_RETURN_IF_ERROR(                                                       \
        SimpleMusaActivationOp<T>(prepare, #x));                               \
                                                                               \
    return Status::OK();                                                       \
  }

// Macro for registering typed kernel
#define REGISTER_MUSA_ACTIVATION_TYPED_KERNEL_IN_DOMAIN(x, domain, ver, T)     \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                               \
      x, domain, ver, T, kMusaExecutionProvider,                               \
      (*KernelDefBuilder::Create())                                            \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),              \
      x<T>);

#define REGISTER_MUSA_ACTIVATION_TYPED_KERNEL(x, ver, T)                       \
  REGISTER_MUSA_ACTIVATION_TYPED_KERNEL_IN_DOMAIN(x, kOnnxDomain, ver, T)

// Macro for registering versioned typed kernel
#define REGISTER_MUSA_ACTIVATION_VERSIONED_TYPED_KERNEL_IN_DOMAIN(x, domain, startver, endver, T) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                     \
      x, domain, startver, endver, T, kMusaExecutionProvider,                  \
      (*KernelDefBuilder::Create())                                            \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),              \
      x<T>);

#define REGISTER_MUSA_ACTIVATION_VERSIONED_TYPED_KERNEL(x, startver, endver, T) \
  REGISTER_MUSA_ACTIVATION_VERSIONED_TYPED_KERNEL_IN_DOMAIN(x, kOnnxDomain, startver, endver, T)

// Combined macro for both kernel and compute registration
#define REGISTER_MUSA_ACTIVATION_TYPED(name, ver, T)                          \
  REGISTER_MUSA_ACTIVATION_TYPED_KERNEL(name, ver, T)                         \
  REGISTER_MUSA_ACTIVATION_TYPED_COMPUTE(name, T)

// Macro for versioned typed registration (kernel only, no compute)
#define REGISTER_MUSA_ACTIVATION_VERSIONED_TYPED(name, startver, endver, T)   \
  REGISTER_MUSA_ACTIVATION_VERSIONED_TYPED_KERNEL(name, startver, endver, T)

// Register Relu operations for different versions
#define REGISTER_MUSA_ACTIVATION_HFD(name, ver)                              \
  REGISTER_MUSA_ACTIVATION_TYPED(name, ver, MLFloat16)                       \
  REGISTER_MUSA_ACTIVATION_TYPED(name, ver, float)

// Register versioned operations (kernel only, for HFD types)
#define REGISTER_MUSA_ACTIVATION_VERSIONED_HFD(name, startver, endver)       \
  REGISTER_MUSA_ACTIVATION_VERSIONED_TYPED(name, startver, endver, MLFloat16) \
  REGISTER_MUSA_ACTIVATION_VERSIONED_TYPED(name, startver, endver, float)

// Create a specialized compute macro for LeakyRelu that handles the alpha parameter
#define REGISTER_MUSA_LEAKY_RELU_TYPED_COMPUTE(T)                              \
  template <> Status LeakyRelu<T>::ComputeInternal(OpKernelContext *ctx) const { \
    /* Get EP and prepare MUSA operation */                                    \
    const auto* ep = static_cast<const MusaExecutionProvider*>(                \
        Info().GetExecutionProvider());                                         \
    MusaPreparation prepare(ep);                                               \
    ORT_RETURN_IF_ERROR(this->Prepare(ctx, prepare));                          \
                                                                               \
    /* Call specialized MUSA device LeakyRelu operation with alpha parameter */ \
    ORT_RETURN_IF_ERROR(                                                       \
        LeakyReluMusaActivationOp<T>(prepare, alpha_));                         \
                                                                               \
    return Status::OK();                                                       \
  }

// Register Relu operations following ONNX operator versions
REGISTER_MUSA_ACTIVATION_VERSIONED_HFD(Relu, 6, 12)
REGISTER_MUSA_ACTIVATION_VERSIONED_HFD(Relu, 13, 13)
REGISTER_MUSA_ACTIVATION_HFD(Relu, 14)

// Register Tanh operations following ONNX operator versions
REGISTER_MUSA_ACTIVATION_VERSIONED_HFD(Tanh, 6, 12)
REGISTER_MUSA_ACTIVATION_HFD(Tanh, 13)

// Register Sigmoid operations following ONNX operator versions
REGISTER_MUSA_ACTIVATION_VERSIONED_HFD(Sigmoid, 6, 12)
REGISTER_MUSA_ACTIVATION_HFD(Sigmoid, 13)

#ifdef ENABLE_MUSA_NHWC_OPS
REGISTER_MUSA_ACTIVATION_VERSIONED_TYPED_KERNEL_IN_DOMAIN(Relu, kMSInternalNHWCDomain, 6, 12, MLFloat16)
REGISTER_MUSA_ACTIVATION_VERSIONED_TYPED_KERNEL_IN_DOMAIN(Relu, kMSInternalNHWCDomain, 6, 12, float)
REGISTER_MUSA_ACTIVATION_VERSIONED_TYPED_KERNEL_IN_DOMAIN(Relu, kMSInternalNHWCDomain, 13, 13, MLFloat16)
REGISTER_MUSA_ACTIVATION_VERSIONED_TYPED_KERNEL_IN_DOMAIN(Relu, kMSInternalNHWCDomain, 13, 13, float)
REGISTER_MUSA_ACTIVATION_TYPED_KERNEL_IN_DOMAIN(Relu, kMSInternalNHWCDomain, 14, MLFloat16)
REGISTER_MUSA_ACTIVATION_TYPED_KERNEL_IN_DOMAIN(Relu, kMSInternalNHWCDomain, 14, float)

REGISTER_MUSA_ACTIVATION_VERSIONED_TYPED_KERNEL_IN_DOMAIN(Sigmoid, kMSInternalNHWCDomain, 6, 12, MLFloat16)
REGISTER_MUSA_ACTIVATION_VERSIONED_TYPED_KERNEL_IN_DOMAIN(Sigmoid, kMSInternalNHWCDomain, 6, 12, float)
REGISTER_MUSA_ACTIVATION_TYPED_KERNEL_IN_DOMAIN(Sigmoid, kMSInternalNHWCDomain, 13, MLFloat16)
REGISTER_MUSA_ACTIVATION_TYPED_KERNEL_IN_DOMAIN(Sigmoid, kMSInternalNHWCDomain, 13, float)
#endif

// Register LeakyRelu operations following ONNX operator versions
// LeakyRelu requires special handling due to the alpha parameter
REGISTER_MUSA_ACTIVATION_VERSIONED_TYPED_KERNEL(LeakyRelu, 6, 15, MLFloat16)
REGISTER_MUSA_ACTIVATION_VERSIONED_TYPED_KERNEL(LeakyRelu, 6, 15, float)
REGISTER_MUSA_ACTIVATION_TYPED_KERNEL(LeakyRelu, 16, MLFloat16)
REGISTER_MUSA_ACTIVATION_TYPED_KERNEL(LeakyRelu, 16, float)

// Register LeakyRelu compute functions
REGISTER_MUSA_LEAKY_RELU_TYPED_COMPUTE(MLFloat16)
REGISTER_MUSA_LEAKY_RELU_TYPED_COMPUTE(float)

// Register Log operations following ONNX operator versions (v6-12: versioned, v13+: current)
REGISTER_MUSA_ACTIVATION_VERSIONED_HFD(Log, 6, 12)
REGISTER_MUSA_ACTIVATION_HFD(Log, 13)

// Register Softplus operations following ONNX operator versions (opset 1+, no version changes)
REGISTER_MUSA_ACTIVATION_HFD(Softplus, 1)
REGISTER_MUSA_ACTIVATION_TYPED(Softplus, 1, BFloat16)

} // namespace musa
} // namespace onnxruntime
