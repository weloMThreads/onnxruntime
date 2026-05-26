// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/math/erf.h"
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/common.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/musa_execution_provider.h"
#include <musa_runtime.h>
#include <mudnn.h>

using onnxruntime::common::Status;
namespace onnxruntime {
namespace musa {

// MUSA device-based erf implementation using mudnn Unary class
template <typename T>
Status SimpleMusaErfOp(const MusaPreparation& prepare) {
  // Get tensor data from prepared MUSA tensors
  const T* input_data = reinterpret_cast<const T*>(prepare.input_a_ptr);
  T* output_data = reinterpret_cast<T*>(prepare.output_ptr);

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

    // Set the operation mode to ERF
    auto status = unary_op.SetMode(::musa::dnn::Unary::Mode::ERF);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set mudnn Unary mode to ERF");
    }

    // Run the unary operation directly on device
    status = unary_op.Run(prepare.GetHandle(),
                          const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]),  // output tensor
                          prepare.inputTensors[0]);  // input tensor

    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "mudnn Unary Erf operation failed, status: " +
                             std::to_string(static_cast<int>(status)));
    }

  } catch (const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Exception in mudnn Unary Erf operation: " +
                           std::string(e.what()));
  }

  return Status::OK();
}

// Prepare method - reuse similar pattern from sqrt
template <typename T>
Status Erf<T>::Prepare(OpKernelContext* ctx, MusaPreparation& prepare) const {
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

// ComputeInternal method
template <typename T>
Status Erf<T>::ComputeInternal(OpKernelContext* ctx) const {
  const auto* ep = static_cast<const MusaExecutionProvider*>(
      Info().GetExecutionProvider());
  MusaPreparation prepare(ep);
  ORT_RETURN_IF_ERROR(this->Prepare(ctx, prepare));
  ORT_RETURN_IF_ERROR(SimpleMusaErfOp<T>(prepare));
  return Status::OK();
}

// Explicit template instantiations for supported types
template class Erf<float>;
template class Erf<MLFloat16>;
template class Erf<BFloat16>;

// Register kernels for different ONNX versions
#define REGISTER_MUSA_ERF_TYPED_KERNEL(ver, T)                          \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      Erf, kOnnxDomain, ver, T, kMusaExecutionProvider,                 \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),       \
      Erf<T>);

#define REGISTER_MUSA_ERF_VERSIONED_TYPED_KERNEL(startver, endver, T)   \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                              \
      Erf, kOnnxDomain, startver, endver, T, kMusaExecutionProvider,    \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),       \
      Erf<T>);

// Register versions 9-12
REGISTER_MUSA_ERF_VERSIONED_TYPED_KERNEL(9, 12, float)
REGISTER_MUSA_ERF_VERSIONED_TYPED_KERNEL(9, 12, MLFloat16)

// Register version 13+ (current) - adds BFloat16 support
REGISTER_MUSA_ERF_TYPED_KERNEL(13, float)
REGISTER_MUSA_ERF_TYPED_KERNEL(13, MLFloat16)
REGISTER_MUSA_ERF_TYPED_KERNEL(13, BFloat16)

}  // namespace musa
}  // namespace onnxruntime
