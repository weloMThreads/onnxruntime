// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/math/neg.h"
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/common.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/musa_execution_provider.h"
#include <musa_runtime.h>
#include <mudnn.h>

using onnxruntime::common::Status;
namespace onnxruntime {
namespace musa {

// MUSA device-based neg implementation using mudnn Unary class with MUL mode and alpha=-1
template <typename T>
Status SimpleMusaNegOp(const MusaPreparation& prepare) {
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

  // Use mudnn Unary class for device computation - following torch_musa implementation
  try {
    // Create mudnn Unary operation
    ::musa::dnn::Unary unary_op;

    // Set the operation mode to MUL (multiply by alpha)
    auto status = unary_op.SetMode(::musa::dnn::Unary::Mode::MUL);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set mudnn Unary mode to MUL");
    }

    // Set alpha to -1 to implement negation via multiplication
    status = unary_op.SetAlpha(-1.0);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set mudnn Unary alpha to -1");
    }

    // Run the unary operation directly on device
    status = unary_op.Run(prepare.GetHandle(),
                          const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]),  // output tensor
                          prepare.inputTensors[0]);  // input tensor

    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "mudnn Unary Neg operation failed, status: " +
                             std::to_string(static_cast<int>(status)));
    }

  } catch (const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Exception in mudnn Unary Neg operation: " +
                           std::string(e.what()));
  }

  return Status::OK();
}

// Prepare method - similar pattern to Abs and other unary operations
template <typename T>
Status Neg<T>::Prepare(OpKernelContext* ctx, MusaPreparation& prepare) const {
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
    // In EP mode, stream is already set in PerThreadContext constructor
    // In legacy mode, we need to set stream here
    auto* stream = Stream(ctx);
    if (prepare.handle && stream) {
      auto status = prepare.handle->SetStream(stream);
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Failed to set MUSA stream, status: " +
                               std::to_string(static_cast<int>(status)));
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
Status Neg<T>::ComputeInternal(OpKernelContext* ctx) const {
  const auto* ep = static_cast<const MusaExecutionProvider*>(
      Info().GetExecutionProvider());
  MusaPreparation prepare(ep);
  ORT_RETURN_IF_ERROR(this->Prepare(ctx, prepare));
  ORT_RETURN_IF_ERROR(SimpleMusaNegOp<T>(prepare));
  return Status::OK();
}

// Explicit template instantiations for supported types
// Note: According to user requirements, we exclude double and bfloat16
template class Neg<float>;
template class Neg<MLFloat16>;
template class Neg<int32_t>;
template class Neg<int64_t>;
template class Neg<int8_t>;
template class Neg<uint8_t>;
template class Neg<int16_t>;
template class Neg<uint16_t>;
template class Neg<uint32_t>;
template class Neg<uint64_t>;

// Register kernels for different ONNX versions
#define REGISTER_MUSA_NEG_TYPED_KERNEL(ver, T)                          \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      Neg, kOnnxDomain, ver, T, kMusaExecutionProvider,                 \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),       \
      Neg<T>);

#define REGISTER_MUSA_NEG_VERSIONED_TYPED_KERNEL(startver, endver, T)   \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                              \
      Neg, kOnnxDomain, startver, endver, T, kMusaExecutionProvider,    \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),       \
      Neg<T>);

// Register versions 6-12 (excluding double and bfloat16 as per requirements)
REGISTER_MUSA_NEG_VERSIONED_TYPED_KERNEL(6, 12, float)
REGISTER_MUSA_NEG_VERSIONED_TYPED_KERNEL(6, 12, MLFloat16)
REGISTER_MUSA_NEG_VERSIONED_TYPED_KERNEL(6, 12, int32_t)
REGISTER_MUSA_NEG_VERSIONED_TYPED_KERNEL(6, 12, int64_t)
REGISTER_MUSA_NEG_VERSIONED_TYPED_KERNEL(6, 12, int8_t)
REGISTER_MUSA_NEG_VERSIONED_TYPED_KERNEL(6, 12, uint8_t)
REGISTER_MUSA_NEG_VERSIONED_TYPED_KERNEL(6, 12, int16_t)
REGISTER_MUSA_NEG_VERSIONED_TYPED_KERNEL(6, 12, uint16_t)
REGISTER_MUSA_NEG_VERSIONED_TYPED_KERNEL(6, 12, uint32_t)
REGISTER_MUSA_NEG_VERSIONED_TYPED_KERNEL(6, 12, uint64_t)

// Register version 13+ (current, excluding double and bfloat16)
REGISTER_MUSA_NEG_TYPED_KERNEL(13, float)
REGISTER_MUSA_NEG_TYPED_KERNEL(13, MLFloat16)
REGISTER_MUSA_NEG_TYPED_KERNEL(13, int32_t)
REGISTER_MUSA_NEG_TYPED_KERNEL(13, int64_t)
REGISTER_MUSA_NEG_TYPED_KERNEL(13, int8_t)
REGISTER_MUSA_NEG_TYPED_KERNEL(13, uint8_t)
REGISTER_MUSA_NEG_TYPED_KERNEL(13, int16_t)
REGISTER_MUSA_NEG_TYPED_KERNEL(13, uint16_t)
REGISTER_MUSA_NEG_TYPED_KERNEL(13, uint32_t)
REGISTER_MUSA_NEG_TYPED_KERNEL(13, uint64_t)

}  // namespace musa
}  // namespace onnxruntime