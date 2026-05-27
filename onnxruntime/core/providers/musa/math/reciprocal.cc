// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/math/reciprocal.h"

#include "core/providers/common.h"
#include "core/providers/musa/musa_execution_provider.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/shared_library/provider_api.h"

#include <mudnn.h>
#include <musa_runtime.h>

using onnxruntime::common::Status;

namespace onnxruntime {
namespace musa {

template <typename T>
Status SimpleMusaReciprocalOp(const MusaPreparation& prepare) {
  if (prepare.output_size == 0) {
    return Status::OK();
  }

  if (prepare.input_a_ptr == nullptr || prepare.output_ptr == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid prepared tensor pointers");
  }

  if (prepare.inputTensors.empty() || prepare.outputTensors.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid prepared tensor configuration");
  }

  ORT_TRY {
    ::musa::dnn::Unary unary_op;
    auto status = unary_op.SetMode(::musa::dnn::Unary::Mode::RECIPROCAL);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set mudnn Unary mode to RECIPROCAL, status: " +
                                 std::to_string(static_cast<int>(status)));
    }

    status = unary_op.Run(prepare.GetHandle(),
                          const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]),
                          prepare.inputTensors[0]);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "mudnn Unary Reciprocal operation failed, status: " +
                                 std::to_string(static_cast<int>(status)));
    }
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Exception in mudnn Unary Reciprocal operation: " +
                               std::string(e.what()));
  }

  return Status::OK();
}

template <typename T>
Status Reciprocal<T>::Prepare(OpKernelContext* ctx, MusaPreparation& prepare) const {
  const Tensor* X = ctx->Input<Tensor>(0);
  if (X == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input tensor");
  }

  Tensor* Y = ctx->Output(0, X->Shape());
  if (Y == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor");
  }

  prepare.input_a_ptr = X->DataRaw();
  prepare.output_ptr = Y->MutableDataRaw();
  prepare.output_size = Y->Shape().Size();
  prepare.input_a_shape = X->Shape();
  prepare.output_shape = Y->Shape();

  if (prepare.output_size == 0) {
    return Status::OK();
  }

  if (prepare.input_a_ptr == nullptr || prepare.output_ptr == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid tensor data pointers");
  }

  const auto musa_type = GetMusaDataType<T>();

  ORT_TRY {
    auto* stream = Stream(ctx);
    if (stream != nullptr) {
      auto status = prepare.GetHandle().SetStream(stream);
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Failed to set MUSA stream, status: " +
                                   std::to_string(static_cast<int>(status)));
      }
    }

    prepare.inputTensors.resize(1);
    prepare.outputTensors.resize(1);

    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.inputTensors[0], X, musa_type, &prepare));
    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.outputTensors[0], Y, musa_type, &prepare));
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  return Status::OK();
}

template <typename T>
Status Reciprocal<T>::ComputeInternal(OpKernelContext* ctx) const {
  const auto* ep = static_cast<const MusaExecutionProvider*>(Info().GetExecutionProvider());
  MusaPreparation prepare(ep);
  ORT_RETURN_IF_ERROR(this->Prepare(ctx, prepare));
  ORT_RETURN_IF_ERROR(SimpleMusaReciprocalOp<T>(prepare));
  return Status::OK();
}

template class Reciprocal<float>;
template class Reciprocal<double>;
template class Reciprocal<MLFloat16>;
template class Reciprocal<BFloat16>;

#define REGISTER_MUSA_RECIPROCAL_TYPED_KERNEL(ver, T)                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                         \
      Reciprocal, kOnnxDomain, ver, T, kMusaExecutionProvider,           \
      (*KernelDefBuilder::Create())                                      \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),        \
      Reciprocal<T>);

#define REGISTER_MUSA_RECIPROCAL_VERSIONED_TYPED_KERNEL(startver, endver, T) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                   \
      Reciprocal, kOnnxDomain, startver, endver, T, kMusaExecutionProvider,  \
      (*KernelDefBuilder::Create())                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),            \
      Reciprocal<T>);

REGISTER_MUSA_RECIPROCAL_VERSIONED_TYPED_KERNEL(6, 12, float)
REGISTER_MUSA_RECIPROCAL_VERSIONED_TYPED_KERNEL(6, 12, double)
REGISTER_MUSA_RECIPROCAL_VERSIONED_TYPED_KERNEL(6, 12, MLFloat16)

REGISTER_MUSA_RECIPROCAL_TYPED_KERNEL(13, float)
REGISTER_MUSA_RECIPROCAL_TYPED_KERNEL(13, double)
REGISTER_MUSA_RECIPROCAL_TYPED_KERNEL(13, MLFloat16)
REGISTER_MUSA_RECIPROCAL_TYPED_KERNEL(13, BFloat16)

}  // namespace musa
}  // namespace onnxruntime
