// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/common.h"
#include "core/providers/musa/math/round.h"
#include "core/providers/musa/musa_execution_provider.h"
#include "core/providers/musa/musa_fwd.h"
#include <musa_runtime.h>
#include <mudnn.h>
#include <string>

using onnxruntime::common::Status;
namespace onnxruntime {
namespace musa {

template <typename T>
Status SimpleMusaRoundOp(const MusaPreparation& prepare) {
  if (prepare.output_size == 0) {
    return Status::OK();
  }

  if (!prepare.input_a_ptr || !prepare.output_ptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid prepared tensor pointers");
  }

  if (prepare.inputTensors.empty() || prepare.outputTensors.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid prepared tensor configuration");
  }

  try {
    ::musa::dnn::Unary unary_op;
    auto status = unary_op.SetMode(::musa::dnn::Unary::Mode::ROUND);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set mudnn Unary mode to ROUND");
    }

    status = unary_op.Run(prepare.GetHandle(),
                          const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]),
                          prepare.inputTensors[0]);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "mudnn Unary Round operation failed, status: " +
                                 std::to_string(static_cast<int>(status)));
    }
  } catch (const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Exception in mudnn Unary Round operation: " +
                               std::string(e.what()));
  }

  return Status::OK();
}

template <typename T>
Status Round<T>::Prepare(OpKernelContext* ctx, MusaPreparation& prepare) const {
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

  if (prepare.output_size > 0 && (!prepare.input_a_ptr || !prepare.output_ptr)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid tensor data pointers");
  }

  ORT_TRY {
    if (prepare.handle) {
      auto* stream = Stream(ctx);
      if (stream) {
        auto status = prepare.handle->SetStream(stream);
        if (status != ::musa::dnn::Status::SUCCESS) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                 "Failed to set MUSA stream, status: " +
                                     std::to_string(static_cast<int>(status)));
        }
      }
    }

    prepare.inputTensors.resize(1);
    prepare.outputTensors.resize(1);
    const auto musa_type = GetMusaDataType<T>();
    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.inputTensors[0], X, musa_type, &prepare));
    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.outputTensors[0], Y, musa_type, &prepare));
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  return Status::OK();
}

template <typename T>
Status Round<T>::ComputeInternal(OpKernelContext* ctx) const {
  const auto* ep = static_cast<const MusaExecutionProvider*>(Info().GetExecutionProvider());
  MusaPreparation prepare(ep);
  ORT_RETURN_IF_ERROR(Prepare(ctx, prepare));
  return SimpleMusaRoundOp<T>(prepare);
}

template class Round<MLFloat16>;
template class Round<float>;

#define REGISTER_MUSA_ROUND_TYPED_KERNEL(ver, T)                             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                             \
      Round, kOnnxDomain, ver, T, kMusaExecutionProvider,                    \
      (*KernelDefBuilder::Create())                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),            \
      Round<T>);

#define REGISTER_MUSA_ROUND_VERSIONED_TYPED_KERNEL(startver, endver, T)      \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                   \
      Round, kOnnxDomain, startver, endver, T, kMusaExecutionProvider,       \
      (*KernelDefBuilder::Create())                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),            \
      Round<T>);

REGISTER_MUSA_ROUND_VERSIONED_TYPED_KERNEL(11, 21, MLFloat16)
REGISTER_MUSA_ROUND_VERSIONED_TYPED_KERNEL(11, 21, float)

REGISTER_MUSA_ROUND_TYPED_KERNEL(22, MLFloat16)
REGISTER_MUSA_ROUND_TYPED_KERNEL(22, float)

}  // namespace musa
}  // namespace onnxruntime
