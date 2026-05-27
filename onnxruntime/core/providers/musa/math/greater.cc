// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/common.h"
#include "core/providers/musa/math/greater.h"
#include "core/providers/musa/musa_execution_provider.h"
#include "core/providers/musa/musa_fwd.h"
#include <musa_runtime.h>
#include <mudnn.h>
#include <string>

using onnxruntime::common::Status;
namespace onnxruntime {
namespace musa {

template <typename T>
Status SimpleMusaGreaterOp(const MusaPreparation& prepare) {
  if (prepare.output_size == 0) {
    return Status::OK();
  }

  if (!prepare.input_a_ptr || !prepare.input_b_ptr || !prepare.output_ptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid prepared tensor pointers");
  }

  if (prepare.inputTensors.size() < 2 || prepare.outputTensors.size() < 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid prepared tensor configuration");
  }

  try {
    ::musa::dnn::Binary binary_op;
    auto status = binary_op.SetMode(::musa::dnn::Binary::Mode::GT);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set mudnn Binary mode to GT");
    }

    status = binary_op.Run(prepare.GetHandle(),
                           const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]),
                           prepare.inputTensors[0],
                           prepare.inputTensors[1]);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "mudnn Binary Greater operation failed, status: " +
                                 std::to_string(static_cast<int>(status)));
    }
  } catch (const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Exception in mudnn Binary Greater operation: " +
                               std::string(e.what()));
  }

  return Status::OK();
}

template <typename T>
template <typename U>
Status Greater<T>::Prepare(OpKernelContext* ctx, MusaPreparation& prepare) const {
  const Tensor* A = ctx->Input<Tensor>(0);
  const Tensor* B = ctx->Input<Tensor>(1);

  if (!A || !B) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid input tensors");
  }

  TensorShape output_shape;
  ORT_RETURN_IF_ERROR(ComputeBroadcastOutputShape(Node().Name(), A->Shape(),
                                                  B->Shape(), output_shape));

  Tensor* C = ctx->Output(0, output_shape);
  if (!C) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor");
  }

  prepare.input_a_ptr = A->DataRaw();
  prepare.input_b_ptr = B->DataRaw();
  prepare.output_ptr = C->MutableDataRaw();
  prepare.output_size = output_shape.Size();
  prepare.input_a_shape = A->Shape();
  prepare.input_b_shape = B->Shape();
  prepare.output_shape = output_shape;

  ORT_TRY {
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

    prepare.inputTensors.resize(2);
    prepare.outputTensors.resize(1);

    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.inputTensors[0], A, GetMusaDataType<T>(), &prepare));
    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.inputTensors[1], B, GetMusaDataType<T>(), &prepare));
    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.outputTensors[0], C, GetMusaDataType<bool>(), &prepare));
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  return Status::OK();
}

#define REGISTER_MUSA_GREATER_TYPED_COMPUTE(T)                         \
  template <>                                                          \
  Status Greater<T>::ComputeInternal(OpKernelContext* ctx) const {     \
    const auto* ep = static_cast<const MusaExecutionProvider*>(        \
        Info().GetExecutionProvider());                                \
    MusaPreparation prepare(ep);                                       \
    ORT_RETURN_IF_ERROR(Prepare<T>(ctx, prepare));                     \
    ORT_RETURN_IF_ERROR(SimpleMusaGreaterOp<T>(prepare));              \
    return Status::OK();                                               \
  }

#define REGISTER_MUSA_GREATER_TYPED_KERNEL(ver, T)                     \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                       \
      Greater, kOnnxDomain, ver, T, kMusaExecutionProvider,            \
      (*KernelDefBuilder::Create())                                    \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())       \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>()),  \
      Greater<T>);

#define REGISTER_MUSA_GREATER_VERSIONED_TYPED(startver, endver, T)     \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                             \
      Greater, kOnnxDomain, startver, endver, T, kMusaExecutionProvider, \
      (*KernelDefBuilder::Create())                                    \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())       \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>()),  \
      Greater<T>);

REGISTER_MUSA_GREATER_VERSIONED_TYPED(7, 8, int8_t)
REGISTER_MUSA_GREATER_VERSIONED_TYPED(7, 8, uint8_t)
REGISTER_MUSA_GREATER_VERSIONED_TYPED(7, 8, int16_t)
REGISTER_MUSA_GREATER_VERSIONED_TYPED(7, 8, int32_t)
REGISTER_MUSA_GREATER_VERSIONED_TYPED(7, 8, int64_t)
REGISTER_MUSA_GREATER_VERSIONED_TYPED(7, 8, MLFloat16)
REGISTER_MUSA_GREATER_VERSIONED_TYPED(7, 8, float)

REGISTER_MUSA_GREATER_VERSIONED_TYPED(9, 12, int8_t)
REGISTER_MUSA_GREATER_VERSIONED_TYPED(9, 12, uint8_t)
REGISTER_MUSA_GREATER_VERSIONED_TYPED(9, 12, int16_t)
REGISTER_MUSA_GREATER_VERSIONED_TYPED(9, 12, int32_t)
REGISTER_MUSA_GREATER_VERSIONED_TYPED(9, 12, int64_t)
REGISTER_MUSA_GREATER_VERSIONED_TYPED(9, 12, MLFloat16)
REGISTER_MUSA_GREATER_VERSIONED_TYPED(9, 12, float)

REGISTER_MUSA_GREATER_TYPED_KERNEL(13, int8_t)
REGISTER_MUSA_GREATER_TYPED_KERNEL(13, uint8_t)
REGISTER_MUSA_GREATER_TYPED_KERNEL(13, int16_t)
REGISTER_MUSA_GREATER_TYPED_KERNEL(13, int32_t)
REGISTER_MUSA_GREATER_TYPED_KERNEL(13, int64_t)
REGISTER_MUSA_GREATER_TYPED_KERNEL(13, MLFloat16)
REGISTER_MUSA_GREATER_TYPED_KERNEL(13, float)

REGISTER_MUSA_GREATER_TYPED_COMPUTE(int8_t)
REGISTER_MUSA_GREATER_TYPED_COMPUTE(uint8_t)
REGISTER_MUSA_GREATER_TYPED_COMPUTE(int16_t)
REGISTER_MUSA_GREATER_TYPED_COMPUTE(int32_t)
REGISTER_MUSA_GREATER_TYPED_COMPUTE(int64_t)
REGISTER_MUSA_GREATER_TYPED_COMPUTE(MLFloat16)
REGISTER_MUSA_GREATER_TYPED_COMPUTE(float)

}  // namespace musa
}  // namespace onnxruntime
