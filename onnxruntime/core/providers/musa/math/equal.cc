// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/common.h"
#include "core/providers/musa/math/equal.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/musa_execution_provider.h"
#include <musa_runtime.h>
#include <mudnn.h>
#include <string>
#include <vector>

using onnxruntime::common::Status;
namespace onnxruntime {
namespace musa {

// MUSA device-based implementation using MusaPreparation and mudnn library
template <typename T>
Status SimpleMusaEqualOp(const MusaPreparation& prepare, size_t size) {
  // Get tensor data from prepared MUSA tensors
  const T* input_a = reinterpret_cast<const T*>(prepare.input_a_ptr);
  const T* input_b = reinterpret_cast<const T*>(prepare.input_b_ptr);
  bool* output = reinterpret_cast<bool*>(prepare.output_ptr);

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

    // Set the operation mode to EQ (Equal)
    auto status = binary_op.SetMode(::musa::dnn::Binary::Mode::EQ);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set mudnn Binary mode to EQ");
    }

    // Run the binary operation directly on device with automatic broadcasting
    status = binary_op.Run(prepare.GetHandle(),
                          const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]),  // output tensor
                          prepare.inputTensors[0],   // input A tensor
                          prepare.inputTensors[1]);  // input B tensor

    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "mudnn Binary Equal operation failed, status: " +
                             std::to_string(static_cast<int>(status)));
    }

  } catch (const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Exception in mudnn Binary Equal operation: " +
                           std::string(e.what()));
  }

  return Status::OK();
}

template <typename T>
template <typename U>
Status Equal<T>::Prepare(OpKernelContext* ctx, MusaPreparation& prepare) const {
  // 1. Get input tensors A and B
  const Tensor* A = ctx->Input<Tensor>(0);
  const Tensor* B = ctx->Input<Tensor>(1);

  if (!A || !B) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid input tensors");
  }

  // 2. Check if shapes are broadcastable and compute output shape
  TensorShape output_shape;
  ORT_RETURN_IF_ERROR(ComputeBroadcastOutputShape(Node().Name(), A->Shape(),
                                                 B->Shape(), output_shape));

  // 3. Create output tensor - Equal always outputs bool type
  Tensor* C = ctx->Output(0, output_shape);
  if (!C) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor");
  }

  // 4. Store basic tensor info
  prepare.input_a_ptr = A->DataRaw();
  prepare.input_b_ptr = B->DataRaw();
  prepare.output_ptr = C->MutableDataRaw();
  prepare.output_size = output_shape.Size();
  prepare.input_a_shape = A->Shape();
  prepare.input_b_shape = B->Shape();
  prepare.output_shape = output_shape;

  // 5. Setup MUSA tensors
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

    prepare.inputTensors.resize(2);
    prepare.outputTensors.resize(1);

    // Setup input tensors with their actual data types
    const auto input_musa_type = GetMusaDataType<T>();
    const auto output_musa_type = GetMusaDataType<bool>();

    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.inputTensors[0], A, input_musa_type, &prepare));
    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.inputTensors[1], B, input_musa_type, &prepare));
    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.outputTensors[0], C, output_musa_type, &prepare));
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  return Status::OK();
}

// Macro for registering typed compute function
#define REGISTER_MUSA_EQUAL_TYPED_COMPUTE(T)                                   \
  template <>                                                                  \
  Status Equal<T>::ComputeInternal(OpKernelContext* ctx) const {               \
    const auto* ep = static_cast<const MusaExecutionProvider*>(                \
        Info().GetExecutionProvider());                                        \
    MusaPreparation prepare(ep);                                               \
    ORT_RETURN_IF_ERROR(Prepare<T>(ctx, prepare));                            \
    ORT_RETURN_IF_ERROR(SimpleMusaEqualOp<T>(prepare, prepare.output_size));  \
    return Status::OK();                                                       \
  }

// Macro for registering typed kernel
#define REGISTER_MUSA_EQUAL_TYPED_KERNEL(ver, T)                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                               \
      Equal, kOnnxDomain, ver, T, kMusaExecutionProvider,                      \
      (*KernelDefBuilder::Create())                                            \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())               \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>()),          \
      Equal<T>);

// Macro for registering versioned typed kernel
#define REGISTER_MUSA_EQUAL_VERSIONED_TYPED_KERNEL(startver, endver, T)        \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                     \
      Equal, kOnnxDomain, startver, endver, T, kMusaExecutionProvider,         \
      (*KernelDefBuilder::Create())                                            \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())               \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>()),          \
      Equal<T>);

// Combined macro for both kernel and compute registration
#define REGISTER_MUSA_EQUAL_TYPED(ver, T)                                      \
  REGISTER_MUSA_EQUAL_TYPED_KERNEL(ver, T)                                     \
  REGISTER_MUSA_EQUAL_TYPED_COMPUTE(T)

// Combined macro for versioned registration (kernel only, no compute)
#define REGISTER_MUSA_EQUAL_VERSIONED_TYPED(startver, endver, T)               \
  REGISTER_MUSA_EQUAL_VERSIONED_TYPED_KERNEL(startver, endver, T)

// Register Equal operations for supported data types
// According to the research docs, we should support multiple input types but avoid double and bfloat16
// NOTE: Removed uint16_t, uint32_t, uint64_t due to MUDNN Binary Equal NOT_SUPPORTED error (status: 4)
REGISTER_MUSA_EQUAL_VERSIONED_TYPED(1, 6, bool)
REGISTER_MUSA_EQUAL_VERSIONED_TYPED(1, 6, int8_t)
REGISTER_MUSA_EQUAL_VERSIONED_TYPED(1, 6, uint8_t)
REGISTER_MUSA_EQUAL_VERSIONED_TYPED(1, 6, int16_t)
// REGISTER_MUSA_EQUAL_VERSIONED_TYPED(1, 6, uint16_t)  // MUDNN NOT_SUPPORTED
REGISTER_MUSA_EQUAL_VERSIONED_TYPED(1, 6, int32_t)
// REGISTER_MUSA_EQUAL_VERSIONED_TYPED(1, 6, uint32_t)  // MUDNN NOT_SUPPORTED
REGISTER_MUSA_EQUAL_VERSIONED_TYPED(1, 6, int64_t)
// REGISTER_MUSA_EQUAL_VERSIONED_TYPED(1, 6, uint64_t)  // MUDNN NOT_SUPPORTED
REGISTER_MUSA_EQUAL_VERSIONED_TYPED(1, 6, MLFloat16)
REGISTER_MUSA_EQUAL_VERSIONED_TYPED(1, 6, float)

REGISTER_MUSA_EQUAL_VERSIONED_TYPED(7, 10, bool)
REGISTER_MUSA_EQUAL_VERSIONED_TYPED(7, 10, int8_t)
REGISTER_MUSA_EQUAL_VERSIONED_TYPED(7, 10, uint8_t)
REGISTER_MUSA_EQUAL_VERSIONED_TYPED(7, 10, int16_t)
// REGISTER_MUSA_EQUAL_VERSIONED_TYPED(7, 10, uint16_t)  // MUDNN NOT_SUPPORTED
REGISTER_MUSA_EQUAL_VERSIONED_TYPED(7, 10, int32_t)
// REGISTER_MUSA_EQUAL_VERSIONED_TYPED(7, 10, uint32_t)  // MUDNN NOT_SUPPORTED
REGISTER_MUSA_EQUAL_VERSIONED_TYPED(7, 10, int64_t)
// REGISTER_MUSA_EQUAL_VERSIONED_TYPED(7, 10, uint64_t)  // MUDNN NOT_SUPPORTED
REGISTER_MUSA_EQUAL_VERSIONED_TYPED(7, 10, MLFloat16)
REGISTER_MUSA_EQUAL_VERSIONED_TYPED(7, 10, float)

REGISTER_MUSA_EQUAL_VERSIONED_TYPED(11, 12, bool)
REGISTER_MUSA_EQUAL_VERSIONED_TYPED(11, 12, int8_t)
REGISTER_MUSA_EQUAL_VERSIONED_TYPED(11, 12, uint8_t)
REGISTER_MUSA_EQUAL_VERSIONED_TYPED(11, 12, int16_t)
// REGISTER_MUSA_EQUAL_VERSIONED_TYPED(11, 12, uint16_t)  // MUDNN NOT_SUPPORTED
REGISTER_MUSA_EQUAL_VERSIONED_TYPED(11, 12, int32_t)
// REGISTER_MUSA_EQUAL_VERSIONED_TYPED(11, 12, uint32_t)  // MUDNN NOT_SUPPORTED
REGISTER_MUSA_EQUAL_VERSIONED_TYPED(11, 12, int64_t)
// REGISTER_MUSA_EQUAL_VERSIONED_TYPED(11, 12, uint64_t)  // MUDNN NOT_SUPPORTED
REGISTER_MUSA_EQUAL_VERSIONED_TYPED(11, 12, MLFloat16)
REGISTER_MUSA_EQUAL_VERSIONED_TYPED(11, 12, float)

// Register typed kernels for later versions (kernel only)
#define REGISTER_MUSA_EQUAL_TYPED_KERNEL_ONLY(ver, T) \
  REGISTER_MUSA_EQUAL_TYPED_KERNEL(ver, T)

REGISTER_MUSA_EQUAL_TYPED_KERNEL_ONLY(13, bool)
REGISTER_MUSA_EQUAL_TYPED_KERNEL_ONLY(13, int8_t)
REGISTER_MUSA_EQUAL_TYPED_KERNEL_ONLY(13, uint8_t)
REGISTER_MUSA_EQUAL_TYPED_KERNEL_ONLY(13, int16_t)
// REGISTER_MUSA_EQUAL_TYPED_KERNEL_ONLY(13, uint16_t)  // MUDNN NOT_SUPPORTED
REGISTER_MUSA_EQUAL_TYPED_KERNEL_ONLY(13, int32_t)
// REGISTER_MUSA_EQUAL_TYPED_KERNEL_ONLY(13, uint32_t)  // MUDNN NOT_SUPPORTED
REGISTER_MUSA_EQUAL_TYPED_KERNEL_ONLY(13, int64_t)
// REGISTER_MUSA_EQUAL_TYPED_KERNEL_ONLY(13, uint64_t)  // MUDNN NOT_SUPPORTED
REGISTER_MUSA_EQUAL_TYPED_KERNEL_ONLY(13, MLFloat16)
REGISTER_MUSA_EQUAL_TYPED_KERNEL_ONLY(13, float)

REGISTER_MUSA_EQUAL_TYPED_KERNEL_ONLY(19, bool)
REGISTER_MUSA_EQUAL_TYPED_KERNEL_ONLY(19, int8_t)
REGISTER_MUSA_EQUAL_TYPED_KERNEL_ONLY(19, uint8_t)
REGISTER_MUSA_EQUAL_TYPED_KERNEL_ONLY(19, int16_t)
// REGISTER_MUSA_EQUAL_TYPED_KERNEL_ONLY(19, uint16_t)  // MUDNN NOT_SUPPORTED
REGISTER_MUSA_EQUAL_TYPED_KERNEL_ONLY(19, int32_t)
// REGISTER_MUSA_EQUAL_TYPED_KERNEL_ONLY(19, uint32_t)  // MUDNN NOT_SUPPORTED
REGISTER_MUSA_EQUAL_TYPED_KERNEL_ONLY(19, int64_t)
// REGISTER_MUSA_EQUAL_TYPED_KERNEL_ONLY(19, uint64_t)  // MUDNN NOT_SUPPORTED
REGISTER_MUSA_EQUAL_TYPED_KERNEL_ONLY(19, MLFloat16)
REGISTER_MUSA_EQUAL_TYPED_KERNEL_ONLY(19, float)

// Register ComputeInternal implementations (only once per type)
REGISTER_MUSA_EQUAL_TYPED_COMPUTE(bool)
REGISTER_MUSA_EQUAL_TYPED_COMPUTE(int8_t)
REGISTER_MUSA_EQUAL_TYPED_COMPUTE(uint8_t)
REGISTER_MUSA_EQUAL_TYPED_COMPUTE(int16_t)
// REGISTER_MUSA_EQUAL_TYPED_COMPUTE(uint16_t)  // MUDNN NOT_SUPPORTED
REGISTER_MUSA_EQUAL_TYPED_COMPUTE(int32_t)
// REGISTER_MUSA_EQUAL_TYPED_COMPUTE(uint32_t)  // MUDNN NOT_SUPPORTED
REGISTER_MUSA_EQUAL_TYPED_COMPUTE(int64_t)
// REGISTER_MUSA_EQUAL_TYPED_COMPUTE(uint64_t)  // MUDNN NOT_SUPPORTED
REGISTER_MUSA_EQUAL_TYPED_COMPUTE(MLFloat16)
REGISTER_MUSA_EQUAL_TYPED_COMPUTE(float)

}  // namespace musa
}  // namespace onnxruntime