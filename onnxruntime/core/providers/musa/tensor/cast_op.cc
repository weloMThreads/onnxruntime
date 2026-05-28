// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/common.h"
#include "core/providers/musa/tensor/cast_op.h"
#include "core/providers/musa/musa_allocator.h"
#include "core/providers/musa/musa_utils.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/musa_execution_provider.h"
#include "core/providers/musa/musa_call.h"
#include "mudnn.h"

#include <string>
#include <vector>

namespace onnxruntime {
namespace musa {

template <typename SrcT>
Status Cast<SrcT>::Prepare(OpKernelContext* ctx, MusaPreparation& prepare) const {
  const Tensor* input_tensor = ctx->Input<Tensor>(0);
  Tensor* output_tensor = ctx->Output<Tensor>(0);

  ORT_ENFORCE(input_tensor != nullptr, "Input tensor is null");
  ORT_ENFORCE(output_tensor != nullptr, "Output tensor is null");

  // Set up tensor pointers
  prepare.input_a_ptr = input_tensor->DataRaw();
  prepare.output_ptr = output_tensor->MutableDataRaw();
  prepare.output_size = output_tensor->SizeInBytes();

  // Store tensor shapes
  prepare.input_a_shape = input_tensor->Shape();
  prepare.output_shape = output_tensor->Shape();
  
  return Status::OK();
}

template <typename SrcT>
Status Cast<SrcT>::ComputeInternal(OpKernelContext* context) const {
  // Get input tensor
  const Tensor* input = context->Input<Tensor>(0);
  if (!input) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Cast: input tensor is null");
  }

  // Create output tensor with same shape as input but different data type
  TensorShape output_shape = input->Shape();
  Tensor* output = context->Output(0, output_shape);
  if (!output) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Cast: Failed to create output tensor");
  }

  try {
    // Check if this is a same-type conversion
    auto input_type = GetMusaDataType<SrcT>();
    auto output_type = GetMusaDataTypeFromOnnx(to_);
    
    if (input_type == output_type) {
      // Same type conversion - just copy the data (MUSA doesn't support same-type cast)
      auto stream = Stream(context);
      
      musaError_t copy_result = musaMemcpyAsync(output->MutableDataRaw(), 
                                                input->DataRaw(), 
                                                input->SizeInBytes(), 
                                                musaMemcpyDeviceToDevice, 
                                                stream);
      if (copy_result != musaSuccess) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "MUSA memory copy failed with error: " +
                               std::to_string(static_cast<int>(copy_result)));
      }
    } else {
      // Different type conversion - use MUSA Cast operation
      const auto* ep = static_cast<const MusaExecutionProvider*>(
          Info().GetExecutionProvider());
      MusaPreparation prepare(ep);
      
      ::musa::dnn::Unary cast_op;
      auto mode_status = cast_op.SetMode(::musa::dnn::Unary::Mode::CAST);
      if (mode_status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set CAST mode for Unary operation");
      }

      ::musa::dnn::Tensor input_mu_tensor;
      ::musa::dnn::Tensor output_mu_tensor;

      // Set up MUSA tensors
      ORT_RETURN_IF_ERROR(SetupMusaTensor(input_mu_tensor, input, input_type, &prepare));
      ORT_RETURN_IF_ERROR(SetupMusaTensor(output_mu_tensor, output, output_type, &prepare));
      
      // Execute Cast operation
      auto status = cast_op.Run(prepare.GetHandle(), output_mu_tensor, input_mu_tensor);
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "MUSA Cast operation failed with status: " +
                               std::to_string(static_cast<int>(status)));
      }
    }

  } catch (const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Exception in MUSA Cast operation: " + std::string(e.what()));
  }

  return Status::OK();
}

Status CastStringToInt32::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  ORT_RETURN_IF_NOT(input != nullptr, "CastStringToInt32: input tensor is null");

  Tensor* output = context->Output(0, input->Shape());
  ORT_RETURN_IF_NOT(output != nullptr, "CastStringToInt32: output tensor is null");

  const int64_t count = input->Shape().Size();
  if (count == 0) {
    return Status::OK();
  }

  const auto* input_data = reinterpret_cast<const std::string*>(input->DataRaw());
  std::vector<int32_t> host_output(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    try {
      host_output[static_cast<size_t>(i)] = gsl::narrow_cast<int32_t>(std::stoll(input_data[i]));
    } catch (const std::exception& e) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "CastStringToInt32: failed to parse input string: ",
                             input_data[i], ", error: ", e.what());
    }
  }

  auto stream = Stream(context);
  MUSA_RETURN_IF_ERROR(musaMemcpyAsync(output->MutableData<int32_t>(),
                                       host_output.data(),
                                       host_output.size() * sizeof(int32_t),
                                       musaMemcpyHostToDevice,
                                       stream));
  MUSA_RETURN_IF_ERROR(musaStreamSynchronize(stream));
  return Status::OK();
}

// Helper function to get type constraints for Cast operation
const std::vector<MLDataType>& CastOpTypeConstraints() {
  // Must be done as a local static for a shared provider
  static std::vector<MLDataType> types{
      DataTypeImpl::GetTensorType<MLFloat16>(),
      DataTypeImpl::GetTensorType<float>(),
      DataTypeImpl::GetTensorType<double>(),
      DataTypeImpl::GetTensorType<int8_t>(),
      DataTypeImpl::GetTensorType<int16_t>(),
      DataTypeImpl::GetTensorType<int32_t>(),
      DataTypeImpl::GetTensorType<int64_t>(),
      DataTypeImpl::GetTensorType<uint8_t>(),
      DataTypeImpl::GetTensorType<uint16_t>(),
      DataTypeImpl::GetTensorType<uint32_t>(),
      DataTypeImpl::GetTensorType<uint64_t>(),
      DataTypeImpl::GetTensorType<bool>()
  };
  return types;
}

// Macro for registering Cast kernel
#define REGISTER_MUSA_CAST_VERSIONED_TYPED_KERNEL(ver_start, ver_end, T) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                               \
      Cast,                                                               \
      kOnnxDomain,                                                        \
      ver_start,                                                          \
      ver_end,                                                            \
      T,                                                                  \
      kMusaExecutionProvider,                                            \
      (*KernelDefBuilder::Create())                                      \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("T2", CastOpTypeConstraints()),                \
      Cast<T>);

#define REGISTER_MUSA_CAST_TYPED_KERNEL(ver, T) \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                \
      Cast,                                      \
      kOnnxDomain,                               \
      ver,                                       \
      T,                                         \
      kMusaExecutionProvider,                   \
      (*KernelDefBuilder::Create())             \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()) \
          .TypeConstraint("T2", CastOpTypeConstraints()),          \
      Cast<T>);

#define REGISTER_MUSA_CAST_STRING_TO_INT32_VERSIONED_KERNEL(ver_start, ver_end)              \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                   \
      Cast, kOnnxDomain, ver_start, ver_end, string, kMusaExecutionProvider,                 \
      (*KernelDefBuilder::Create())                                                         \
          .TypeConstraint("T1", DataTypeImpl::GetTensorTypeFromOnnxType(ONNX_NAMESPACE::TensorProto_DataType_STRING))                \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<int32_t>())                    \
          .InputMemoryType(OrtMemTypeCPUInput, 0),                                          \
      CastStringToInt32);
// Register Cast operators (version 6-8)
REGISTER_MUSA_CAST_VERSIONED_TYPED_KERNEL(6, 8, int32_t)
REGISTER_MUSA_CAST_VERSIONED_TYPED_KERNEL(6, 8, int64_t)
REGISTER_MUSA_CAST_VERSIONED_TYPED_KERNEL(6, 8, MLFloat16)
REGISTER_MUSA_CAST_VERSIONED_TYPED_KERNEL(6, 8, float)
REGISTER_MUSA_CAST_VERSIONED_TYPED_KERNEL(6, 8, int8_t)
REGISTER_MUSA_CAST_VERSIONED_TYPED_KERNEL(6, 8, int16_t)
REGISTER_MUSA_CAST_VERSIONED_TYPED_KERNEL(6, 8, uint8_t)
REGISTER_MUSA_CAST_VERSIONED_TYPED_KERNEL(6, 8, uint16_t)
REGISTER_MUSA_CAST_VERSIONED_TYPED_KERNEL(6, 8, uint32_t)
REGISTER_MUSA_CAST_VERSIONED_TYPED_KERNEL(6, 8, uint64_t)
REGISTER_MUSA_CAST_VERSIONED_TYPED_KERNEL(6, 8, bool)

// Register Cast operators (version 9-12)
REGISTER_MUSA_CAST_VERSIONED_TYPED_KERNEL(9, 12, int32_t)
REGISTER_MUSA_CAST_VERSIONED_TYPED_KERNEL(9, 12, int64_t)
REGISTER_MUSA_CAST_VERSIONED_TYPED_KERNEL(9, 12, MLFloat16)
REGISTER_MUSA_CAST_VERSIONED_TYPED_KERNEL(9, 12, float)
REGISTER_MUSA_CAST_VERSIONED_TYPED_KERNEL(9, 12, int8_t)
REGISTER_MUSA_CAST_VERSIONED_TYPED_KERNEL(9, 12, int16_t)
REGISTER_MUSA_CAST_VERSIONED_TYPED_KERNEL(9, 12, uint8_t)
REGISTER_MUSA_CAST_VERSIONED_TYPED_KERNEL(9, 12, uint16_t)
REGISTER_MUSA_CAST_VERSIONED_TYPED_KERNEL(9, 12, uint32_t)
REGISTER_MUSA_CAST_VERSIONED_TYPED_KERNEL(9, 12, uint64_t)
REGISTER_MUSA_CAST_VERSIONED_TYPED_KERNEL(9, 12, bool)

// Register Cast operators (version 13-18)
REGISTER_MUSA_CAST_VERSIONED_TYPED_KERNEL(13, 18, int32_t)
REGISTER_MUSA_CAST_VERSIONED_TYPED_KERNEL(13, 18, int64_t)
REGISTER_MUSA_CAST_VERSIONED_TYPED_KERNEL(13, 18, MLFloat16)
REGISTER_MUSA_CAST_VERSIONED_TYPED_KERNEL(13, 18, float)
REGISTER_MUSA_CAST_VERSIONED_TYPED_KERNEL(13, 18, int8_t)
REGISTER_MUSA_CAST_VERSIONED_TYPED_KERNEL(13, 18, int16_t)
REGISTER_MUSA_CAST_VERSIONED_TYPED_KERNEL(13, 18, uint8_t)
REGISTER_MUSA_CAST_VERSIONED_TYPED_KERNEL(13, 18, uint16_t)
REGISTER_MUSA_CAST_VERSIONED_TYPED_KERNEL(13, 18, uint32_t)
REGISTER_MUSA_CAST_VERSIONED_TYPED_KERNEL(13, 18, uint64_t)
REGISTER_MUSA_CAST_VERSIONED_TYPED_KERNEL(13, 18, bool)
REGISTER_MUSA_CAST_STRING_TO_INT32_VERSIONED_KERNEL(13, 18)

// Register Cast operators (version 19+)
REGISTER_MUSA_CAST_TYPED_KERNEL(19, int32_t)
REGISTER_MUSA_CAST_TYPED_KERNEL(19, int64_t)
REGISTER_MUSA_CAST_TYPED_KERNEL(19, MLFloat16)
REGISTER_MUSA_CAST_TYPED_KERNEL(19, float)
REGISTER_MUSA_CAST_TYPED_KERNEL(19, int8_t)
REGISTER_MUSA_CAST_TYPED_KERNEL(19, int16_t)
REGISTER_MUSA_CAST_TYPED_KERNEL(19, uint8_t)
REGISTER_MUSA_CAST_TYPED_KERNEL(19, uint16_t)
REGISTER_MUSA_CAST_TYPED_KERNEL(19, uint32_t)
REGISTER_MUSA_CAST_TYPED_KERNEL(19, uint64_t)
REGISTER_MUSA_CAST_TYPED_KERNEL(19, bool)

#undef REGISTER_MUSA_CAST_STRING_TO_INT32_VERSIONED_KERNEL

} // namespace musa
} // namespace onnxruntime