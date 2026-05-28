// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/tensor/concat.h"
#include "core/providers/common.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/musa_execution_provider.h"
#include <algorithm>
#include <musa_runtime.h>
#include <mudnn.h>
#include <string>
#include <vector>

using onnxruntime::common::Status;
namespace onnxruntime {

namespace musa {

namespace {

Status ReadConcatV2Axis(const Tensor& tensor, musaStream_t stream, int64_t& value) {
  ORT_RETURN_IF_NOT(tensor.Shape().Size() == 1, "ConcatV2 axis must be a scalar tensor.");
  if (tensor.IsDataType<int32_t>()) {
    int32_t scalar = 0;
    if (tensor.Location().device.UsesCpuMemory()) {
      scalar = *tensor.Data<int32_t>();
    } else {
      MUSA_RETURN_IF_ERROR(musaMemcpyAsync(&scalar, tensor.DataRaw(), sizeof(scalar), musaMemcpyDeviceToHost, stream));
      MUSA_RETURN_IF_ERROR(musaStreamSynchronize(stream));
    }
    value = static_cast<int64_t>(scalar);
    return Status::OK();
  }
  if (tensor.IsDataType<int64_t>()) {
    int64_t scalar = 0;
    if (tensor.Location().device.UsesCpuMemory()) {
      scalar = *tensor.Data<int64_t>();
    } else {
      MUSA_RETURN_IF_ERROR(musaMemcpyAsync(&scalar, tensor.DataRaw(), sizeof(scalar), musaMemcpyDeviceToHost, stream));
      MUSA_RETURN_IF_ERROR(musaStreamSynchronize(stream));
    }
    value = scalar;
    return Status::OK();
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "ConcatV2 axis must be int32 or int64.");
}

Status PrepareConcatV2(OpKernelContext* ctx, musaStream_t stream, int64_t& axis,
                       TensorShapeVector& output_dims, std::vector<const Tensor*>& inputs) {
  const int input_count = ctx->InputCount();
  ORT_RETURN_IF_NOT(input_count >= 2, "ConcatV2 requires at least one value input and one axis input.");

  const Tensor* axis_tensor = ctx->Input<Tensor>(input_count - 1);
  ORT_RETURN_IF_NOT(axis_tensor != nullptr, "ConcatV2 axis input is null.");
  ORT_RETURN_IF_ERROR(ReadConcatV2Axis(*axis_tensor, stream, axis));

  const int value_count = input_count - 1;
  inputs.reserve(static_cast<size_t>(value_count));
  const Tensor* first = ctx->Input<Tensor>(0);
  ORT_RETURN_IF_NOT(first != nullptr, "ConcatV2 first value input is null.");
  ORT_RETURN_IF_NOT(first->Shape().NumDimensions() > 0, "ConcatV2 cannot concatenate scalar tensors.");
  const auto rank = first->Shape().NumDimensions();
  axis = HandleNegativeAxis(axis, gsl::narrow_cast<int64_t>(rank));
  output_dims = first->Shape().AsShapeVector();
  output_dims[onnxruntime::narrow<size_t>(axis)] = 0;

  for (int i = 0; i < value_count; ++i) {
    const Tensor* input = ctx->Input<Tensor>(i);
    ORT_RETURN_IF_NOT(input != nullptr, "ConcatV2 value input is null.");
    ORT_RETURN_IF_NOT(input->DataType() == first->DataType(), "ConcatV2 value input data types must match.");
    ORT_RETURN_IF_NOT(input->Shape().NumDimensions() == rank, "ConcatV2 value input ranks must match.");
    const auto& dims = input->Shape().GetDims();
    for (size_t d = 0; d < rank; ++d) {
      if (d == static_cast<size_t>(axis)) {
        continue;
      }
      ORT_RETURN_IF_NOT(dims[d] == output_dims[d], "ConcatV2 non-concat dimensions must match.");
    }
    output_dims[onnxruntime::narrow<size_t>(axis)] += dims[onnxruntime::narrow<size_t>(axis)];
    inputs.push_back(input);
  }

  return Status::OK();
}

class ConcatV2 final : public MusaKernel {
 public:
  explicit ConcatV2(const OpKernelInfo& info) : MusaKernel(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
    int64_t axis = 0;
    TensorShapeVector output_dims;
    std::vector<const Tensor*> inputs;
    ORT_RETURN_IF_ERROR(PrepareConcatV2(ctx, Stream(ctx), axis, output_dims, inputs));

    Tensor* output = ctx->Output(0, TensorShape{output_dims});
    ORT_RETURN_IF_NOT(output != nullptr, "ConcatV2 output tensor is null.");
    if (output->Shape().Size() == 0) {
      return Status::OK();
    }

    const size_t element_size = inputs.front()->DataType()->Size();
    const size_t axis_index = onnxruntime::narrow<size_t>(axis);
    const int64_t outer = inputs.front()->Shape().SizeToDimension(axis_index);
    const int64_t inner = (axis_index + 1 == output_dims.size())
                              ? 1
                              : inputs.front()->Shape().SizeFromDimension(axis_index + 1);
    const int64_t output_axis = output_dims[axis_index];
    auto* dst_base = static_cast<uint8_t*>(output->MutableDataRaw());

    for (int64_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
      int64_t dst_axis_offset = 0;
      for (const Tensor* input : inputs) {
        const int64_t input_axis = input->Shape()[axis_index];
        const size_t bytes = onnxruntime::narrow<size_t>(input_axis * inner) * element_size;
        if (bytes > 0) {
          const auto* src = static_cast<const uint8_t*>(input->DataRaw()) +
                            onnxruntime::narrow<size_t>(outer_idx * input_axis * inner) * element_size;
          auto* dst = dst_base +
                      onnxruntime::narrow<size_t>((outer_idx * output_axis + dst_axis_offset) * inner) * element_size;
          MUSA_RETURN_IF_ERROR(musaMemcpyAsync(dst, src, bytes, musaMemcpyDeviceToDevice, Stream(ctx)));
        }
        dst_axis_offset += input_axis;
      }
    }

    return Status::OK();
  }
};

}  // namespace

ONNX_OPERATOR_KERNEL_EX(
    ConcatV2,
    kOnnxDomain,
    1,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    ConcatV2);

// MUSA device-based implementation using MusaPreparation and mudnn Concat class
template <typename T>
Status SimpleMusaConcatOp(const MusaPreparation& prepare, const Prepare& p) {
  // Validate prepared tensors
  if (prepare.inputTensors.empty() || prepare.outputTensors.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid prepared tensor configuration");
  }

  if (prepare.inputTensors.size() != p.inputs.size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Mismatch between prepared tensors and input count");
  }

  // Use mudnn Concat class for device computation
  try {
    // Create mudnn Concat operation
    ::musa::dnn::Concat concat_op;

    // Set axis for concatenation
    auto status = concat_op.SetAxis(static_cast<int>(p.axis));
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set concat axis, status: " +
                             std::to_string(static_cast<int>(status)));
    }

    // Get mutable references to tensors for mudnn interface
    ::musa::dnn::Tensor mutable_output = prepare.outputTensors[0];

    // Run the concat operation
    status = concat_op.Run(prepare.GetHandle(),
                          mutable_output,
                          static_cast<int>(prepare.inputTensors.size()),
                          prepare.inputTensors.data());

    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "mudnn Concat operation failed, status: " +
                             std::to_string(static_cast<int>(status)));
    }

  } catch (const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Exception in mudnn Concat operation: " +
                           std::string(e.what()));
  }

  return Status::OK();
}

template <typename T>
Status Concat<T>::Prepare(OpKernelContext* ctx, MusaPreparation& prepare, struct Prepare& p) const {
  // 1. Get input tensors count
  const auto num_inputs = ctx->InputCount();
  if (num_inputs < 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Must have at least one input tensor");
  }

  // 2. Build input tensors vector
  InlinedTensorsVector input_tensors;
  input_tensors.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    input_tensors.push_back(ctx->Input<Tensor>(i));
  }

  // 3. Use ConcatBase::PrepareForCompute to validate inputs and prepare metadata
  ORT_RETURN_IF_ERROR(PrepareForCompute(ctx, input_tensors, p));

  // Early return for empty output tensor - no data operation needed
  if (p.output_num_elements == 0) {
    return Status::OK();
  }

  // 4. Store tensor pointers in preparation
  prepare.input_a_ptr = p.inputs[0].tensor->DataRaw();
  prepare.output_ptr = p.output_tensor->MutableDataRaw();
  prepare.output_size = p.output_tensor->Shape().Size();

  // Note: input_a_ptr or individual input pointers can be null for empty tensors (size=0)
  // This is a valid case (e.g., att_cache with shape (12, 4, 0, 128) in streaming models)
  // SetupMusaTensor and mudnn Concat correctly handle null pointers for empty tensors
  // Only check output_ptr for non-empty output (already handled by early return above)
  if (!prepare.output_ptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid output tensor data pointer");
  }

  // 5. Get MUSA data type
  const auto musaType = GetMusaDataType<T>();

  // 6. Prepare MUSA tensors
  ORT_TRY {
    // Set up MUSA stream for asynchronous execution
    // In EP mode, stream is already set in PerThreadContext constructor
    // In legacy mode, we need to set stream here
    auto* stream = Stream(ctx);
    if (prepare.handle && stream) {
      auto musa_status = prepare.handle->SetStream(stream);
      if (musa_status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Failed to set MUSA stream, status: " +
                               std::to_string(static_cast<int>(musa_status)));
      }
    }

    // Initialize tensors vectors
    prepare.inputTensors.resize(num_inputs);
    prepare.outputTensors.resize(1);

    // Setup input tensors
    for (int i = 0; i < num_inputs; ++i) {
      ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.inputTensors[i], p.inputs[i].tensor, musaType, &prepare));
    }

    // Setup output tensor
    ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.outputTensors[0], p.output_tensor, musaType, &prepare));
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  return Status::OK();
}

template <typename T>
Status Concat<T>::ComputeInternal(OpKernelContext* ctx) const {
  // Prepare metadata using ConcatBase
  struct Prepare p;

  // Prepare MUSA operation - use EP mode for thread-safe Handle management
  const auto* ep = static_cast<const MusaExecutionProvider*>(
      Info().GetExecutionProvider());
  MusaPreparation prepare(ep);
  auto status = Prepare(ctx, prepare, p);
  ORT_RETURN_IF_ERROR(status);

  // Early return for empty output - Prepare already handled it
  if (p.output_num_elements == 0) {
    return Status::OK();
  }

  // Call MUSA device concat operation using prepared data
  ORT_RETURN_IF_ERROR(SimpleMusaConcatOp<T>(prepare, p));

  return Status::OK();
}

// Macro for registering typed compute function with MUSA implementation
#define REGISTER_MUSA_CONCAT_TYPED_COMPUTE(T)                               \
  template Status Concat<T>::ComputeInternal(OpKernelContext* ctx) const;

// Macro for registering typed kernel
#define REGISTER_MUSA_CONCAT_TYPED_KERNEL_IN_DOMAIN(domain, ver, T)         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                            \
      Concat, domain, ver, T, kMusaExecutionProvider,                       \
      (*KernelDefBuilder::Create())                                         \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),           \
      Concat<T>);

#define REGISTER_MUSA_CONCAT_TYPED_KERNEL(ver, T)                           \
  REGISTER_MUSA_CONCAT_TYPED_KERNEL_IN_DOMAIN(kOnnxDomain, ver, T)

// Versioned kernel registration macro
#define REGISTER_MUSA_CONCAT_VERSIONED_TYPED_KERNEL_IN_DOMAIN(domain, startver, endver, T) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                  \
      Concat, domain, startver, endver, T, kMusaExecutionProvider,          \
      (*KernelDefBuilder::Create())                                         \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),           \
      Concat<T>);

#define REGISTER_MUSA_CONCAT_VERSIONED_TYPED_KERNEL(startver, endver, T)    \
  REGISTER_MUSA_CONCAT_VERSIONED_TYPED_KERNEL_IN_DOMAIN(kOnnxDomain, startver, endver, T)

// Combined macro for both kernel and compute registration (only kernel)
#define REGISTER_MUSA_CONCAT_TYPED(ver, T)                                  \
  REGISTER_MUSA_CONCAT_TYPED_KERNEL(ver, T)

// Combined macro for versioned kernel registration (only kernel)
#define REGISTER_MUSA_CONCAT_VERSIONED_TYPED(startver, endver, T)           \
  REGISTER_MUSA_CONCAT_VERSIONED_TYPED_KERNEL(startver, endver, T)

// Register Concat operations for supported data types
// Based on MUSA interface survey, supporting int8/16/32/64, uint8/16/32/64, half, float
// Excluding double and bfloat16 as per user requirement

// Concat v4-v10
REGISTER_MUSA_CONCAT_VERSIONED_TYPED(4, 10, uint8_t)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED(4, 10, uint16_t)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED(4, 10, uint32_t)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED(4, 10, uint64_t)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED(4, 10, int8_t)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED(4, 10, int16_t)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED(4, 10, int32_t)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED(4, 10, int64_t)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED(4, 10, MLFloat16)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED(4, 10, float)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED(4, 10, bool)

// Concat v11-v12
REGISTER_MUSA_CONCAT_VERSIONED_TYPED(11, 12, uint8_t)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED(11, 12, uint16_t)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED(11, 12, uint32_t)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED(11, 12, uint64_t)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED(11, 12, int8_t)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED(11, 12, int16_t)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED(11, 12, int32_t)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED(11, 12, int64_t)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED(11, 12, MLFloat16)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED(11, 12, float)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED(11, 12, bool)

// Concat v13 (latest)
REGISTER_MUSA_CONCAT_TYPED(13, uint8_t)
REGISTER_MUSA_CONCAT_TYPED(13, uint16_t)
REGISTER_MUSA_CONCAT_TYPED(13, uint32_t)
REGISTER_MUSA_CONCAT_TYPED(13, uint64_t)
REGISTER_MUSA_CONCAT_TYPED(13, int8_t)
REGISTER_MUSA_CONCAT_TYPED(13, int16_t)
REGISTER_MUSA_CONCAT_TYPED(13, int32_t)
REGISTER_MUSA_CONCAT_TYPED(13, int64_t)
REGISTER_MUSA_CONCAT_TYPED(13, MLFloat16)
REGISTER_MUSA_CONCAT_TYPED(13, float)
REGISTER_MUSA_CONCAT_TYPED(13, bool)

#ifdef ENABLE_MUSA_NHWC_OPS
REGISTER_MUSA_CONCAT_VERSIONED_TYPED_KERNEL_IN_DOMAIN(kMSInternalNHWCDomain, 4, 10, uint8_t)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED_KERNEL_IN_DOMAIN(kMSInternalNHWCDomain, 4, 10, uint16_t)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED_KERNEL_IN_DOMAIN(kMSInternalNHWCDomain, 4, 10, uint32_t)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED_KERNEL_IN_DOMAIN(kMSInternalNHWCDomain, 4, 10, uint64_t)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED_KERNEL_IN_DOMAIN(kMSInternalNHWCDomain, 4, 10, int8_t)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED_KERNEL_IN_DOMAIN(kMSInternalNHWCDomain, 4, 10, int16_t)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED_KERNEL_IN_DOMAIN(kMSInternalNHWCDomain, 4, 10, int32_t)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED_KERNEL_IN_DOMAIN(kMSInternalNHWCDomain, 4, 10, int64_t)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED_KERNEL_IN_DOMAIN(kMSInternalNHWCDomain, 4, 10, MLFloat16)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED_KERNEL_IN_DOMAIN(kMSInternalNHWCDomain, 4, 10, float)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED_KERNEL_IN_DOMAIN(kMSInternalNHWCDomain, 4, 10, bool)

REGISTER_MUSA_CONCAT_VERSIONED_TYPED_KERNEL_IN_DOMAIN(kMSInternalNHWCDomain, 11, 12, uint8_t)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED_KERNEL_IN_DOMAIN(kMSInternalNHWCDomain, 11, 12, uint16_t)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED_KERNEL_IN_DOMAIN(kMSInternalNHWCDomain, 11, 12, uint32_t)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED_KERNEL_IN_DOMAIN(kMSInternalNHWCDomain, 11, 12, uint64_t)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED_KERNEL_IN_DOMAIN(kMSInternalNHWCDomain, 11, 12, int8_t)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED_KERNEL_IN_DOMAIN(kMSInternalNHWCDomain, 11, 12, int16_t)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED_KERNEL_IN_DOMAIN(kMSInternalNHWCDomain, 11, 12, int32_t)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED_KERNEL_IN_DOMAIN(kMSInternalNHWCDomain, 11, 12, int64_t)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED_KERNEL_IN_DOMAIN(kMSInternalNHWCDomain, 11, 12, MLFloat16)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED_KERNEL_IN_DOMAIN(kMSInternalNHWCDomain, 11, 12, float)
REGISTER_MUSA_CONCAT_VERSIONED_TYPED_KERNEL_IN_DOMAIN(kMSInternalNHWCDomain, 11, 12, bool)

REGISTER_MUSA_CONCAT_TYPED_KERNEL_IN_DOMAIN(kMSInternalNHWCDomain, 13, uint8_t)
REGISTER_MUSA_CONCAT_TYPED_KERNEL_IN_DOMAIN(kMSInternalNHWCDomain, 13, uint16_t)
REGISTER_MUSA_CONCAT_TYPED_KERNEL_IN_DOMAIN(kMSInternalNHWCDomain, 13, uint32_t)
REGISTER_MUSA_CONCAT_TYPED_KERNEL_IN_DOMAIN(kMSInternalNHWCDomain, 13, uint64_t)
REGISTER_MUSA_CONCAT_TYPED_KERNEL_IN_DOMAIN(kMSInternalNHWCDomain, 13, int8_t)
REGISTER_MUSA_CONCAT_TYPED_KERNEL_IN_DOMAIN(kMSInternalNHWCDomain, 13, int16_t)
REGISTER_MUSA_CONCAT_TYPED_KERNEL_IN_DOMAIN(kMSInternalNHWCDomain, 13, int32_t)
REGISTER_MUSA_CONCAT_TYPED_KERNEL_IN_DOMAIN(kMSInternalNHWCDomain, 13, int64_t)
REGISTER_MUSA_CONCAT_TYPED_KERNEL_IN_DOMAIN(kMSInternalNHWCDomain, 13, MLFloat16)
REGISTER_MUSA_CONCAT_TYPED_KERNEL_IN_DOMAIN(kMSInternalNHWCDomain, 13, float)
REGISTER_MUSA_CONCAT_TYPED_KERNEL_IN_DOMAIN(kMSInternalNHWCDomain, 13, bool)
#endif

// Register template instantiations for compute function (only once per type)
REGISTER_MUSA_CONCAT_TYPED_COMPUTE(uint8_t)
REGISTER_MUSA_CONCAT_TYPED_COMPUTE(uint16_t)
REGISTER_MUSA_CONCAT_TYPED_COMPUTE(uint32_t)
REGISTER_MUSA_CONCAT_TYPED_COMPUTE(uint64_t)
REGISTER_MUSA_CONCAT_TYPED_COMPUTE(int8_t)
REGISTER_MUSA_CONCAT_TYPED_COMPUTE(int16_t)
REGISTER_MUSA_CONCAT_TYPED_COMPUTE(int32_t)
REGISTER_MUSA_CONCAT_TYPED_COMPUTE(int64_t)
REGISTER_MUSA_CONCAT_TYPED_COMPUTE(MLFloat16)
REGISTER_MUSA_CONCAT_TYPED_COMPUTE(float)
REGISTER_MUSA_CONCAT_TYPED_COMPUTE(bool)

} // namespace musa
} // namespace onnxruntime
