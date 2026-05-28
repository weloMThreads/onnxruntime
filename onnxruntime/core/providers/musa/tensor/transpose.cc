// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/tensor/transpose.h"
#include "core/providers/common.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/musa_call.h"
#include "core/providers/musa/musa_execution_provider.h"
#include <algorithm>
#include <cstring>
#include <musa_runtime.h>
#include <mudnn.h>
#include <string>
#include <vector>

using onnxruntime::common::Status;
namespace onnxruntime {
namespace musa {

namespace {

Status ReadAxisVector(const Tensor& axis_tensor, int64_t rank, std::vector<bool>& reverse_axes) {
  ORT_RETURN_IF_NOT(axis_tensor.Shape().NumDimensions() == 1, "ReverseV2 axis input must be a 1D tensor.");
  reverse_axes.assign(static_cast<size_t>(rank), false);
  const int64_t axis_count = axis_tensor.Shape()[0];
  for (int64_t i = 0; i < axis_count; ++i) {
    int64_t axis = 0;
    if (axis_tensor.IsDataType<int32_t>()) {
      axis = static_cast<int64_t>(axis_tensor.Data<int32_t>()[i]);
    } else if (axis_tensor.IsDataType<int64_t>()) {
      axis = axis_tensor.Data<int64_t>()[i];
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "ReverseV2 axis input must be int32 or int64.");
    }
    axis = HandleNegativeAxis(axis, rank);
    ORT_RETURN_IF_NOT(!reverse_axes[onnxruntime::narrow<size_t>(axis)], "ReverseV2 axis values must be unique.");
    reverse_axes[onnxruntime::narrow<size_t>(axis)] = true;
  }
  return Status::OK();
}

void ReverseRawBuffer(const uint8_t* input, uint8_t* output, const TensorShape& shape,
                      const std::vector<bool>& reverse_axes, size_t element_size) {
  const auto dims = shape.GetDims();
  const size_t rank = dims.size();
  const int64_t element_count = shape.Size();
  TensorShapeVector strides(rank, 1);
  for (size_t i = rank; i-- > 1;) {
    strides[i - 1] = strides[i] * dims[i];
  }

  for (int64_t linear = 0; linear < element_count; ++linear) {
    int64_t remaining = linear;
    int64_t source_index = 0;
    for (size_t axis = 0; axis < rank; ++axis) {
      const int64_t coord = strides[axis] == 0 ? 0 : remaining / strides[axis];
      remaining = strides[axis] == 0 ? 0 : remaining % strides[axis];
      const int64_t source_coord = reverse_axes[axis] ? dims[axis] - 1 - coord : coord;
      source_index += source_coord * strides[axis];
    }
    std::memcpy(output + onnxruntime::narrow<size_t>(linear) * element_size,
                input + onnxruntime::narrow<size_t>(source_index) * element_size,
                element_size);
  }
}

class ReverseV2 final : public MusaKernel {
 public:
  explicit ReverseV2(const OpKernelInfo& info) : MusaKernel(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
    const Tensor* input = ctx->Input<Tensor>(0);
    const Tensor* axis = ctx->Input<Tensor>(1);
    ORT_RETURN_IF_NOT(input != nullptr && axis != nullptr, "ReverseV2 inputs must not be null.");

    std::vector<bool> reverse_axes;
    ORT_RETURN_IF_ERROR(ReadAxisVector(*axis, gsl::narrow_cast<int64_t>(input->Shape().NumDimensions()), reverse_axes));
    Tensor* output = ctx->Output(0, input->Shape());
    ORT_RETURN_IF_NOT(output != nullptr, "ReverseV2 output tensor is null.");

    const size_t bytes = input->SizeInBytes();
    if (bytes == 0) {
      return Status::OK();
    }

    const bool need_reverse = std::any_of(reverse_axes.begin(), reverse_axes.end(), [](bool v) { return v; });
    if (!need_reverse) {
      if (input->DataRaw() != output->MutableDataRaw()) {
        MUSA_RETURN_IF_ERROR(musaMemcpyAsync(output->MutableDataRaw(), input->DataRaw(), bytes,
                                             musaMemcpyDeviceToDevice, Stream(ctx)));
      }
      return Status::OK();
    }

    std::vector<uint8_t> host_input(bytes);
    std::vector<uint8_t> host_output(bytes);
    MUSA_RETURN_IF_ERROR(musaMemcpyAsync(host_input.data(), input->DataRaw(), bytes,
                                         musaMemcpyDeviceToHost, Stream(ctx)));
    MUSA_RETURN_IF_ERROR(musaStreamSynchronize(Stream(ctx)));
    ReverseRawBuffer(host_input.data(), host_output.data(), input->Shape(), reverse_axes, input->DataType()->Size());
    MUSA_RETURN_IF_ERROR(musaMemcpyAsync(output->MutableDataRaw(), host_output.data(), bytes,
                                         musaMemcpyHostToDevice, Stream(ctx)));
    return Status::OK();
  }
};

template <typename T>
Status ComputeInvertPermutationTyped(const Tensor& input, Tensor& output, musaStream_t stream) {
  ORT_RETURN_IF_NOT(input.Shape().NumDimensions() == 1, "InvertPermutation input must be a 1D tensor.");
  const int64_t n = input.Shape().Size();
  std::vector<T> host_input(static_cast<size_t>(n));
  std::vector<T> host_output(static_cast<size_t>(n));
  if (n > 0) {
    MUSA_RETURN_IF_ERROR(musaMemcpyAsync(host_input.data(), input.DataRaw(), static_cast<size_t>(n) * sizeof(T),
                                         musaMemcpyDeviceToHost, stream));
    MUSA_RETURN_IF_ERROR(musaStreamSynchronize(stream));
  }
  std::vector<bool> seen(static_cast<size_t>(n), false);
  for (int64_t i = 0; i < n; ++i) {
    const int64_t value = static_cast<int64_t>(host_input[onnxruntime::narrow<size_t>(i)]);
    ORT_RETURN_IF_NOT(value >= 0 && value < n, "InvertPermutation value out of range.");
    ORT_RETURN_IF_NOT(!seen[onnxruntime::narrow<size_t>(value)], "InvertPermutation input contains duplicates.");
    seen[onnxruntime::narrow<size_t>(value)] = true;
    host_output[onnxruntime::narrow<size_t>(value)] = static_cast<T>(i);
  }
  if (n > 0) {
    MUSA_RETURN_IF_ERROR(musaMemcpyAsync(output.MutableDataRaw(), host_output.data(), static_cast<size_t>(n) * sizeof(T),
                                         musaMemcpyHostToDevice, stream));
  }
  return Status::OK();
}

class InvertPermutation final : public MusaKernel {
 public:
  explicit InvertPermutation(const OpKernelInfo& info) : MusaKernel(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
    const Tensor* input = ctx->Input<Tensor>(0);
    ORT_RETURN_IF_NOT(input != nullptr, "InvertPermutation input is null.");
    Tensor* output = ctx->Output(0, input->Shape());
    ORT_RETURN_IF_NOT(output != nullptr, "InvertPermutation output tensor is null.");
    if (input->IsDataType<int32_t>()) {
      return ComputeInvertPermutationTyped<int32_t>(*input, *output, Stream(ctx));
    }
    if (input->IsDataType<int64_t>()) {
      return ComputeInvertPermutationTyped<int64_t>(*input, *output, Stream(ctx));
    }
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "InvertPermutation input must be int32 or int64.");
  }
};

}  // namespace

ONNX_OPERATOR_KERNEL_EX(
    ReverseV2,
    kOnnxDomain,
    1,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("Tidx", BuildKernelDefConstraints<int32_t, int64_t>()),
    ReverseV2);

ONNX_OPERATOR_KERNEL_EX(
    InvertPermutation,
    kOnnxDomain,
    1,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", BuildKernelDefConstraints<int32_t, int64_t>()),
    InvertPermutation);

// Helper function to perform MUSA transpose using muDNN Permute operation
// This is a lower-level API that can be used by other ops (e.g., Conv for weight prepacking)
Status DoMusaTranspose(const void* input_data,
                       void* output_data,
                       const std::vector<int64_t>& input_shape,
                       const std::vector<size_t>& perm,
                       ::musa::dnn::Tensor::Type data_type,
                       musaStream_t stream,
                       int device_id) {
  try {
    // Create muDNN tensors
    ::musa::dnn::Tensor input_tensor;
    ::musa::dnn::Tensor output_tensor;

    // Set tensor types
    auto status = input_tensor.SetType(data_type);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set input tensor type in transpose");
    }
    status = output_tensor.SetType(data_type);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set output tensor type in transpose");
    }

    // Set addresses
    status = input_tensor.SetAddr(input_data);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set input tensor address in transpose");
    }
    status = output_tensor.SetAddr(output_data);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set output tensor address in transpose");
    }

    // Set format (use NCHW for weights)
    status = input_tensor.SetFormat(::musa::dnn::Tensor::Format::NCHW);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set input tensor format in transpose");
    }
    status = output_tensor.SetFormat(::musa::dnn::Tensor::Format::NCHW);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set output tensor format in transpose");
    }

    // Set shapes
    status = input_tensor.SetNdInfo(static_cast<int>(input_shape.size()), const_cast<int64_t*>(input_shape.data()));
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set input tensor shape in transpose");
    }

    // Calculate output shape based on permutation
    std::vector<int64_t> output_shape(input_shape.size());
    for (size_t i = 0; i < perm.size(); ++i) {
      output_shape[i] = input_shape[perm[i]];
    }
    status = output_tensor.SetNdInfo(static_cast<int>(output_shape.size()), output_shape.data());
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set output tensor shape in transpose");
    }

    // Create muDNN handle
    ::musa::dnn::Handle handle(device_id);
    if (stream) {
      handle.SetStream(stream);
    }

    // Create and configure permute operation
    ::musa::dnn::Permute permute_op;
    
    // Convert perm to int64_t array
    std::vector<int64_t> permute_dims(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) {
      permute_dims[i] = static_cast<int64_t>(perm[i]);
    }

    // Configure dimension stride for permutation
    status = ::musa::dnn::Permute::ConfigDimStride(output_tensor, input_tensor,
                                                    static_cast<int>(permute_dims.size()),
                                                    permute_dims.data());
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, 
                             "Failed to configure mudnn Permute dimensions, status: " +
                             std::to_string(static_cast<int>(status)));
    }

    // Run the permute operation
    status = permute_op.Run(handle, output_tensor, input_tensor);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, 
                             "mudnn Permute operation failed, status: " +
                             std::to_string(static_cast<int>(status)));
    }

  } catch (const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, 
                           "Exception in mudnn Permute operation: " + std::string(e.what()));
  }

  return Status::OK();
}

// MUSA device-based implementation using MusaPreparation and mudnn Permute class
template <typename T>
Status SimpleMusaTransposeOp(const MusaPreparation& prepare,
                               const InlinedVector<size_t>& perm) {
  // Get tensor data from prepared MUSA tensors
  const T* input_data = reinterpret_cast<const T*>(prepare.input_a_ptr);
  T* output_data = reinterpret_cast<T*>(prepare.output_ptr);

  // Note: input_data and output_data can be null for zero-dimension tensors (size=0)
  // This is a valid case (e.g., shape (12, 4, 0, 128) in streaming models)
  // mudnn Permute and SetupMusaTensor correctly handle null pointers for empty tensors

  if (prepare.inputTensors.empty() || prepare.outputTensors.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid prepared tensor configuration");
  }

  // Use mudnn Permute class for device computation
  try {
    // Create mudnn Permute operation
    ::musa::dnn::Permute permute_op;

    // Create mutable copies of input and output tensors for Permute configuration
    ::musa::dnn::Tensor mutable_input = prepare.inputTensors[0];
    ::musa::dnn::Tensor mutable_output = prepare.outputTensors[0];

    // Convert perm to int64_t array for ConfigDimStride
    std::vector<int64_t> permute_dims(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) {
      permute_dims[i] = static_cast<int64_t>(perm[i]);
    }

    // Configure dimension stride for permutation
    auto status = ::musa::dnn::Permute::ConfigDimStride(mutable_output, mutable_input,
                                                        static_cast<int>(permute_dims.size()),
                                                        permute_dims.data());
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to configure mudnn Permute dimensions, status: " +
                             std::to_string(static_cast<int>(status)));
    }

    // Run the permute operation directly on device
    status = permute_op.Run(prepare.GetHandle(), mutable_output, mutable_input);

    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "mudnn Permute operation failed, status: " +
                             std::to_string(static_cast<int>(status)));
    }


  } catch (const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Exception in mudnn Permute operation: " +
                           std::string(e.what()));
  }

  return Status::OK();
}

template <typename T>
Status Transpose<T>::Prepare(OpKernelContext* ctx, MusaPreparation& prepare) const {
  // 1. Get input tensor
  const Tensor* X = ctx->Input<Tensor>(0);
  if (!X) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid input tensor");
  }

  // 2. Compute output shape using TransposeBase functionality
  int32_t rank = gsl::narrow_cast<int32_t>(X->Shape().NumDimensions());
  TensorShapeVector Y_dims(rank);
  InlinedVector<size_t> default_perm(rank);
  const InlinedVector<size_t>* perm = nullptr;
  const auto& status = ComputeOutputShape(*X, Y_dims, default_perm, perm);
  if (!status.IsOK()) {
    return status;
  }

  // 3. Create output tensor
  TensorShape output_shape{Y_dims};
  Tensor* Y = ctx->Output(0, output_shape);
  if (!Y) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor");
  }

  // 4. Store tensor pointers and shapes in preparation
  prepare.input_a_ptr = X->DataRaw();
  prepare.output_ptr = Y->MutableDataRaw();
  prepare.output_size = output_shape.Size();
  prepare.input_a_shape = X->Shape();
  prepare.output_shape = output_shape;

  // Note: input_a_ptr and output_ptr can be null for zero-dimension tensors (size=0)
  // This is a valid case (e.g., att_cache with shape (12, 4, 0, 128) in streaming models)
  // SetupMusaTensor and mudnn Permute correctly handle null pointers for empty tensors
  // No null pointer check needed here - empty tensors are valid

  // 5. Store permutation for use in ComputeInternal
  // Note: We need to store the permutation somewhere accessible to ComputeInternal
  // For now, we'll recalculate it in ComputeInternal. A better approach would be
  // to extend MusaPreparation to store additional operation-specific data.

  // 6. Get MUSA data type
  const auto musaType = GetMusaDataType<T>();

  // 7. Prepare MUSA tensors - using ORT_TRY/ORT_CATCH like other MusaEP operations
  ORT_TRY {
    // Set up MUSA stream for asynchronous execution
    auto* stream = Stream(ctx);
    if (prepare.handle) {
      if (stream) {
        auto musa_status = prepare.handle->SetStream(stream);
        if (musa_status != ::musa::dnn::Status::SUCCESS) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                 "Failed to set MUSA stream, status: " +
                                     std::to_string(static_cast<int>(musa_status)));
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
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  return Status::OK();
}

template <typename T>
Status Transpose<T>::ComputeInternal(OpKernelContext* ctx) const {
  // Get input tensor to recalculate permutation
  const Tensor* X = ctx->Input<Tensor>(0);

  // Recalculate permutation (this is not ideal but works for now)
  int32_t rank = gsl::narrow_cast<int32_t>(X->Shape().NumDimensions());
  TensorShapeVector Y_dims(rank);
  InlinedVector<size_t> default_perm(rank);
  const InlinedVector<size_t>* perm = nullptr;
  const auto& perm_status = ComputeOutputShape(*X, Y_dims, default_perm, perm);
  if (!perm_status.IsOK()) {
    return perm_status;
  }

  // Create output tensor
  TensorShape output_shape{Y_dims};
  Tensor* Y = ctx->Output(0, output_shape);

  // Early return for empty tensors - mudnn Permute does not support zero-dimension tensors
  // This is a valid case in streaming models (e.g., att_cache with shape (12, 4, 0, 128))
  if (output_shape.Size() == 0) {
    return Status::OK();
  }

  // Prepare MUSA operation
  const auto* ep = static_cast<const MusaExecutionProvider*>(
      Info().GetExecutionProvider());
  MusaPreparation prepare(ep);
  ORT_RETURN_IF_ERROR(Prepare(ctx, prepare));

  // Call MUSA device transpose operation using prepared data
  ORT_RETURN_IF_ERROR(SimpleMusaTransposeOp<T>(prepare, *perm));

  return Status::OK();
}

// Macro for registering typed compute function with MUSA implementation
#define REGISTER_MUSA_TRANSPOSE_TYPED_COMPUTE(T)                               \
  template Status Transpose<T>::ComputeInternal(OpKernelContext* ctx) const;

// Macro for registering versioned typed kernel (opset 1-12)
#define REGISTER_MUSA_TRANSPOSE_VERSIONED_TYPED_KERNEL(start_ver, end_ver, T)  \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                     \
      Transpose, kOnnxDomain, start_ver, end_ver, T, kMusaExecutionProvider,   \
      (*KernelDefBuilder::Create())                                            \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),              \
      Transpose<T>);

// Macro for registering typed kernel (opset 13+)
#define REGISTER_MUSA_TRANSPOSE_TYPED_KERNEL(ver, T)                           \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                               \
      Transpose, kOnnxDomain, ver, T, kMusaExecutionProvider,                  \
      (*KernelDefBuilder::Create())                                            \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),              \
      Transpose<T>);

// Register Transpose operations for supported data types
// Opset 1-12 (versioned registration)
REGISTER_MUSA_TRANSPOSE_VERSIONED_TYPED_KERNEL(1, 12, uint8_t)
REGISTER_MUSA_TRANSPOSE_VERSIONED_TYPED_KERNEL(1, 12, uint16_t)
REGISTER_MUSA_TRANSPOSE_VERSIONED_TYPED_KERNEL(1, 12, uint32_t)
REGISTER_MUSA_TRANSPOSE_VERSIONED_TYPED_KERNEL(1, 12, uint64_t)
REGISTER_MUSA_TRANSPOSE_VERSIONED_TYPED_KERNEL(1, 12, int8_t)
REGISTER_MUSA_TRANSPOSE_VERSIONED_TYPED_KERNEL(1, 12, int16_t)
REGISTER_MUSA_TRANSPOSE_VERSIONED_TYPED_KERNEL(1, 12, int32_t)
REGISTER_MUSA_TRANSPOSE_VERSIONED_TYPED_KERNEL(1, 12, int64_t)
REGISTER_MUSA_TRANSPOSE_VERSIONED_TYPED_KERNEL(1, 12, MLFloat16)
REGISTER_MUSA_TRANSPOSE_VERSIONED_TYPED_KERNEL(1, 12, float)
REGISTER_MUSA_TRANSPOSE_VERSIONED_TYPED_KERNEL(1, 12, bool)

// Opset 13+ (current version)
REGISTER_MUSA_TRANSPOSE_TYPED_KERNEL(13, uint8_t)
REGISTER_MUSA_TRANSPOSE_TYPED_KERNEL(13, uint16_t)
REGISTER_MUSA_TRANSPOSE_TYPED_KERNEL(13, uint32_t)
REGISTER_MUSA_TRANSPOSE_TYPED_KERNEL(13, uint64_t)
REGISTER_MUSA_TRANSPOSE_TYPED_KERNEL(13, int8_t)
REGISTER_MUSA_TRANSPOSE_TYPED_KERNEL(13, int16_t)
REGISTER_MUSA_TRANSPOSE_TYPED_KERNEL(13, int32_t)
REGISTER_MUSA_TRANSPOSE_TYPED_KERNEL(13, int64_t)
REGISTER_MUSA_TRANSPOSE_TYPED_KERNEL(13, MLFloat16)
REGISTER_MUSA_TRANSPOSE_TYPED_KERNEL(13, float)
REGISTER_MUSA_TRANSPOSE_TYPED_KERNEL(13, bool)

// Register compute function template instantiations
REGISTER_MUSA_TRANSPOSE_TYPED_COMPUTE(uint8_t)
REGISTER_MUSA_TRANSPOSE_TYPED_COMPUTE(uint16_t)
REGISTER_MUSA_TRANSPOSE_TYPED_COMPUTE(uint32_t)
REGISTER_MUSA_TRANSPOSE_TYPED_COMPUTE(uint64_t)
REGISTER_MUSA_TRANSPOSE_TYPED_COMPUTE(int8_t)
REGISTER_MUSA_TRANSPOSE_TYPED_COMPUTE(int16_t)
REGISTER_MUSA_TRANSPOSE_TYPED_COMPUTE(int32_t)
REGISTER_MUSA_TRANSPOSE_TYPED_COMPUTE(int64_t)
REGISTER_MUSA_TRANSPOSE_TYPED_COMPUTE(MLFloat16)
REGISTER_MUSA_TRANSPOSE_TYPED_COMPUTE(float)
REGISTER_MUSA_TRANSPOSE_TYPED_COMPUTE(bool)

} // namespace musa
} // namespace onnxruntime
