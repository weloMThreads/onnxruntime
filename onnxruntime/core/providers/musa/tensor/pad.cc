// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/tensor/pad.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/musa_call.h"
#include <musa_runtime.h>
#include <mudnn.h>

using onnxruntime::common::Status;
namespace onnxruntime {
namespace musa {

namespace {

::musa::dnn::Pad::Mode GetMusaPadMode(int mode) {
  switch (mode) {
    case 0:
      return ::musa::dnn::Pad::Mode::CONSTANT;
    case 1:
      return ::musa::dnn::Pad::Mode::REFLECT;
    case 2:
      return ::musa::dnn::Pad::Mode::REPLICATE;  // edge -> replicate mapping
    case 3:
      return ::musa::dnn::Pad::Mode::CIRCULAR;  // wrap -> circular mapping
    default:
      return ::musa::dnn::Pad::Mode::CONSTANT;
  }
}

template <typename T>
::musa::dnn::Tensor::Type GetMusaTensorType();

template <>
::musa::dnn::Tensor::Type GetMusaTensorType<float>() {
  return ::musa::dnn::Tensor::Type::FLOAT;
}

template <>
::musa::dnn::Tensor::Type GetMusaTensorType<MLFloat16>() {
  return ::musa::dnn::Tensor::Type::HALF;
}

}  // anonymous namespace

template <typename T>
Status Pad<T>::ExtractDynamicParams(OpKernelContext* ctx,
                                    std::vector<int64_t>& pads,
                                    T& pad_value) const {
  // Get pads from input tensor (v11+)
  const Tensor* pads_tensor = ctx->Input<Tensor>(1);
  if (pads_tensor == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Pads input tensor is null");
  }

  // Extract pads data
  if (pads_tensor->DataType() == DataTypeImpl::GetType<int64_t>()) {
    const int64_t* pads_data = pads_tensor->Data<int64_t>();
    size_t pads_size = pads_tensor->Shape().Size();
    pads.assign(pads_data, pads_data + pads_size);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Pads tensor must have int64_t data type");
  }

  // Get value from input tensor if present (v11+)
  const Tensor* value_tensor = ctx->Input<Tensor>(2);
  if (value_tensor != nullptr && value_tensor->Shape().Size() > 0) {
    if (mode_ == 0) {  // constant mode
      // In dynamic mode, value tensor should have the same type as T
      const T* value_data = value_tensor->Data<T>();
      pad_value = *value_data;
    }
  } else {
    pad_value = value_;  // use default from attribute
  }

  return Status::OK();
}

template <typename T>
std::vector<int64_t> Pad<T>::ComputeOutputShape(const TensorShape& input_shape,
                                                const std::vector<int64_t>& pads) const {
  const auto& input_dims = input_shape.GetDims();
  std::vector<int64_t> output_dims;
  output_dims.reserve(input_dims.size());

  size_t rank = input_dims.size();

  // Pads format: [start_pad_0, start_pad_1, ..., end_pad_0, end_pad_1, ...]
  for (size_t i = 0; i < rank; ++i) {
    int64_t start_pad = pads[i];
    int64_t end_pad = pads[i + rank];
    int64_t output_dim = input_dims[i] + start_pad + end_pad;
    output_dims.push_back(output_dim);
  }

  return output_dims;
}

template <typename T>
Status Pad<T>::ValidatePadding(const TensorShape& input_shape,
                               const std::vector<int64_t>& pads) const {
  const auto& input_dims = input_shape.GetDims();
  size_t rank = input_dims.size();

  // Pads should have exactly 2 * rank elements
  if (pads.size() != 2 * rank) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Pads size should be 2 * input rank. Expected: ",
                           2 * rank, " Got: ", pads.size());
  }

  // Check for negative result dimensions
  for (size_t i = 0; i < rank; ++i) {
    int64_t start_pad = pads[i];
    int64_t end_pad = pads[i + rank];
    int64_t output_dim = input_dims[i] + start_pad + end_pad;
    if (output_dim <= 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Output dimension would be non-positive for axis ", i,
                             ". Input dim: ", input_dims[i],
                             " Start pad: ", start_pad,
                             " End pad: ", end_pad);
    }
  }

  return Status::OK();
}

template <typename T>
Status Pad<T>::ComputePadding(OpKernelContext* ctx,
                              const Tensor& input_tensor,
                              Tensor& output_tensor,
                              const std::vector<int64_t>& pads,
                              T pad_value) const {
  const auto& input_shape = input_tensor.Shape();
  size_t rank = input_shape.NumDimensions();

  // Handle 1D tensor limitation: MUSA Pad doesn't support 1D tensors
  // We need to upgrade 1D tensors to 2D temporarily
  if (rank == 1) {
    // Create a temporary 2D shape [1, original_dim]
    std::vector<int64_t> temp_input_dims = {1, input_shape.GetDims()[0]};
    std::vector<int64_t> temp_output_dims = {1, output_tensor.Shape().GetDims()[0]};

    // TEST: Try moving start padding to first dimension
    std::vector<int64_t> temp_pads = {pads[0], 0, 0, pads[1]};

    // Continue with the upgraded 2D processing
    return ComputePadding2DWorkaround(ctx, input_tensor, output_tensor, temp_input_dims, temp_output_dims, temp_pads, pad_value);
  }

  // Get MUSA stream
  musaStream_t stream = Stream(ctx);
  if (!stream) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to get MUSA stream");
  }

  // Create MUSA Handle and set stream
  ::musa::dnn::Handle handle(Info().GetExecutionProvider()->GetDeviceId());
  auto status = handle.SetStream(stream);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set MUSA stream on handle");
  }

  // Create input tensor descriptor
  ::musa::dnn::Tensor input_desc;

  status = input_desc.SetType(GetMusaTensorType<T>());
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set input tensor type");
  }

  std::vector<int64_t> input_dims_int64;
  for (auto dim : input_shape.GetDims()) {
    input_dims_int64.push_back(dim);
  }
  status = input_desc.SetFormat(::musa::dnn::Tensor::Format::NCHW);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set input tensor format");
  }

  status = input_desc.SetNdInfo(static_cast<int>(input_dims_int64.size()), input_dims_int64.data());
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set input tensor descriptor");
  }

  status = input_desc.SetAddr(const_cast<void*>(input_tensor.DataRaw()));
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set input tensor address");
  }

  // Create output tensor descriptor
  ::musa::dnn::Tensor output_desc;
  const auto& output_shape = output_tensor.Shape();

  status = output_desc.SetType(GetMusaTensorType<T>());
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set output tensor type");
  }

  std::vector<int64_t> output_dims_int64;
  for (auto dim : output_shape.GetDims()) {
    output_dims_int64.push_back(dim);
  }
  status = output_desc.SetFormat(::musa::dnn::Tensor::Format::NCHW);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set output tensor format");
  }

  status = output_desc.SetNdInfo(static_cast<int>(output_dims_int64.size()), output_dims_int64.data());
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set output tensor descriptor");
  }

  status = output_desc.SetAddr(output_tensor.MutableDataRaw());
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set output tensor address");
  }

  // Create and configure Pad operation
  ::musa::dnn::Pad pad_op;

  // Set padding mode
  status = pad_op.SetMode(GetMusaPadMode(mode_));
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set pad mode");
  }

  // Set padding value for constant mode
  if (mode_ == 0) {  // constant mode
    if constexpr (std::is_floating_point_v<T>) {
      status = pad_op.SetValue(static_cast<double>(pad_value));
    } else {
      status = pad_op.SetValue(static_cast<int64_t>(pad_value));
    }
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set pad value");
    }
  }

  // Convert from ONNX format to PyTorch-style format for MUSA
  // ONNX: [start_dim0, start_dim1, start_dim2, ..., end_dim0, end_dim1, end_dim2, ...]
  // PyTorch-style: From highest dim to lowest, each dim has start then end
  // 2D: [start_dim1, end_dim1, start_dim0, end_dim0] (left, right, top, bottom)
  // 3D: [start_dim2, end_dim2, start_dim1, end_dim1, start_dim0, end_dim0]
  std::vector<int> musa_pads;

  size_t num_dims = pads.size() / 2;  // Each dimension has start and end

  // Check for 4D limitation - MUSA API does not support 4D padding
  if (num_dims == 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "4D padding is not supported by MUSA API. Maximum supported dimensions: 3D");
  }

  if (num_dims >= 2 && num_dims <= 3) {
    // Apply PyTorch-style format conversion for 2D and 3D tensors
    // For each dimension from highest to lowest: add start_dim then end_dim
    for (int dim = num_dims - 1; dim >= 0; --dim) {
      size_t start_idx = dim;           // start_dim positions: 0, 1, 2, ...
      size_t end_idx = dim + num_dims;  // end_dim positions: num_dims, num_dims+1, ...

      musa_pads.push_back(static_cast<int>(pads[start_idx]));  // start_dim
      musa_pads.push_back(static_cast<int>(pads[end_idx]));    // end_dim
    }
  } else {
    // Fallback for other dimensions (1D uses 2D workaround)
    for (size_t i = 0; i < pads.size(); ++i) {
      musa_pads.push_back(static_cast<int>(pads[i]));
    }
  }

  status = pad_op.SetPaddingInfo(static_cast<int>(musa_pads.size()), musa_pads.data());
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set padding info");
  }

  // Execute the padding operation
  status = pad_op.Run(handle, output_desc, input_desc);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to execute pad operation, status: " + std::to_string(static_cast<int>(status)));
  }

  return Status::OK();
}

template <typename T>
Status Pad<T>::ComputeInternal(OpKernelContext* ctx) const {
  // Get input tensor
  const Tensor* input_tensor = ctx->Input<Tensor>(0);
  if (input_tensor == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input tensor is null");
  }

  // Extract padding parameters
  std::vector<int64_t> pads;
  T pad_value;

  if (is_dynamic_) {
    // Dynamic padding (v11+)
    ORT_RETURN_IF_ERROR(ExtractDynamicParams(ctx, pads, pad_value));
  } else {
    // Static padding (v2-10)
    pads = pads_;
    pad_value = value_;
  }

  // Validate padding
  ORT_RETURN_IF_ERROR(ValidatePadding(input_tensor->Shape(), pads));

  // Compute output shape
  auto output_dims = ComputeOutputShape(input_tensor->Shape(), pads);
  TensorShape output_shape(output_dims);

  // Create output tensor
  Tensor* output_tensor = ctx->Output(0, output_shape);
  if (output_tensor == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor");
  }

  // Handle empty input case
  if (input_tensor->Shape().Size() == 0) {
    // For empty input, handle constant mode by directly filling output
    if (mode_ == 0 && output_tensor->Shape().Size() > 0) {
      // Fill output tensor with pad_value using MUSA memory operations
      musaStream_t stream = Stream(ctx);
      if (!stream) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to get MUSA stream");
      }

      // For all types, handle padding by filling with appropriate pattern
      if constexpr (std::is_same_v<T, float>) {
        // For float, use proper value filling
        if (pad_value == 0.0f) {
          MUSA_RETURN_IF_ERROR(musaMemsetAsync(
              output_tensor->MutableDataRaw(),
              0,
              output_tensor->SizeInBytes(),
              stream));
        } else {
          // For non-zero float values, use a host-to-device copy approach
          // Create a host buffer with the correct value, then copy to device
          size_t num_elements = output_tensor->Shape().Size();
          std::vector<float> host_buffer(num_elements, pad_value);

          MUSA_RETURN_IF_ERROR(musaMemcpyAsync(
              output_tensor->MutableData<float>(),
              host_buffer.data(),
              output_tensor->SizeInBytes(),
              musaMemcpyHostToDevice,
              stream));
        }
      } else {
        // For other types, fill with zero pattern first
        MUSA_RETURN_IF_ERROR(musaMemsetAsync(
            output_tensor->MutableDataRaw(),
            0,
            output_tensor->SizeInBytes(),
            stream));

        // Check for non-zero padding values
        T zero_value;
        if constexpr (std::is_same_v<T, MLFloat16>) {
          zero_value = MLFloat16(0.0f);
        } else {
          zero_value = T{0};
        }

        if (pad_value != zero_value) {
          // For non-zero pad values of non-float types, also use host-to-device copy
          size_t num_elements = output_tensor->Shape().Size();
          std::vector<T> host_buffer(num_elements, pad_value);

          MUSA_RETURN_IF_ERROR(musaMemcpyAsync(
              output_tensor->MutableData<T>(),
              host_buffer.data(),
              output_tensor->SizeInBytes(),
              musaMemcpyHostToDevice,
              stream));
        }
      }

      printf("=== EMPTY TENSOR PADDING COMPLETED ===\n");
      return Status::OK();
    }

    // For non-constant modes with empty input, output should also be empty in most cases
    if (output_tensor->Shape().Size() == 0) {
      return Status::OK();
    }

    // If we reach here, it's an unsupported case
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "Non-constant mode padding of empty tensors not yet supported");
  }

  // Check for no-op case (all pads are zero)
  bool has_padding = false;
  for (auto pad : pads) {
    if (pad != 0) {
      has_padding = true;
      break;
    }
  }

  if (!has_padding) {
    // Just copy input to output
    MUSA_RETURN_IF_ERROR(musaMemcpyAsync(
        output_tensor->MutableDataRaw(),
        input_tensor->DataRaw(),
        input_tensor->SizeInBytes(),
        musaMemcpyDeviceToDevice,
        Stream(ctx)));
    return Status::OK();
  }

  // Perform the actual padding
  return ComputePadding(ctx, *input_tensor, *output_tensor, pads, pad_value);
}

// Register kernels for different ONNX versions
// Version 2-10: pads and value as attributes
#define REGISTER_MUSA_PAD_VERSIONED_KERNEL(start_ver, end_ver, T)      \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                             \
      Pad, kOnnxDomain, start_ver, end_ver, T, kMusaExecutionProvider, \
      (*KernelDefBuilder::Create())                                    \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),      \
      Pad<T>);

// Version 11+: pads and value as input tensors (need CPU memory constraint)
#define REGISTER_MUSA_PAD_DYNAMIC_KERNEL(start_ver, end_ver, T)        \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                             \
      Pad, kOnnxDomain, start_ver, end_ver, T, kMusaExecutionProvider, \
      (*KernelDefBuilder::Create())                                    \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())       \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                      \
          .InputMemoryType(OrtMemTypeCPUInput, 2),                     \
      Pad<T>);

// Version 18+: latest version with axes support
#define REGISTER_MUSA_PAD_LATEST_KERNEL(ver, T)                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                 \
      Pad, kOnnxDomain, ver, T, kMusaExecutionProvider,          \
      (*KernelDefBuilder::Create())                              \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()) \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                \
          .InputMemoryType(OrtMemTypeCPUInput, 3),               \
      Pad<T>);

// Register for version 2-10 (static parameters)
REGISTER_MUSA_PAD_VERSIONED_KERNEL(2, 10, float)
// V0.17.5/V0.17.6.1: muDNN 3.1+ supports Pad fp16, conditionally registered.
// Older SDKs (e.g. M1000 SDK 4.1.2 / muDNN 2.9.x) auto-fallback to CPU EP, same behavior as V0.17.1.
#if defined(MUDNN_VERSION) && (MUDNN_VERSION >= 3100)
REGISTER_MUSA_PAD_VERSIONED_KERNEL(2, 10, MLFloat16)
#endif

// Register for version 11-17 (dynamic parameters)
REGISTER_MUSA_PAD_DYNAMIC_KERNEL(11, 17, float)
#if defined(MUDNN_VERSION) && (MUDNN_VERSION >= 3100)
REGISTER_MUSA_PAD_DYNAMIC_KERNEL(11, 17, MLFloat16)
#endif

// Register for version 18+ (with axes support)
REGISTER_MUSA_PAD_LATEST_KERNEL(18, float)
#if defined(MUDNN_VERSION) && (MUDNN_VERSION >= 3100)
REGISTER_MUSA_PAD_LATEST_KERNEL(18, MLFloat16)
#endif

template <typename T>
Status Pad<T>::ComputePadding2DWorkaround(OpKernelContext* ctx,
                                          const Tensor& input_tensor,
                                          Tensor& output_tensor,
                                          const std::vector<int64_t>& temp_input_dims,
                                          const std::vector<int64_t>& temp_output_dims,
                                          const std::vector<int64_t>& temp_pads,
                                          T pad_value) const {
  // Get MUSA stream
  musaStream_t stream = Stream(ctx);
  if (!stream) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to get MUSA stream");
  }

  // Create MUSA Handle and set stream
  ::musa::dnn::Handle handle(Info().GetExecutionProvider()->GetDeviceId());
  auto status = handle.SetStream(stream);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set MUSA stream on handle");
  }

  // Create input tensor descriptor with upgraded 2D shape
  ::musa::dnn::Tensor input_desc;

  status = input_desc.SetType(GetMusaTensorType<T>());
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set input tensor type");
  }

  status = input_desc.SetFormat(::musa::dnn::Tensor::Format::NCHW);  // Set 2D format
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set input tensor format");
  }

  status = input_desc.SetNdInfo(static_cast<int>(temp_input_dims.size()), temp_input_dims.data());
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set input tensor descriptor");
  }

  status = input_desc.SetAddr(const_cast<void*>(input_tensor.DataRaw()));
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set input tensor address");
  }

  // Create output tensor descriptor with upgraded 2D shape
  ::musa::dnn::Tensor output_desc;

  status = output_desc.SetType(GetMusaTensorType<T>());
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set output tensor type");
  }

  status = output_desc.SetFormat(::musa::dnn::Tensor::Format::NCHW);  // Set 2D format
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set output tensor format");
  }

  status = output_desc.SetNdInfo(static_cast<int>(temp_output_dims.size()), temp_output_dims.data());
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set output tensor descriptor");
  }

  status = output_desc.SetAddr(output_tensor.MutableDataRaw());
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set output tensor address");
  }

  // Create and configure Pad operation
  ::musa::dnn::Pad pad_op;

  // Set padding mode
  status = pad_op.SetMode(GetMusaPadMode(mode_));
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set pad mode");
  }

  // Set padding value for constant mode
  if (mode_ == 0) {  // constant mode
    if constexpr (std::is_floating_point_v<T>) {
      status = pad_op.SetValue(static_cast<double>(pad_value));
    } else {
      status = pad_op.SetValue(static_cast<int64_t>(pad_value));
    }
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set pad value");
    }
  }

  // Convert padding format for 2D case
  // temp_pads format: [0, start, 0, end] (ONNX format for 2D)
  // MUSA format: [dim0_start, dim1_start, ..., dim0_end, dim1_end, ...]
  std::vector<int> musa_pads;

  // Convert padding format for 2D case
  for (size_t i = 0; i < temp_pads.size(); ++i) {
    musa_pads.push_back(static_cast<int>(temp_pads[i]));
  }

  status = pad_op.SetPaddingInfo(static_cast<int>(musa_pads.size()), musa_pads.data());
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set padding info");
  }

  // Execute the padding operation
  status = pad_op.Run(handle, output_desc, input_desc);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to execute pad operation");
  }

  return Status::OK();
}

// Explicit template instantiation for supported types
// Note: Avoid double and bfloat16 as mentioned in requirements
template class Pad<float>;
template class Pad<MLFloat16>;

}  // namespace musa
}  // namespace onnxruntime
