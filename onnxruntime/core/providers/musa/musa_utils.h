// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/shared_library/provider_api.h"
#include <memory>
#include <mudnn.h>
#include <musa_runtime.h>
#include <vector>

namespace onnxruntime {

// Forward declaration for MusaExecutionProvider
class MusaExecutionProvider;

namespace musa {

constexpr bool LAYOUT_NCHW = false;
constexpr bool LAYOUT_NHWC = true;

template <bool IsNHWC>
struct LayoutDims;

template <>
struct LayoutDims<LAYOUT_NHWC> {
  static constexpr size_t N = 0;
  static constexpr size_t C = 3;  // Last dimension for NHWC
};

template <>
struct LayoutDims<LAYOUT_NCHW> {
  static constexpr size_t N = 0;
  static constexpr size_t C = 1;  // Second dimension for NCHW
};

inline std::vector<int64_t> CalculateStridesNHWC(const TensorShape& shape) {
  size_t rank = shape.NumDimensions();
  std::vector<int64_t> strides(rank);
  if (rank == 0) return strides;
  
  // NHWC format: strides[C] = 1, then work backwards
  // For [N, H, W, C]: strides = [H*W*C, W*C, C, 1]
  strides[rank - 1] = 1;
  for (int64_t i = static_cast<int64_t>(rank) - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

inline bool IsNHWCFormat(::musa::dnn::Tensor::Format format) {
  return format == ::musa::dnn::Tensor::Format::NHWC;
}

// ============================================================================
// NHWC Shape Conversion Utilities
// ============================================================================

// Convert NCHW shape to NHWC shape
// NCHW: [N, C, H, W, ...] -> NHWC: [N, H, W, ..., C]
inline std::vector<int64_t> GetNHWCShape(const std::vector<int64_t>& nchw_shape) {
  if (nchw_shape.size() < 2) return nchw_shape;
  std::vector<int64_t> nhwc_shape = nchw_shape;
  size_t rank = nchw_shape.size();
  for (size_t i = 1; i < rank - 1; ++i) {
    nhwc_shape[i] = nchw_shape[i + 1];  // H, W, ...
  }
  nhwc_shape[rank - 1] = nchw_shape[1];  // C at the end
  return nhwc_shape;
}

// Convert NHWC shape back to NCHW shape
// NHWC: [N, H, W, ..., C] -> NCHW: [N, C, H, W, ...]
inline std::vector<int64_t> GetNCHWShapeFromNHWC(const std::vector<int64_t>& nhwc_shape) {
  if (nhwc_shape.size() < 2) return nhwc_shape;
  std::vector<int64_t> nchw_shape = nhwc_shape;
  size_t rank = nhwc_shape.size();
  nchw_shape[1] = nhwc_shape[rank - 1];  // C at position 1
  for (size_t i = 2; i < rank; ++i) {
    nchw_shape[i] = nhwc_shape[i - 1];  // H, W, ...
  }
  return nchw_shape;
}

// Get permutation for converting Conv weight from NCHW to NHWC
// NCHW weight: [M, C/group, kH, kW]
// NHWC weight: [M, kH, kW, C/group]
inline std::vector<size_t> GetNHWCWeightPerm(size_t rank) {
  std::vector<size_t> perm;
  perm.push_back(0);  // M stays at position 0
  for (size_t i = 2; i < rank; i++) {
    perm.push_back(i);  // Spatial dims (H, W, ...) move forward
  }
  perm.push_back(1);  // C moves to the end
  return perm;
}

struct MusaPreparation {
  // ============================================================================
  // Constructors
  // ============================================================================

  // Legacy constructor: creates its own Handle (DEPRECATED)
  // WARNING: This creates a new Handle every time, which is inefficient and
  // NOT thread-safe. Use the EP-based constructor for new code.
  // DEPRECATED: Use MusaPreparation(const MusaExecutionProvider* ep) instead.
  [[deprecated("Use MusaPreparation(const MusaExecutionProvider* ep) instead")]]
  MusaPreparation()
      : input_a_ptr(nullptr), input_b_ptr(nullptr), input_c_ptr(nullptr), bias_ptr(nullptr), output_ptr(nullptr),
        output_size(0), processed_indices_ptr(nullptr), handle_ptr_(nullptr) {
    // TODO: ep-aware after migration
    musaSetDevice(0);

    // Initialize MUSA handle (legacy mode)
    handle = std::make_unique<::musa::dnn::Handle>();
    if (!handle) {
      ORT_THROW("Failed to create MUSA handle");
    }
  }

  // New constructor: gets Handle from PerThreadContext (recommended)
  // This is thread-safe and reuses Handle across kernel executions.
  // Implementation in musa_utils.cc
  explicit MusaPreparation(const MusaExecutionProvider* ep);

  ~MusaPreparation() {
    // Clean up allocated GPU memory for processed indices
    if (processed_indices_ptr != nullptr) {
      musaFree(processed_indices_ptr);
      processed_indices_ptr = nullptr;
    }
    // Note: handle is auto-cleaned by unique_ptr (legacy mode)
    // Note: handle_ptr_ is NOT owned, no cleanup needed (EP mode)
  }

  // ============================================================================
  // Handle Access
  // ============================================================================

  // Unified access to Handle (works for both legacy and EP modes)
  ::musa::dnn::Handle& GetHandle() const {
    if (handle_ptr_) {
      return *handle_ptr_;  // EP mode: from PerThreadContext
    }
    return *handle;  // Legacy mode: owned handle
  }

  // ============================================================================
  // Public Members (kept for backward compatibility)
  // ============================================================================

  // Legacy public member - kept for backward compatibility
  // Existing kernels can still use: prepare.handle->...
  std::unique_ptr<::musa::dnn::Handle> handle;

  std::vector<::musa::dnn::Tensor> inputTensors;
  std::vector<::musa::dnn::Tensor> outputTensors;

  // Tensor data pointers for direct access
  const void *input_a_ptr;
  const void *input_b_ptr;
  const void *input_c_ptr;  // Added for third input tensor (values)
  const void *bias_ptr;  // Added for bias tensor access
  void *output_ptr;
  size_t output_size;

  // Shape information for broadcasting
  TensorShape input_a_shape;
  TensorShape input_b_shape;
  TensorShape input_c_shape;  // Added for third input tensor shape
  TensorShape output_shape;

  // GPU memory for processed indices
  void* processed_indices_ptr = nullptr;

 private:
  // ============================================================================
  // Private Members
  // ============================================================================

  // EP mode: borrowed Handle pointer from PerThreadContext (NOT owned)
  ::musa::dnn::Handle* handle_ptr_ = nullptr;
};

// Data type conversion functions
template <typename T>
::musa::dnn::Tensor::Type GetMusaDataType();

template <> inline ::musa::dnn::Tensor::Type GetMusaDataType<float>() {
  return ::musa::dnn::Tensor::Type::FLOAT;
}

template <> inline ::musa::dnn::Tensor::Type GetMusaDataType<int32_t>() {
  return ::musa::dnn::Tensor::Type::INT32;
}

template <> inline ::musa::dnn::Tensor::Type GetMusaDataType<int64_t>() {
  return ::musa::dnn::Tensor::Type::INT64;
}

template <> inline ::musa::dnn::Tensor::Type GetMusaDataType<uint8_t>() {
  return ::musa::dnn::Tensor::Type::UINT8;
}

template <> inline ::musa::dnn::Tensor::Type GetMusaDataType<int8_t>() {
  return ::musa::dnn::Tensor::Type::INT8;
}

template <> inline ::musa::dnn::Tensor::Type GetMusaDataType<int16_t>() {
  return ::musa::dnn::Tensor::Type::INT16;
}

template <> inline ::musa::dnn::Tensor::Type GetMusaDataType<MLFloat16>() {
  return ::musa::dnn::Tensor::Type::HALF;
}

template <> inline ::musa::dnn::Tensor::Type GetMusaDataType<double>() {
  return ::musa::dnn::Tensor::Type::DOUBLE;
}

template <> inline ::musa::dnn::Tensor::Type GetMusaDataType<BFloat16>() {
  return ::musa::dnn::Tensor::Type::BFLOAT16;
}

// Add missing data type specializations for Transpose operator
template <> inline ::musa::dnn::Tensor::Type GetMusaDataType<uint16_t>() {
  return ::musa::dnn::Tensor::Type::UINT16;
}

template <> inline ::musa::dnn::Tensor::Type GetMusaDataType<uint32_t>() {
  return ::musa::dnn::Tensor::Type::UINT32;
}

template <> inline ::musa::dnn::Tensor::Type GetMusaDataType<uint64_t>() {
  return ::musa::dnn::Tensor::Type::UINT64;
}

template <> inline ::musa::dnn::Tensor::Type GetMusaDataType<bool>() {
  // Use proper BOOL type for mudnn Binary comparison operations
  return ::musa::dnn::Tensor::Type::BOOL;
}

// Utility functions
template <typename T>
Status BroadcastTensor(const Tensor *input, const Tensor *output,
                       void *output_data, musaStream_t stream);

// Calculate strides for tensor
inline std::vector<int64_t> CalculateStrides(const TensorShape &shape) {
  std::vector<int64_t> strides(shape.NumDimensions());
  if (shape.NumDimensions() == 0) {
    return strides;
  }

  strides[shape.NumDimensions() - 1] = 1;
  for (int64_t i = static_cast<int64_t>(shape.NumDimensions()) - 2; i >= 0;
       --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

// Helper function to create MUSA tensor from ORT tensor
inline Status SetupMusaTensor(::musa::dnn::Tensor &musa_tensor,
                              const Tensor *ort_tensor,
                              ::musa::dnn::Tensor::Type data_type,
                              MusaPreparation *preparation = nullptr
                            ) {

  // Set tensor type
  auto status = musa_tensor.SetType(data_type);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to set MUSA tensor type, status: " +
                               std::to_string(static_cast<int>(status)));
  }

  // Set data address - allow null for empty tensors
  const void *data_ptr = ort_tensor->DataRaw();
  const auto &shape = ort_tensor->Shape();

  if (!data_ptr && shape.Size() > 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "ORT tensor data pointer is null for non-empty tensor");
  }

  // Only set address if tensor is not empty
  if (data_ptr) {
    status = musa_tensor.SetAddr(data_ptr);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set MUSA tensor address, status: " +
                                 std::to_string(static_cast<int>(status)));
    }
  }

  // Get shape information (already defined above)

  // Choose format based on tensor dimensions
  ::musa::dnn::Tensor::Format format = ::musa::dnn::Tensor::Format::UNKNOWN;
  if (shape.NumDimensions() == 0) {
    // Scalar case - use NCW format
    format = ::musa::dnn::Tensor::Format::NCW;
  } else if (shape.NumDimensions() == 1) {
    // 1D tensor - use NCW format (like the working test_mudnn_binary.cpp)
    format = ::musa::dnn::Tensor::Format::NCW;
  } else if (shape.NumDimensions() == 3) {
    // 3D tensor (for 1D convolution) - use NCW format
    format = ::musa::dnn::Tensor::Format::NCW;
  } else if (shape.NumDimensions() == 4) {
    // 4D tensor (for 2D convolution) - use NCHW format
    format = ::musa::dnn::Tensor::Format::NCHW;
  } else if (shape.NumDimensions() == 5) {
    // 5D tensor (for 3D convolution) - use NCDHW format
    format = ::musa::dnn::Tensor::Format::NCDHW;
  } else {
    // Default to NCHW for other cases
    format = ::musa::dnn::Tensor::Format::NCHW;
  }

    // Set tensor format
  status = musa_tensor.SetFormat(format);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to set MUSA tensor format, status: " +
                               std::to_string(static_cast<int>(status)));
  }



  if (shape.NumDimensions() == 0) {
    // Handle scalar case
    std::vector<int64_t> scalar_dims = {1};
    std::vector<int64_t> scalar_strides = {1};
    status = musa_tensor.SetNdInfo(static_cast<int>(scalar_dims.size()), scalar_dims.data(), scalar_strides.data());
  } else {
    // Handle normal tensor case
    std::vector<int64_t> dims;
    for (size_t i = 0; i < shape.NumDimensions(); ++i) {
      dims.push_back(shape[i]);
    }

    const auto strides = CalculateStrides(shape);
    status = musa_tensor.SetNdInfo(static_cast<int>(dims.size()), dims.data(), strides.data());
  }

  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to set MUSA tensor shape info, status: " +
                               std::to_string(static_cast<int>(status)));
  }

  return Status::OK();
}

// Convert ONNX data type to MUSA data type for Cast operation
inline ::musa::dnn::Tensor::Type GetMusaDataTypeFromOnnx(ONNX_NAMESPACE::TensorProto_DataType onnx_type) {
  switch (onnx_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return ::musa::dnn::Tensor::Type::FLOAT;
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      return ::musa::dnn::Tensor::Type::DOUBLE;
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      return ::musa::dnn::Tensor::Type::INT8;
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
      return ::musa::dnn::Tensor::Type::INT16;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      return ::musa::dnn::Tensor::Type::INT32;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      return ::musa::dnn::Tensor::Type::INT64;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      return ::musa::dnn::Tensor::Type::UINT8;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
      return ::musa::dnn::Tensor::Type::UINT16;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      return ::musa::dnn::Tensor::Type::UINT32;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
      return ::musa::dnn::Tensor::Type::UINT64;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return ::musa::dnn::Tensor::Type::HALF;
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      return ::musa::dnn::Tensor::Type::UINT8;  // MUSA represents bool as uint8
    default:
      return ::musa::dnn::Tensor::Type::FLOAT;  // Default fallback
  }
}

} // namespace musa
} // namespace onnxruntime
