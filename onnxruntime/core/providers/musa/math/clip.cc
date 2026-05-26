// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/math/clip.h"
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/common.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/musa_execution_provider.h"
#include <musa_runtime.h>
#include <mudnn.h>

using onnxruntime::common::Status;
namespace onnxruntime {
namespace musa {

// MUSA device-based clip implementation using mudnn Unary class
template <typename T>
Status SimpleMusaClipOp(const MusaPreparation& prepare, T min_val, T max_val) {
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

  // Use mudnn Unary class for device computation
  try {
    // Create mudnn Unary operation
    ::musa::dnn::Unary unary_op;

    // Set the operation mode to CLIP
    auto status = unary_op.SetMode(::musa::dnn::Unary::Mode::CLIP);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set mudnn Unary mode to CLIP");
    }

    // Set Alpha (minimum value)
    status = unary_op.SetAlpha(static_cast<double>(min_val));
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set mudnn Unary alpha (min value)");
    }

    // Set Beta (maximum value)
    status = unary_op.SetBeta(static_cast<double>(max_val));
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set mudnn Unary beta (max value)");
    }

    // Run the unary operation directly on device
    status = unary_op.Run(prepare.GetHandle(),
                          const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]),  // output tensor
                          prepare.inputTensors[0]);  // input tensor

    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "mudnn Unary Clip operation failed, status: " +
                             std::to_string(static_cast<int>(status)));
    }

  } catch (const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Exception in mudnn Unary Clip operation: " +
                           std::string(e.what()));
  }

  return Status::OK();
}

// Prepare method for Clip_6 (versions 6-10)
template <typename T>
Status Clip_6<T>::Prepare(OpKernelContext* ctx, MusaPreparation& prepare) const {
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
    auto* stream = Stream(ctx);
    if (prepare.handle) {
      if (stream) {
        auto status = prepare.handle->SetStream(stream);
        if (status != ::musa::dnn::Status::SUCCESS) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                 "Failed to set MUSA stream, status: " +
                                 std::to_string(static_cast<int>(status)));
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
  ORT_CATCH(const std::exception &e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  return Status::OK();
}

// ComputeInternal method for Clip_6 (versions 6-10)
template <typename T>
Status Clip_6<T>::ComputeInternal(OpKernelContext* ctx) const {
  const auto* ep = static_cast<const MusaExecutionProvider*>(
      Info().GetExecutionProvider());
  MusaPreparation prepare(ep);
  ORT_RETURN_IF_ERROR(this->Prepare(ctx, prepare));
  ORT_RETURN_IF_ERROR(SimpleMusaClipOp<T>(prepare, this->min_, this->max_));
  return Status::OK();
}

// Prepare method for Clip (versions 11+)
template <typename T>
Status Clip::Prepare(OpKernelContext* ctx, MusaPreparation& prepare) const {
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
    auto* stream = Stream(ctx);
    if (prepare.handle) {
      if (stream) {
        auto status = prepare.handle->SetStream(stream);
        if (status != ::musa::dnn::Status::SUCCESS) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                 "Failed to set MUSA stream, status: " +
                                 std::to_string(static_cast<int>(status)));
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
  ORT_CATCH(const std::exception &e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  return Status::OK();
}

template <typename T>
struct Clip::ComputeImpl {
  Status operator()(const MusaPreparation& prepare, const Tensor* X, const Tensor* min, const Tensor* max, Tensor* Y) const {
    constexpr T min_default = std::numeric_limits<T>::lowest();
    constexpr T max_default = std::numeric_limits<T>::max();

    T min_val = min_default;
    T max_val = max_default;

    // Get min value
    if (min && min->Shape().Size() > 0) {
      ORT_ENFORCE(min->Shape().IsScalar(), "min should be a scalar.");
      min_val = *(min->Data<T>());
    }

    // Get max value
    if (max && max->Shape().Size() > 0) {
      ORT_ENFORCE(max->Shape().IsScalar(), "max should be a scalar.");
      max_val = *(max->Data<T>());
    }

    // Validate min <= max
    ORT_ENFORCE(min_val <= max_val, "min value must be less than or equal to max value");

    // Call the MUSA clip operation
    return SimpleMusaClipOp<T>(prepare, min_val, max_val);
  }
};

// Helper function template for Clip versions 11+
namespace musa_clip_internal {

template <typename T>
struct CallMusaClipImpl {
  Status operator()(OpKernelContext* ctx, const Clip* clip_op) const {
    
    const auto* X = ctx->Input<Tensor>(0);
    const auto* min = ctx->Input<Tensor>(1);
    const auto* max = ctx->Input<Tensor>(2);

    if (X->Shape().Size() == 0) {
      // Handle empty tensor case
      ctx->Output(0, X->Shape());
      return Status::OK();
    }

    const auto* ep = static_cast<const MusaExecutionProvider*>(
        clip_op->Info().GetExecutionProvider());
    MusaPreparation prepare(ep);
    ORT_RETURN_IF_ERROR(clip_op->Prepare<T>(ctx, prepare));

    // Extract min/max values
    T min_val = std::numeric_limits<T>::lowest();
    T max_val = std::numeric_limits<T>::max();

    if (min != nullptr && min->Shape().Size() > 0) {
      const T* min_data = min->Data<T>();
      min_val = *min_data;
    }

    if (max != nullptr && max->Shape().Size() > 0) {
      const T* max_data = max->Data<T>();
      max_val = *max_data;
    }

    return SimpleMusaClipOp<T>(prepare, min_val, max_val);
  }
};

}  // namespace musa_clip_internal

Status Clip::ComputeInternal(OpKernelContext* ctx) const {
  const auto* X = ctx->Input<Tensor>(0);
  if (X->Shape().Size() == 0) {
    // Handle empty tensor case
    ctx->Output(0, X->Shape());
    return Status::OK();
  }

  utils::MLTypeCallDispatcher<float, MLFloat16, int32_t, int64_t>
      t_disp(X->GetElementType());
  
  return t_disp.InvokeRet<Status, musa_clip_internal::CallMusaClipImpl>(ctx, this);
}

// Explicit template instantiations for supported types
template class Clip_6<float>;
template class Clip_6<MLFloat16>;
template class Clip_6<int32_t>;
template class Clip_6<int64_t>;

// Register kernels for different ONNX versions

// Register Clip_6 for versions 6-10 (attributes-based)
#define REGISTER_MUSA_CLIP6_VERSIONED_TYPED_KERNEL(startver, endver, T) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                              \
      Clip_6, kOnnxDomain, startver, endver, T, kMusaExecutionProvider, \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),       \
      Clip_6<T>);

// Register versions 6-10 for basic types (avoiding double and bfloat16)
REGISTER_MUSA_CLIP6_VERSIONED_TYPED_KERNEL(6, 10, float)
REGISTER_MUSA_CLIP6_VERSIONED_TYPED_KERNEL(6, 10, MLFloat16)

// Register Clip for versions 11+ (inputs-based)
#define REGISTER_MUSA_CLIP_VERSIONED_KERNEL(startver, endver)           \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                    \
      Clip, kOnnxDomain, startver, endver, kMusaExecutionProvider,      \
      (*KernelDefBuilder::Create())                                     \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                       \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                       \
          .TypeConstraint("T", BuildKernelDefConstraints<float, MLFloat16, int32_t, int64_t>()), \
      Clip);

#define REGISTER_MUSA_CLIP_KERNEL(ver)                                   \
  ONNX_OPERATOR_KERNEL_EX(                                              \
      Clip, kOnnxDomain, ver, kMusaExecutionProvider,                   \
      (*KernelDefBuilder::Create())                                     \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                       \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                       \
          .TypeConstraint("T", BuildKernelDefConstraints<float, MLFloat16, int32_t, int64_t>()), \
      Clip);

// Register version 11
REGISTER_MUSA_CLIP_VERSIONED_KERNEL(11, 11)
// Register version 12
REGISTER_MUSA_CLIP_VERSIONED_KERNEL(12, 12)
// Register version 13+
REGISTER_MUSA_CLIP_KERNEL(13)

}  // namespace musa
}  // namespace onnxruntime
