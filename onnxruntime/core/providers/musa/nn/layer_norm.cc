// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/nn/layer_norm.h"
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/musa_call.h"
#include "core/providers/musa/musa_execution_provider.h"
#include "core/providers/cpu/nn/layer_norm_helper.h"
#include "core/providers/common.h"
#include <musa_runtime.h>
#include <mudnn.h>
#include <numeric>
#include <vector>

namespace onnxruntime {
namespace musa {

// Helper function to setup muTensor with explicit format and dimensions
static Status SetupLayerNormMusaTensor(::musa::dnn::Tensor& musa_tensor,
                                       const void* data_ptr,
                                       const TensorShape& shape,
                                       ::musa::dnn::Tensor::Type data_type) {
  auto status = musa_tensor.SetType(data_type);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set MUSA tensor type for LayerNorm");
  }

  if (data_ptr) {
    status = musa_tensor.SetAddr(data_ptr);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set MUSA tensor address for LayerNorm");
    }
  }

  // Choose format based on dimensions
  ::musa::dnn::Tensor::Format format;
  if (shape.NumDimensions() <= 2) {
    format = ::musa::dnn::Tensor::Format::NCW;
  } else if (shape.NumDimensions() == 3) {
    format = ::musa::dnn::Tensor::Format::NCW;
  } else if (shape.NumDimensions() == 4) {
    format = ::musa::dnn::Tensor::Format::NCHW;
  } else {
    format = ::musa::dnn::Tensor::Format::NCHW;
  }

  status = musa_tensor.SetFormat(format);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set MUSA tensor format for LayerNorm");
  }

  // Set dimensions
  if (shape.NumDimensions() == 0) {
    std::vector<int64_t> scalar_dims = {1};
    status = musa_tensor.SetNdInfo(1, scalar_dims.data());
  } else {
    std::vector<int64_t> dims(shape.GetDims().begin(), shape.GetDims().end());
    status = musa_tensor.SetNdInfo(static_cast<int>(dims.size()), dims.data());
  }

  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set MUSA tensor shape for LayerNorm");
  }

  return Status::OK();
}

template <typename T, typename U, typename V, bool simplified>
LayerNorm<T, U, V, simplified>::LayerNorm(const OpKernelInfo& op_kernel_info) : MusaKernel(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr("axis", &axis_).IsOK());
  float tmp_epsilon;
  ORT_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &tmp_epsilon).IsOK());
  epsilon_ = tmp_epsilon;
}

template <typename T, typename U, typename V, bool simplified>
Status LayerNorm<T, U, V, simplified>::ComputeInternal(OpKernelContext* ctx) const {
  // Get inputs
  const Tensor* X = ctx->Input<Tensor>(0);
  const Tensor* scale = ctx->Input<Tensor>(1);
  const Tensor* bias = simplified ? nullptr : ctx->Input<Tensor>(2);

  const TensorShape& x_shape = X->Shape();
  auto x_num_dims = x_shape.NumDimensions();
  const int64_t axis = HandleNegativeAxis(axis_, x_num_dims);

  // Validate inputs using helper
  const TensorShape& scale_shape = scale->Shape();
  const TensorShape& bias_shape = bias ? bias->Shape() : TensorShape();

  LayerNormParams params;
  ORT_RETURN_IF_ERROR(LayerNormHelper::CheckInputs(x_shape, scale_shape, bias_shape, bias != nullptr, axis, params));

  // Handle empty tensor
  if (x_shape.Size() == 0) {
    return Status::OK();
  }

  // Allocate output Y
  Tensor* Y = ctx->Output(0, x_shape);

  // Compute mean and inv_std_dev shape: X.shape[:axis] + [1]*(rank-axis)
  std::vector<int64_t> mean_inv_std_var_dim;
  for (size_t i = 0; i < x_num_dims; ++i) {
    if (static_cast<int64_t>(i) < axis) {
      mean_inv_std_var_dim.emplace_back(x_shape.GetDims()[i]);
    } else {
      mean_inv_std_var_dim.emplace_back(1);
    }
  }

  // Handle optional outputs
  int output_index = 1;
  Tensor* mean_output = nullptr;
  if (!simplified) {
    mean_output = ctx->Output(output_index++, TensorShape(mean_inv_std_var_dim));
  }
  Tensor* inv_std_dev_output = ctx->Output(output_index, TensorShape(mean_inv_std_var_dim));

  // Get EP for MusaPreparation
  const auto* ep = static_cast<const MusaExecutionProvider*>(Info().GetExecutionProvider());
  MusaPreparation prepare(ep);

  // Determine data type for MUSA tensors
  const auto inputMusaType = GetMusaDataType<T>();
  const auto scaleMusaType = GetMusaDataType<V>();
  const auto statMusaType = GetMusaDataType<U>();

  // Create input muTensor for X
  ::musa::dnn::Tensor mu_input;
  ORT_RETURN_IF_ERROR(SetupLayerNormMusaTensor(mu_input, X->DataRaw(), x_shape, inputMusaType));

  // Create output muTensor for Y
  ::musa::dnn::Tensor mu_output;
  ORT_RETURN_IF_ERROR(SetupLayerNormMusaTensor(mu_output, Y->MutableDataRaw(), x_shape, scaleMusaType));

  // Create muTensor for scale (gamma)
  ::musa::dnn::Tensor mu_scale;
  ORT_RETURN_IF_ERROR(SetupLayerNormMusaTensor(mu_scale, scale->DataRaw(), scale_shape, scaleMusaType));

  // Create muTensor for bias (beta)
  // Key insight from torch_musa: mudnn requires both gamma and beta
  // If bias is not provided, create a zero-filled tensor
  ::musa::dnn::Tensor mu_bias;
  IAllocatorUniquePtr<V> zero_bias_buffer;

  if (bias != nullptr) {
    ORT_RETURN_IF_ERROR(SetupLayerNormMusaTensor(mu_bias, bias->DataRaw(), bias_shape, scaleMusaType));
  } else {
    size_t bias_count = scale->Shape().Size();
    zero_bias_buffer = GetScratchBuffer<V>(bias_count, ctx->GetComputeStream());
    auto musa_err = musaMemsetAsync(zero_bias_buffer.get(), 0, bias_count * sizeof(V), Stream(ctx));
    if (musa_err != musaSuccess) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "musaMemsetAsync failed for zero bias buffer, error: ",
                             static_cast<int>(musa_err));
    }
    ORT_RETURN_IF_ERROR(SetupLayerNormMusaTensor(mu_bias, zero_bias_buffer.get(), scale_shape, scaleMusaType));
  }

  // Create muTensor for mean output
  // mudnn requires mean tensor even if ONNX doesn't output it
  ::musa::dnn::Tensor mu_mean;
  IAllocatorUniquePtr<U> temp_mean_buffer;
  TensorShape mean_shape(mean_inv_std_var_dim);

  if (mean_output != nullptr) {
    ORT_RETURN_IF_ERROR(SetupLayerNormMusaTensor(mu_mean, mean_output->MutableDataRaw(), mean_shape, statMusaType));
  } else {
    size_t mean_count = mean_shape.Size();
    temp_mean_buffer = GetScratchBuffer<U>(mean_count, ctx->GetComputeStream());
    ORT_RETURN_IF_ERROR(SetupLayerNormMusaTensor(mu_mean, temp_mean_buffer.get(), mean_shape, statMusaType));
  }

  // Create muTensor for inv_std_dev output
  // mudnn requires inv_var tensor even if ONNX doesn't output it
  ::musa::dnn::Tensor mu_inv_var;
  IAllocatorUniquePtr<U> temp_inv_var_buffer;

  if (inv_std_dev_output != nullptr) {
    ORT_RETURN_IF_ERROR(SetupLayerNormMusaTensor(mu_inv_var, inv_std_dev_output->MutableDataRaw(), mean_shape, statMusaType));
  } else {
    size_t inv_var_count = mean_shape.Size();
    temp_inv_var_buffer = GetScratchBuffer<U>(inv_var_count, ctx->GetComputeStream());
    ORT_RETURN_IF_ERROR(SetupLayerNormMusaTensor(mu_inv_var, temp_inv_var_buffer.get(), mean_shape, statMusaType));
  }

  // Create mudnn LayerNorm operation
  ::musa::dnn::LayerNorm ln_op;

  // Set epsilon
  auto status = ln_op.SetEpsilon(epsilon_);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "mudnn LayerNorm SetEpsilon failed, status: ",
                           static_cast<int>(status));
  }

  // Set VarMode based on data type
  // Float32 uses WELFORD for numerical stability, others use DIRECT for performance
#if defined(MUDNN_VERSION) && (MUDNN_VERSION >= 3100)
  if (std::is_same<T, float>::value) {
    status = ln_op.SetVarMode(::musa::dnn::LayerNorm::VarMode::WELFORD);
  } else {
    status = ln_op.SetVarMode(::musa::dnn::LayerNorm::VarMode::DIRECT);
  }
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "mudnn LayerNorm SetVarMode failed, status: ",
                           static_cast<int>(status));
  }
#endif

  // Convert ONNX axis to mudnn axes array
  // ONNX: normalize from axis to end (axis, axis+1, ..., rank-1)
  // mudnn: SetAxis takes array of axes to normalize
  std::vector<int> axes;
  for (int64_t i = axis; i < static_cast<int64_t>(x_num_dims); ++i) {
    axes.push_back(static_cast<int>(i));
  }
  status = ln_op.SetAxis(axes.size(), axes.data());
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "mudnn LayerNorm SetAxis failed, status: ",
                           static_cast<int>(status));
  }

  // Get Handle from MusaPreparation
  auto& handle = prepare.GetHandle();

  // Set stream
  status = handle.SetStream(Stream(ctx));
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "mudnn Handle SetStream failed, status: ",
                           static_cast<int>(status));
  }

  // Run LayerNorm
  // API: Run(handle, output, mean, inv_var, input, gamma, beta, maintainer)
  status = ln_op.Run(handle, mu_output, mu_mean, mu_inv_var, mu_input, mu_scale, mu_bias);

  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "mudnn LayerNorm Run failed, status: ",
                           static_cast<int>(status));
  }

  return Status::OK();
}

// Explicit template instantiations for supported types
// Note: MusaEP does NOT support double type
template class LayerNorm<float, float, float, false>;
template class LayerNorm<MLFloat16, float, MLFloat16, false>;
template class LayerNorm<BFloat16, float, BFloat16, false>;

// ============================================================================
// Kernel Registration - First Layer of Three-Layer Architecture
// ============================================================================

// LayerNormalization is an official ONNX operator in opset 17, but it was
// historically registered as a contrib op in kOnnxDomain for opsets 1-16.
// Note: MusaEP does NOT register double type
#define REGISTER_CONTRIB_KERNEL_TYPED(T, U, V)                                                                  \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(LayerNormalization, kOnnxDomain, 1, 16, T##_##U##_##V,                 \
                                          kMusaExecutionProvider,                                                \
                                          (*KernelDefBuilder::Create())                                          \
                                              .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())             \
                                              .TypeConstraint("U", DataTypeImpl::GetTensorType<U>())             \
                                              .TypeConstraint("V", DataTypeImpl::GetTensorType<V>()),            \
                                          LayerNorm<T, U, V, false>);

#define REGISTER_KERNEL_TYPED(T, U)                                                             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(LayerNormalization, kOnnxDomain, 17, T, kMusaExecutionProvider, \
                                (*KernelDefBuilder::Create())                                   \
                                    .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())      \
                                    .TypeConstraint("U", DataTypeImpl::GetTensorType<U>()),     \
                                LayerNorm<T, U, T, false>);

REGISTER_CONTRIB_KERNEL_TYPED(float, float, float)
REGISTER_CONTRIB_KERNEL_TYPED(MLFloat16, float, MLFloat16)
REGISTER_CONTRIB_KERNEL_TYPED(BFloat16, float, BFloat16)

REGISTER_KERNEL_TYPED(float, float)
REGISTER_KERNEL_TYPED(MLFloat16, float)
REGISTER_KERNEL_TYPED(BFloat16, float)

}  // namespace musa
}  // namespace onnxruntime
