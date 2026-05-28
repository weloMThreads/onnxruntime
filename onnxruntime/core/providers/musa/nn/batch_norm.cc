// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/nn/batch_norm.h"

#include <musa_runtime.h>
#include <type_traits>

#include "core/providers/common.h"
#include "core/providers/cpu/nn/batch_norm_helper.h"
#include "core/providers/musa/musa_execution_provider.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/nn/batch_norm_impl.h"
#include "core/providers/shared_library/provider_api.h"

using onnxruntime::common::Status;

namespace onnxruntime {
namespace musa {

template <typename T>
BatchNormalization<T>::BatchNormalization(const OpKernelInfo& info)
    : MusaKernel(info),
      epsilon_(info.GetAttrOrDefault<float>("epsilon", 1e-5f)),
      is_train_(info.GetAttrOrDefault<int64_t>("training_mode", 0) == 1),
      output_count_(info.GetOutputCount()) {}

template <typename T>
Status BatchNormalization<T>::ComputeInternal(OpKernelContext* ctx) const {
  if (is_train_ || output_count_ != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "MUSA BatchNormalization only supports inference mode with one output");
  }

  const Tensor* X = ctx->Input<Tensor>(0);
  const Tensor* scale = ctx->Input<Tensor>(1);
  const Tensor* B = ctx->Input<Tensor>(2);
  const Tensor* mean = ctx->Input<Tensor>(3);
  const Tensor* var = ctx->Input<Tensor>(4);
  ORT_RETURN_IF_ERROR(BatchNormHelper::ValidateInputs(X, scale, B, mean, var, true));

  Tensor* Y = ctx->Output(0, X->Shape());
  ORT_RETURN_IF_NOT(Y != nullptr, "BatchNormalization output tensor is null");

  const TensorShape& x_shape = X->Shape();
  const int64_t total_elements = x_shape.Size();
  if (total_elements == 0) {
    return Status::OK();
  }

  const auto& dims = x_shape.GetDims();
  const int64_t channels = dims.size() == 1 ? 1 : dims[1];
  int64_t spatial_size = 1;
  for (size_t i = 2; i < dims.size(); ++i) {
    spatial_size *= dims[i];
  }

  BatchNormalizationParams params{total_elements, channels, spatial_size, epsilon_};

  musaError_t status = musaSuccess;
  if constexpr (std::is_same_v<T, MLFloat16>) {
    status = LaunchBatchNormalizationHalf(Stream(ctx),
                                          X->DataRaw(),
                                          scale->DataRaw(),
                                          B->DataRaw(),
                                          mean->DataRaw(),
                                          var->DataRaw(),
                                          Y->MutableDataRaw(),
                                          params);
  } else {
    status = LaunchBatchNormalizationFloat(Stream(ctx),
                                           X->Data<float>(),
                                           scale->Data<float>(),
                                           B->Data<float>(),
                                           mean->Data<float>(),
                                           var->Data<float>(),
                                           Y->MutableData<float>(),
                                           params);
  }

  if (status != musaSuccess) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "BatchNormalization MUSA kernel launch failed, status: ",
                           static_cast<int>(status));
  }

  return Status::OK();
}

#define REGISTER_MUSA_BATCHNORM_TYPED_KERNEL(ver, T)                          \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                              \
      BatchNormalization,                                                     \
      kOnnxDomain,                                                            \
      ver,                                                                    \
      T,                                                                      \
      kMusaExecutionProvider,                                                 \
      (*KernelDefBuilder::Create())                                           \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())             \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())            \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()),           \
      BatchNormalization<T>);

REGISTER_MUSA_BATCHNORM_TYPED_KERNEL(15, float)
REGISTER_MUSA_BATCHNORM_TYPED_KERNEL(15, MLFloat16)

}  // namespace musa
}  // namespace onnxruntime
