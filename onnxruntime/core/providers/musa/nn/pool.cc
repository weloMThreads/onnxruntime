// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/nn/pool.h"

using namespace onnxruntime::common;
namespace onnxruntime {
namespace musa {

#define POOLING_KERNEL(op_name, data_type, pool_type, since_version, op_domain, nhwc)              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                                    \
      op_name, op_domain, since_version, data_type, kMusaExecutionProvider,                        \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()), \
      Pool<data_type, pool_type, nhwc>);

#define POOLING_KERNEL_VERSIONED(op_name, data_type, pool_type, since_version, end_version, op_domain, nhwc)   \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(op_name, op_domain, since_version, end_version, data_type,            \
                                          kMusaExecutionProvider,                                                \
                                          (*KernelDefBuilder::Create())                                          \
                                              .TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()),    \
                                          Pool<data_type, pool_type, nhwc>);

#define POOLING_KERNEL_WITH_INDICES(op_name, data_type, pool_type, since_version, op_domain, nhwc)    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(op_name, op_domain, since_version, data_type, kMusaExecutionProvider, \
                                (*KernelDefBuilder::Create())                                           \
                                    .TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>())     \
                                    .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),      \
                                Pool<data_type, pool_type, nhwc>);

#define POOLING_KERNEL_VERSIONED_WITH_INDICES(op_name, data_type, pool_type, since_version, end_version, op_domain, \
                                              nhwc)                                                                  \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(op_name, op_domain, since_version, end_version, data_type,                 \
                                          kMusaExecutionProvider,                                                     \
                                          (*KernelDefBuilder::Create())                                               \
                                              .TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>())         \
                                              .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),          \
                                          Pool<data_type, pool_type, nhwc>);

POOLING_KERNEL_VERSIONED(AveragePool, float, AveragePool, 7, 9, kOnnxDomain, false)
POOLING_KERNEL_VERSIONED(AveragePool, MLFloat16, AveragePool, 7, 9, kOnnxDomain, false)
POOLING_KERNEL_VERSIONED(AveragePool, float, AveragePool, 10, 10, kOnnxDomain, false)
POOLING_KERNEL_VERSIONED(AveragePool, MLFloat16, AveragePool, 10, 10, kOnnxDomain, false)
POOLING_KERNEL(AveragePool, float, AveragePool, 11, kOnnxDomain, false)
POOLING_KERNEL(AveragePool, MLFloat16, AveragePool, 11, kOnnxDomain, false)
POOLING_KERNEL(GlobalAveragePool, float, AveragePool, 1, kOnnxDomain, false)
POOLING_KERNEL(GlobalAveragePool, MLFloat16, AveragePool, 1, kOnnxDomain, false)
POOLING_KERNEL_VERSIONED(MaxPool, float, MaxPool<1>, 1, 7, kOnnxDomain, false)
POOLING_KERNEL_VERSIONED(MaxPool, MLFloat16, MaxPool<1>, 1, 7, kOnnxDomain, false)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, float, MaxPool<8>, 8, 9, kOnnxDomain, false)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, MLFloat16, MaxPool<8>, 8, 9, kOnnxDomain, false)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, float, MaxPool<8>, 10, 10, kOnnxDomain, false)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, MLFloat16, MaxPool<8>, 10, 10, kOnnxDomain, false)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, float, MaxPool<8>, 11, 11, kOnnxDomain, false)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, MLFloat16, MaxPool<8>, 11, 11, kOnnxDomain, false)
POOLING_KERNEL_WITH_INDICES(MaxPool, float, MaxPool<8>, 12, kOnnxDomain, false)
POOLING_KERNEL_WITH_INDICES(MaxPool, MLFloat16, MaxPool<8>, 12, kOnnxDomain, false)
POOLING_KERNEL(GlobalMaxPool, float, MaxPool<1>, 1, kOnnxDomain, false)
POOLING_KERNEL(GlobalMaxPool, MLFloat16, MaxPool<1>, 1, kOnnxDomain, false)

#ifdef ENABLE_MUSA_NHWC_OPS
POOLING_KERNEL_VERSIONED(MaxPool, float, MaxPool<1>, 1, 7, kMSInternalNHWCDomain, true)
POOLING_KERNEL_VERSIONED(MaxPool, MLFloat16, MaxPool<1>, 1, 7, kMSInternalNHWCDomain, true)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, float, MaxPool<8>, 8, 9, kMSInternalNHWCDomain, true)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, MLFloat16, MaxPool<8>, 8, 9, kMSInternalNHWCDomain, true)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, float, MaxPool<8>, 10, 10, kMSInternalNHWCDomain, true)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, MLFloat16, MaxPool<8>, 10, 10, kMSInternalNHWCDomain, true)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, float, MaxPool<8>, 11, 11, kMSInternalNHWCDomain, true)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, MLFloat16, MaxPool<8>, 11, 11, kMSInternalNHWCDomain, true)
POOLING_KERNEL_WITH_INDICES(MaxPool, float, MaxPool<8>, 12, kMSInternalNHWCDomain, true)
POOLING_KERNEL_WITH_INDICES(MaxPool, MLFloat16, MaxPool<8>, 12, kMSInternalNHWCDomain, true)
POOLING_KERNEL(GlobalMaxPool, float, MaxPool<1>, 1, kMSInternalNHWCDomain, true)
POOLING_KERNEL(GlobalMaxPool, MLFloat16, MaxPool<1>, 1, kMSInternalNHWCDomain, true)
POOLING_KERNEL_VERSIONED(AveragePool, float, AveragePool, 7, 9, kMSInternalNHWCDomain, true)
POOLING_KERNEL_VERSIONED(AveragePool, MLFloat16, AveragePool, 7, 9, kMSInternalNHWCDomain, true)
POOLING_KERNEL_VERSIONED(AveragePool, float, AveragePool, 10, 10, kMSInternalNHWCDomain, true)
POOLING_KERNEL_VERSIONED(AveragePool, MLFloat16, AveragePool, 10, 10, kMSInternalNHWCDomain, true)
POOLING_KERNEL(AveragePool, float, AveragePool, 11, kMSInternalNHWCDomain, true)
POOLING_KERNEL(AveragePool, MLFloat16, AveragePool, 11, kMSInternalNHWCDomain, true)
POOLING_KERNEL(GlobalAveragePool, float, AveragePool, 1, kMSInternalNHWCDomain, true)
POOLING_KERNEL(GlobalAveragePool, MLFloat16, AveragePool, 1, kMSInternalNHWCDomain, true)
#endif

}  // namespace musa
}  // namespace onnxruntime
