// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/tensor/identity_op.h"

namespace onnxruntime {
namespace musa {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Identity,
    kOnnxDomain,
    1, 12,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .Alias(0, 0),
    IdentityOp<false>);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Identity,
    kOnnxDomain,
    13, 13,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .Alias(0, 0),
    IdentityOp<false>);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Identity,
    kOnnxDomain,
    14, 18,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypes())
        .Alias(0, 0),
    IdentityOp<false>);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Identity,
    kOnnxDomain,
    19, 20,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypesIRv9())
        .Alias(0, 0),
    IdentityOp<false>);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Identity,
    kOnnxDomain,
    21, 22,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypesIRv9())
        .Alias(0, 0),
    IdentityOp<false>);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Identity,
    kOnnxDomain,
    23, 24,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypesIRv9())
        .Alias(0, 0),
    IdentityOp<false>);

ONNX_OPERATOR_KERNEL_EX(
    Identity,
    kOnnxDomain,
    25,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypesIRv9())
        .Alias(0, 0),
    IdentityOp<false>);

ONNX_OPERATOR_KERNEL_EX(
    StopGradient,
    kOnnxDomain,
    1,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .Alias(0, 0),
    IdentityOp<false>);

ONNX_OPERATOR_KERNEL_EX(
    Snapshot,
    kOnnxDomain,
    1,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .Alias(0, 0),
    IdentityOp<false>);

ONNX_OPERATOR_KERNEL_EX(
    IdentityN,
    kOnnxDomain,
    1,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    IdentityNOp);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Dropout,
    kOnnxDomain,
    7, 9,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, float, double>()),
    IdentityOp<true>);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Dropout,
    kOnnxDomain,
    10, 11,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, float, double>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>()),
    IdentityOp<true>);

template class IdentityOp<false>;
template class IdentityOp<true>;

}  // namespace musa
}  // namespace onnxruntime
