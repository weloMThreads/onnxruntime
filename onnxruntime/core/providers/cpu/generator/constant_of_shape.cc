// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/generator/constant_of_shape_base.h"
#include "core/providers/op_kernel_type_control.h"

#include <algorithm>

namespace onnxruntime {

namespace op_kernel_type_control {
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, ConstantOfShape, Output, 0,
    ConstantOfShapeDefaultOutputTypes);

ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST(
    kCpuExecutionProvider, kOnnxDomain, ConstantOfShape, 20, Output, 0,
    ConstantOfShapeDefaultOutputTypesOpset20);

ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST(
    kCpuExecutionProvider, kOnnxDomain, ConstantOfShape, 21, Output, 0,
    ConstantOfShapeDefaultOutputTypesOpset21);

ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST(
    kCpuExecutionProvider, kOnnxDomain, ConstantOfShape, 23, Output, 0,
    ConstantOfShapeDefaultOutputTypesOpset23);

ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST(
    kCpuExecutionProvider, kOnnxDomain, ConstantOfShape, 24, Output, 0,
    ConstantOfShapeDefaultOutputTypesOpset23);

ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST(
    kCpuExecutionProvider, kOnnxDomain, ConstantOfShape, 25, Output, 0,
    ConstantOfShapeDefaultOutputTypesOpset23);

// pytorch converter uses ConstantOfShape with int64 to create Pad input
// https://github.com/pytorch/pytorch/blob/044b519a80459f6787f6723c1c091a18b153d184/torch/onnx/symbolic_opset11.py#L449
ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, ConstantOfShape, Output, 0,
    int64_t);

}  // namespace op_kernel_type_control

namespace {

using EnabledOutputTypes =
    ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(
        kCpuExecutionProvider, kOnnxDomain, ConstantOfShape, Output, 0);

// ConstantOfShape usually updates the output type list, which is why
// we have a separate type list for it when the opset is updated.
using EnabledOutputTypesOpset20 =
    ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(
        kCpuExecutionProvider, kOnnxDomain, ConstantOfShape, 20, Output, 0);

using EnabledOutputTypesOpset21 =
    ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(
        kCpuExecutionProvider, kOnnxDomain, ConstantOfShape, 21, Output, 0);

using EnabledOutputTypesOpset23 =
    ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(
        kCpuExecutionProvider, kOnnxDomain, ConstantOfShape, 23, Output, 0);

using EnabledOutputTypesOpset24 =
    ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(
        kCpuExecutionProvider, kOnnxDomain, ConstantOfShape, 24, Output, 0);

using EnabledOutputTypesOpset25 =
    ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(
        kCpuExecutionProvider, kOnnxDomain, ConstantOfShape, 25, Output, 0);

class ConstantOfShape final : public ConstantOfShapeBase<EnabledOutputTypes>, public OpKernel {
 public:
  explicit ConstantOfShape(const OpKernelInfo& info) : ConstantOfShapeBase(info), OpKernel(info) {}

  Status Compute(OpKernelContext* ctx) const override;
};

template <class T>
inline void FilloutOutput(T value, void* output_data, size_t size) {
  std::fill_n(reinterpret_cast<T*>(output_data), size, value);
}

Status ConstantOfShape::Compute(OpKernelContext* ctx) const {
  Tensor* output_tensor = nullptr;
  ORT_RETURN_IF_ERROR(PrepareCompute(ctx, &output_tensor));

  auto output_data = output_tensor->MutableDataRaw();
  const void* value_ptr = GetValuePtr();
  const auto size = output_tensor->Shape().Size();
  const auto element_size = output_tensor->DataType()->Size();
  switch (element_size) {
    case sizeof(int8_t):
      FilloutOutput(*(reinterpret_cast<const int8_t*>(value_ptr)), output_data, onnxruntime::narrow<size_t>(size));
      break;
    case sizeof(int16_t):
      FilloutOutput(*(reinterpret_cast<const int16_t*>(value_ptr)), output_data, onnxruntime::narrow<size_t>(size));
      break;
    case sizeof(int32_t):
      FilloutOutput(*(reinterpret_cast<const int32_t*>(value_ptr)), output_data, onnxruntime::narrow<size_t>(size));
      break;
    case sizeof(int64_t):
      FilloutOutput(*(reinterpret_cast<const int64_t*>(value_ptr)), output_data, onnxruntime::narrow<size_t>(size));
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported output datatype with size: ", element_size);
  }

  return Status::OK();
}

template <typename T>
class Fill final : public OpKernel {
 public:
  explicit Fill(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* ctx) const override {
    const Tensor* dims = ctx->Input<Tensor>(0);
    const Tensor* value = ctx->Input<Tensor>(1);
    ORT_RETURN_IF_NOT(dims != nullptr && value != nullptr, "Fill inputs must not be null");
    ORT_RETURN_IF_NOT(dims->Shape().NumDimensions() <= 1,
                      "Fill dims must be a scalar or 1D tensor, got shape ", dims->Shape().ToString());
    ORT_RETURN_IF_NOT(value->Shape().Size() == 1,
                      "Fill value must be a scalar or single-element tensor, got shape ",
                      value->Shape().ToString());

    const int64_t rank = dims->Shape().Size();
    TensorShapeVector output_dims;
    output_dims.reserve(onnxruntime::narrow<size_t>(rank));
    if (dims->IsDataType<int32_t>()) {
      const auto* dims_data = dims->Data<int32_t>();
      for (int64_t i = 0; i < rank; ++i) {
        ORT_RETURN_IF_NOT(dims_data[i] >= 0, "Fill dims must be non-negative, got ", dims_data[i]);
        output_dims.push_back(static_cast<int64_t>(dims_data[i]));
      }
    } else if (dims->IsDataType<int64_t>()) {
      const auto* dims_data = dims->Data<int64_t>();
      for (int64_t i = 0; i < rank; ++i) {
        ORT_RETURN_IF_NOT(dims_data[i] >= 0, "Fill dims must be non-negative, got ", dims_data[i]);
        output_dims.push_back(dims_data[i]);
      }
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Fill dims must be int32 or int64");
    }

    Tensor* output = ctx->Output(0, TensorShape(output_dims));
    ORT_RETURN_IF_NOT(output != nullptr, "Fill failed to allocate output tensor");
    const int64_t output_size = output->Shape().Size();
    if (output_size == 0) {
      return Status::OK();
    }

    std::fill_n(output->MutableData<T>(), onnxruntime::narrow<size_t>(output_size), value->Data<T>()[0]);
    return Status::OK();
  }
};

}  // namespace

#define REGISTER_CPU_FILL_TYPED(T)                                              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      Fill, kOnnxDomain, 1, T, kCpuExecutionProvider,                           \
      KernelDefBuilder()                                                        \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())               \
          .TypeConstraint("index_type", BuildKernelDefConstraints<int32_t, int64_t>()), \
      Fill<T>);

REGISTER_CPU_FILL_TYPED(float)
REGISTER_CPU_FILL_TYPED(double)
REGISTER_CPU_FILL_TYPED(MLFloat16)
REGISTER_CPU_FILL_TYPED(int32_t)
REGISTER_CPU_FILL_TYPED(int64_t)
REGISTER_CPU_FILL_TYPED(bool)

#undef REGISTER_CPU_FILL_TYPED

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    ConstantOfShape,
    9,
    19,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T2",
                        BuildKernelDefConstraintsFromTypeList<EnabledOutputTypes>()),
    ConstantOfShape);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    ConstantOfShape,
    20,
    20,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T2",
                        BuildKernelDefConstraintsFromTypeList<EnabledOutputTypesOpset20>()),
    ConstantOfShape);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    ConstantOfShape,
    21,
    22,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T2",
                        BuildKernelDefConstraintsFromTypeList<EnabledOutputTypesOpset21>()),
    ConstantOfShape);

// Opset 23 added support for float4e2m1.
// TODOd: Add support for float4e2m1.
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    ConstantOfShape,
    23,
    23,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T2",
                        BuildKernelDefConstraintsFromTypeList<EnabledOutputTypesOpset23>()),
    ConstantOfShape);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    ConstantOfShape,
    24,
    24,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T2",
                        BuildKernelDefConstraintsFromTypeList<EnabledOutputTypesOpset24>()),
    ConstantOfShape);

ONNX_CPU_OPERATOR_KERNEL(
    ConstantOfShape,
    25,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T2",
                        BuildKernelDefConstraintsFromTypeList<EnabledOutputTypesOpset25>()),
    ConstantOfShape);
}  // namespace onnxruntime
