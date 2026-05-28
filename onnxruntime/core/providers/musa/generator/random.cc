// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/generator/random.h"

#include <gsl/gsl>
#include <musa_runtime.h>

#include "core/providers/musa/musa_call.h"
#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {
namespace musa {
namespace {

ONNX_NAMESPACE::TensorProto::DataType InferDataType(const Tensor& tensor) {
  const auto elem_type = tensor.GetElementType();
  if (elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
      elem_type == ONNX_NAMESPACE::TensorProto_DataType_DOUBLE ||
      elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    return static_cast<ONNX_NAMESPACE::TensorProto::DataType>(elem_type);
  }
  return ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
}

template <typename T>
void GenerateUniformData(std::default_random_engine& generator, float low, float high, std::vector<T>& output) {
  std::uniform_real_distribution<float> distribution{low, high};
  for (auto& value : output) {
    value = static_cast<T>(distribution(generator));
  }
}

template <>
void GenerateUniformData<double>(std::default_random_engine& generator, float low, float high, std::vector<double>& output) {
  std::uniform_real_distribution<double> distribution{static_cast<double>(low), static_cast<double>(high)};
  for (auto& value : output) {
    value = distribution(generator);
  }
}

template <>
void GenerateUniformData<MLFloat16>(std::default_random_engine& generator, float low, float high, std::vector<MLFloat16>& output) {
  std::uniform_real_distribution<float> distribution{low, high};
  for (auto& value : output) {
    value = MLFloat16(distribution(generator));
  }
}

template <typename T>
Status GenerateAndCopy(OpKernelContext* ctx,
                       musaStream_t stream,
                       std::default_random_engine& generator,
                       float low,
                       float high,
                       Tensor& output) {
  const int64_t count = output.Shape().Size();
  if (count == 0) return Status::OK();

  std::vector<T> host_output(static_cast<size_t>(count));
  GenerateUniformData<T>(generator, low, high, host_output);

  MUSA_RETURN_IF_ERROR(musaMemcpyAsync(output.MutableData<T>(),
                                       host_output.data(),
                                       host_output.size() * sizeof(T),
                                       musaMemcpyHostToDevice,
                                       stream));
  MUSA_RETURN_IF_ERROR(musaStreamSynchronize(stream));
  (void)ctx;
  return Status::OK();
}

}  // namespace

RandomUniformLike::RandomUniformLike(const OpKernelInfo& info) : MusaKernel(info) {
  ORT_ENFORCE(info.GetAttr<float>("high", &high_).IsOK());
  ORT_ENFORCE(info.GetAttr<float>("low", &low_).IsOK());

  float seed = 0.f;
  if (info.GetAttr<float>("seed", &seed).IsOK()) {
    generator_ = std::default_random_engine{gsl::narrow_cast<uint32_t>(seed)};
  } else {
    std::random_device random_device;
    generator_ = std::default_random_engine{random_device() + gsl::narrow_cast<uint32_t>(info.node().Index())};
  }

  int64_t dtype = ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
  if (info.GetAttr<int64_t>("dtype", &dtype).IsOK()) {
    ORT_ENFORCE(ONNX_NAMESPACE::TensorProto::DataType_IsValid(gsl::narrow<int>(dtype)) &&
                    dtype != ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED,
                "Invalid dtype of ", dtype);
    dtype_ = static_cast<ONNX_NAMESPACE::TensorProto::DataType>(dtype);
  }
}

Status RandomUniformLike::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* input = ctx->Input<Tensor>(0);
  ORT_RETURN_IF_NOT(input != nullptr, "RandomUniformLike: input tensor is null");

  Tensor* output = ctx->Output(0, input->Shape());
  ORT_RETURN_IF_NOT(output != nullptr, "RandomUniformLike: output tensor is null");

  const auto dtype = dtype_ != ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED ? dtype_ : InferDataType(*input);
  ORT_RETURN_IF_NOT(dtype != ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED,
                    "RandomUniformLike: could not infer supported output dtype from input tensor");

  std::lock_guard<std::mutex> lock(generator_mutex_);
  switch (dtype) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return GenerateAndCopy<float>(ctx, Stream(ctx), generator_, low_, high_, *output);
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      return GenerateAndCopy<double>(ctx, Stream(ctx), generator_, low_, high_, *output);
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return GenerateAndCopy<MLFloat16>(ctx, Stream(ctx), generator_, low_, high_, *output);
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "RandomUniformLike output type not supported by MUSA EP: ", dtype);
  }
}

ONNX_OPERATOR_KERNEL_EX(
    RandomUniformLike,
    kOnnxDomain,
    1,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("T2", BuildKernelDefConstraints<MLFloat16, float, double>()),
    RandomUniformLike);

}  // namespace musa
}  // namespace onnxruntime
