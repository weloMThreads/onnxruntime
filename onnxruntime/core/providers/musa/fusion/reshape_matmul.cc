// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/fusion/reshape_matmul.h"
#include "core/providers/musa/fusion/reshape_matmul_impl.h"
#include "core/providers/musa/musa_execution_provider.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/musa_utils.h"

#include <mudnn.h>

#include <string>
#include <type_traits>
#include <vector>

namespace onnxruntime {
namespace musa {
namespace {

void NoopMemoryDeleter(void*) {}

Status SetupRawMusaTensor(::musa::dnn::Tensor& tensor,
                          const void* data,
                          const std::vector<int64_t>& dims,
                          ::musa::dnn::Tensor::Type type) {
  auto status = tensor.SetType(type);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "MusaReshapeMatMul SetType failed, status=", static_cast<int>(status));
  }

  status = tensor.SetAddr(data);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "MusaReshapeMatMul SetAddr failed, status=", static_cast<int>(status));
  }

  status = tensor.SetFormat(::musa::dnn::Tensor::Format::NCHW);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "MusaReshapeMatMul SetFormat failed, status=", static_cast<int>(status));
  }

  status = tensor.SetNdInfo(static_cast<int>(dims.size()), dims.data());
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "MusaReshapeMatMul SetNdInfo failed, status=", static_cast<int>(status));
  }

  return Status::OK();
}

template <typename T>
Status RunReshapeMatMulTyped(const MusaReshapeMatMul* kernel,
                             OpKernelContext* ctx,
                             const Tensor* input,
                             const Tensor* weight,
                             Tensor* output,
                             bool transpose_b) {
  const auto* ep = static_cast<const MusaExecutionProvider*>(kernel->Info().GetExecutionProvider());
  MusaPreparation prepare(ep);
  auto* stream = kernel->Stream(ctx);
  if (prepare.handle && stream) {
    auto status = prepare.handle->SetStream(stream);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "MusaReshapeMatMul failed to set stream, status=", static_cast<int>(status));
    }
  }

  const int64_t k = input->Shape()[input->Shape().NumDimensions() - 1];
  const int64_t n = transpose_b ? weight->Shape()[0] : weight->Shape()[1];
  const int64_t m = input->Shape().Size() / k;

  const auto type = GetMusaDataType<T>();
  ::musa::dnn::Tensor input_tensor;
  ::musa::dnn::Tensor weight_tensor;
  ::musa::dnn::Tensor output_tensor;
  const std::vector<int64_t> input_dims{m, k};
  const std::vector<int64_t> weight_dims(weight->Shape().GetDims().begin(), weight->Shape().GetDims().end());
  const std::vector<int64_t> output_dims{m, n};
  ORT_RETURN_IF_ERROR(SetupRawMusaTensor(input_tensor, input->DataRaw(), input_dims, type));
  ORT_RETURN_IF_ERROR(SetupRawMusaTensor(weight_tensor, weight->DataRaw(), weight_dims, type));
  ORT_RETURN_IF_ERROR(SetupRawMusaTensor(output_tensor, output->MutableDataRaw(), output_dims, type));

  ::musa::dnn::MatMul matmul_op;
  auto status = matmul_op.SetAlpha(1.0);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MusaReshapeMatMul SetAlpha failed");
  }
  status = matmul_op.SetBeta(0.0);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MusaReshapeMatMul SetBeta failed");
  }
  status = matmul_op.SetTranspose(false, transpose_b);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MusaReshapeMatMul SetTranspose failed");
  }

  if constexpr (std::is_same_v<T, float>) {
    if (ep != nullptr && ep->UseTF32()) {
      status = matmul_op.SetComputeMode(::musa::dnn::MatMul::ComputeMode::TENSOR);
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MusaReshapeMatMul SetComputeMode failed");
      }
    }
  }

  std::vector<IAllocatorUniquePtr<void>> workspace_buffers;
  auto memory_allocator = [kernel, ctx, &workspace_buffers](size_t size) -> ::musa::dnn::MemoryHandler {
    if (size == 0) {
      return ::musa::dnn::MemoryHandler(nullptr, NoopMemoryDeleter);
    }
    auto scratch = kernel->GetScratchBuffer<void>(size, ctx->GetComputeStream());
    void* ptr = scratch.get();
    workspace_buffers.push_back(std::move(scratch));
    return ::musa::dnn::MemoryHandler(ptr, NoopMemoryDeleter);
  };

  status = matmul_op.Run(prepare.GetHandle(), output_tensor, input_tensor, weight_tensor, memory_allocator);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "MusaReshapeMatMul muDNN MatMul failed, status=", static_cast<int>(status));
  }

  return Status::OK();
}

}  // namespace

Status MusaReshapeMatMul::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* input = ctx->Input<Tensor>(0);
  const Tensor* weight = ctx->Input<Tensor>(1);
  ORT_RETURN_IF_NOT(input != nullptr && weight != nullptr, "MusaReshapeMatMul inputs must not be null");
  ORT_RETURN_IF_NOT(input->Shape().NumDimensions() >= 2,
                    "MusaReshapeMatMul requires input rank >= 2, got ", input->Shape().ToString());
  ORT_RETURN_IF_NOT(weight->Shape().NumDimensions() == 2,
                    "MusaReshapeMatMul requires 2D weight, got ", weight->Shape().ToString());

  if (transpose_input_021_) {
    ORT_RETURN_IF_NOT(input->IsDataType<float>() && weight->IsDataType<float>(),
                      "MusaReshapeMatMul transpose_input_021 supports float only");
    ORT_RETURN_IF_NOT(input->Shape().NumDimensions() == 3,
                      "MusaReshapeMatMul transpose_input_021 requires rank-3 input, got ",
                      input->Shape().ToString());

    const int64_t batch = input->Shape()[0];
    const int64_t k = input->Shape()[1];
    const int64_t tokens = input->Shape()[2];
    const int64_t weight_k = transpose_b_ ? weight->Shape()[1] : weight->Shape()[0];
    const int64_t n = transpose_b_ ? weight->Shape()[0] : weight->Shape()[1];
    ORT_RETURN_IF_NOT(k == weight_k,
                      "MusaReshapeMatMul transpose_input_021 K mismatch: input dim1=", k,
                      ", weight K=", weight_k);

    Tensor* output = ctx->Output(0, TensorShape({batch, tokens, n}));
    ORT_RETURN_IF_NOT(output != nullptr, "MusaReshapeMatMul output is null");
    if (output->Shape().Size() == 0) {
      return Status::OK();
    }

    const musaError_t status = LaunchTranspose021MatMulFloat(Stream(ctx),
                                                             input->Data<float>(),
                                                             weight->Data<float>(),
                                                             output->MutableData<float>(),
                                                             batch,
                                                             k,
                                                             tokens,
                                                             n,
                                                             transpose_b_);
    if (status != musaSuccess) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "MusaReshapeMatMul transpose_input_021 kernel launch failed, status=",
                             static_cast<int>(status));
    }
    return Status::OK();
  }

  const int64_t k = input->Shape()[input->Shape().NumDimensions() - 1];
  const int64_t weight_k = transpose_b_ ? weight->Shape()[1] : weight->Shape()[0];
  const int64_t n = transpose_b_ ? weight->Shape()[0] : weight->Shape()[1];
  ORT_RETURN_IF_NOT(k == weight_k,
                    "MusaReshapeMatMul K mismatch: input last dim=", k,
                    ", weight K=", weight_k);

  TensorShapeVector output_dims = input->Shape().AsShapeVector();
  output_dims.back() = n;
  Tensor* output = ctx->Output(0, TensorShape(output_dims));
  ORT_RETURN_IF_NOT(output != nullptr, "MusaReshapeMatMul output is null");
  if (output->Shape().Size() == 0) {
    return Status::OK();
  }

  if (input->IsDataType<float>()) {
    return RunReshapeMatMulTyped<float>(this, ctx, input, weight, output, transpose_b_);
  }
  if (input->IsDataType<MLFloat16>()) {
    return RunReshapeMatMulTyped<MLFloat16>(this, ctx, input, weight, output, transpose_b_);
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MusaReshapeMatMul supports float and float16 only");
}

ONNX_OPERATOR_KERNEL_EX(
    MusaReshapeMatMul,
    kMSDomain,
    1,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create()),
    MusaReshapeMatMul);

}  // namespace musa
}  // namespace onnxruntime
