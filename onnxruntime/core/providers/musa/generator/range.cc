// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/common.h"
#include "range.h"
#include "core/providers/musa/musa_fwd.h"
#include <musa_runtime.h>
#include <mudnn.h>
#include <vector>
#include <cmath>

using onnxruntime::common::Status;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace musa {

template <typename T>
Status RangeImpl(musaStream_t stream, const T start, const T delta, const int count, T* output) {
  // Create temporary CPU buffer
  std::vector<T> cpu_buffer(count);
  
  // Fill CPU buffer with range values
  for (int i = 0; i < count; ++i) {
    cpu_buffer[i] = start + delta * T(i);
  }
  
  // Copy from CPU to GPU
  auto status = musaMemcpyAsync(output, cpu_buffer.data(), count * sizeof(T), 
                                musaMemcpyHostToDevice, stream);
  if (status != musaSuccess) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MUSA memcpy failed: ", musaGetErrorString(status));
  }
  
  return Status::OK();
}

template <typename T>
static Status ComputeRange(musaStream_t stream, OpKernelContext* ctx) {
  const auto& start_tensor = *ctx->Input<Tensor>(0);
  const auto& limit_tensor = *ctx->Input<Tensor>(1);
  const auto* delta_tensor_ptr = ctx->Input<Tensor>(2);

  if (!start_tensor.Shape().IsScalar()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "start in Range operator should be scalar like tensor, yet got shape:",
                           start_tensor.Shape());
  }
  if (!limit_tensor.Shape().IsScalar()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "limit in Range operator should be scalar like tensor, yet got shape:",
                           limit_tensor.Shape());
  }
  if (delta_tensor_ptr != nullptr && !delta_tensor_ptr->Shape().IsScalar()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "delta in Range operator should be scalar like tensor, yet got shape:",
                           delta_tensor_ptr->Shape());
  }

  // Start, Limit and Delta are stored in CPU.
  T start = *(start_tensor.Data<T>());
  T limit = *(limit_tensor.Data<T>());

  T delta = T(1);
  if (delta_tensor_ptr != nullptr) {
    delta = *(delta_tensor_ptr->Data<T>());
  }

  if (delta == T(0)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "delta in Range operator can not be zero!");
  }

  double num = (static_cast<double>(limit) - static_cast<double>(start)) / static_cast<double>(delta);
  int count = static_cast<int>(ceil(num));
  if (count <= 0)
    count = 0;
  TensorShape shape = {static_cast<int64_t>(count)};
  T* y = ctx->Output(0, shape)->MutableData<T>();

  if (count > 0) {
    ORT_RETURN_IF_ERROR(RangeImpl(stream, start, delta, count, y));
  }

  return Status::OK();
}

namespace musa_range_internal {

template <class T>
struct CallMusaRangeImpl {
  Status operator()(musaStream_t stream, OpKernelContext* ctx) const {
    return ComputeRange<T>(stream, ctx);
  }
};

}  // namespace musa_range_internal

Status Range::ComputeInternal(OpKernelContext* ctx) const {
  const auto* input_tensor = ctx->Input<Tensor>(0);
  if (input_tensor == nullptr) {
    return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  }

  utils::MLTypeCallDispatcher<int32_t, float, int64_t, int16_t>
      t_disp(input_tensor->GetElementType());
  return t_disp.InvokeRet<Status, musa_range_internal::CallMusaRangeImpl>(Stream(ctx), ctx);
}

// Register Range operator with different data types
#define REGISTER_MUSA_RANGE_TYPED_KERNEL(T, ver)                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                    \
      Range,                                                        \
      kOnnxDomain,                                                  \
      ver,                                                          \
      T,                                                            \
      kMusaExecutionProvider,                                       \
      (*KernelDefBuilder::Create())                                 \
          .InputMemoryType(OrtMemTypeCPUInput, 0)                   \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                   \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                   \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),   \
      Range);

REGISTER_MUSA_RANGE_TYPED_KERNEL(float, 11)
REGISTER_MUSA_RANGE_TYPED_KERNEL(int32_t, 11)
REGISTER_MUSA_RANGE_TYPED_KERNEL(int64_t, 11)
REGISTER_MUSA_RANGE_TYPED_KERNEL(int16_t, 11)

}  // namespace musa
}  // namespace onnxruntime