// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/tensor/gather.h"
#include "core/providers/musa/tensor/gather_impl.h"
#include "core/providers/common.h"
#include "core/providers/musa/musa_fwd.h"

#include <vector>

using onnxruntime::common::Status;
namespace onnxruntime {
namespace musa {

template <typename T>
Status Gather<T>::ComputeInternal(OpKernelContext* ctx) const {
  GatherBase::Prepare p;
  ORT_RETURN_IF_ERROR(PrepareForCompute(ctx, p));

  if (p.output_tensor->Shape().Size() == 0) {
    return Status::OK();
  }

  if (!p.indices_tensor->IsDataType<int32_t>() &&
      !p.indices_tensor->IsDataType<int64_t>()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Type for Tind not supported yet in Gather.");
  }

  const TensorShape& input_shape = p.input_tensor->Shape();
  const int64_t block_size = input_shape.SizeFromDimension(p.axis + 1);
  const int64_t input_block_size = input_shape.SizeFromDimension(p.axis);
  const int64_t output_block_size = p.indices_tensor->Shape().Size() * block_size;
  const int64_t indices_max = input_shape[p.axis];

  GatherImpl(
      Stream(ctx),
      input_block_size,
      indices_max,
      output_block_size,
      block_size,
      p.indices_tensor->DataRaw(),
      p.indices_tensor->DataType()->Size(),
      p.input_tensor->DataRaw(),
      p.input_tensor->DataType()->Size(),
      p.output_tensor->MutableDataRaw(),
      static_cast<size_t>(p.output_tensor->Shape().Size()));

  return Status::OK();
}

// Macro for registering typed compute function with MUSA implementation
#define REGISTER_MUSA_GATHER_TYPED_COMPUTE(T)                               \
  template Status Gather<T>::ComputeInternal(OpKernelContext* ctx) const;

// Macro for registering typed kernel
#define REGISTER_MUSA_GATHER_TYPED_KERNEL(ver, T)                           \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                               \
      Gather, kOnnxDomain, ver, T, kMusaExecutionProvider,                  \
      (*KernelDefBuilder::Create())                                            \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())              \
          .TypeConstraint("Tind", std::vector<MLDataType>{                    \
                                      DataTypeImpl::GetTensorType<int32_t>(), \
                                      DataTypeImpl::GetTensorType<int64_t>()}), \
      Gather<T>);

// Macro for registering versioned typed kernel
#define REGISTER_MUSA_GATHER_VERSIONED_TYPED_KERNEL(startver, endver, T)     \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                   \
      Gather, kOnnxDomain, startver, endver, T, kMusaExecutionProvider,      \
      (*KernelDefBuilder::Create())                                            \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())              \
          .TypeConstraint("Tind", std::vector<MLDataType>{                    \
                                      DataTypeImpl::GetTensorType<int32_t>(), \
                                      DataTypeImpl::GetTensorType<int64_t>()}), \
      Gather<T>);

// Combined macro for both kernel and compute registration
#define REGISTER_MUSA_GATHER_TYPED(ver, T)                                  \
  REGISTER_MUSA_GATHER_TYPED_KERNEL(ver, T)                                 \
  REGISTER_MUSA_GATHER_TYPED_COMPUTE(T)

// Macro for versioned typed registration
#define REGISTER_MUSA_GATHER_VERSIONED_TYPED(startver, endver, T)           \
  REGISTER_MUSA_GATHER_VERSIONED_TYPED_KERNEL(startver, endver, T)

// Register Gather operations for supported data types and versions

// Gather operations (version 1-10)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(1, 10, uint8_t)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(1, 10, uint16_t)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(1, 10, uint32_t)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(1, 10, uint64_t)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(1, 10, int8_t)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(1, 10, int16_t)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(1, 10, int32_t)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(1, 10, int64_t)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(1, 10, MLFloat16)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(1, 10, float)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(1, 10, double)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(1, 10, bool)

// Gather operations (version 11-12)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(11, 12, uint8_t)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(11, 12, uint16_t)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(11, 12, uint32_t)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(11, 12, uint64_t)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(11, 12, int8_t)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(11, 12, int16_t)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(11, 12, int32_t)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(11, 12, int64_t)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(11, 12, MLFloat16)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(11, 12, float)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(11, 12, double)
REGISTER_MUSA_GATHER_VERSIONED_TYPED(11, 12, bool)

// Gather operations (version 13+)
REGISTER_MUSA_GATHER_TYPED(13, uint8_t)
REGISTER_MUSA_GATHER_TYPED(13, uint16_t)
REGISTER_MUSA_GATHER_TYPED(13, uint32_t)
REGISTER_MUSA_GATHER_TYPED(13, uint64_t)
REGISTER_MUSA_GATHER_TYPED(13, int8_t)
REGISTER_MUSA_GATHER_TYPED(13, int16_t)
REGISTER_MUSA_GATHER_TYPED(13, int32_t)
REGISTER_MUSA_GATHER_TYPED(13, int64_t)
REGISTER_MUSA_GATHER_TYPED(13, MLFloat16)
REGISTER_MUSA_GATHER_TYPED(13, float)
REGISTER_MUSA_GATHER_TYPED(13, double)
REGISTER_MUSA_GATHER_TYPED(13, bool)

} // namespace musa
} // namespace onnxruntime
