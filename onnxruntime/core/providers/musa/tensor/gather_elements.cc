// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/tensor/gather_elements.h"
#include "core/providers/musa/tensor/gather_elements_impl.h"
#include "core/providers/common.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/cpu/tensor/utils.h"

#include <type_traits>
#include <vector>

using onnxruntime::common::Status;
namespace onnxruntime {
namespace musa {

namespace {

template <typename T, int32_t capacity = 8>
void CopyToTArray(TArray<T, capacity>& dst, const TensorShapeVector& src) {
  ORT_ENFORCE(src.size() <= static_cast<size_t>(dst.Capacity()),
              "GatherElements rank ", src.size(), " exceeds kernel capacity ", dst.Capacity());
  dst.SetSize(static_cast<int32_t>(src.size()));
  for (int32_t i = 0; i < dst.Size(); ++i) {
    dst[i] = src[static_cast<size_t>(i)];
  }
}

template <typename TIndex>
Status ValidateIndicesOnCpu(const Tensor* indices_tensor, const TensorShape& input_shape, int64_t axis) {
  const TIndex* indices_data = indices_tensor->Data<TIndex>();
  const int64_t indices_count = indices_tensor->Shape().Size();
  const int64_t axis_dim_limit = input_shape[static_cast<size_t>(axis)];

  for (int64_t i = 0; i < indices_count; ++i) {
    const TIndex idx = indices_data[i];
    if (idx < -axis_dim_limit || idx >= axis_dim_limit) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "GatherElements: indices element out of data bounds, idx=", idx,
                             " must be within the inclusive range [", -axis_dim_limit, ",", axis_dim_limit - 1, "]");
    }
  }

  return Status::OK();
}

void CoalesceDimensions(TensorShapeVector& input_shape, TensorShapeVector& indices_shape,
                        TensorShapeVector* p_indices_strides, int64_t axis, GatherScatterElementsArgs& args) {
  size_t rank = input_shape.size();
  if (axis < 0 || axis >= static_cast<int64_t>(rank)) {
    ORT_THROW("Invalid axis in CoalesceDimensions: ", axis);
  }

  size_t new_axis = static_cast<size_t>(axis);
  auto CanCoalesce = [&](size_t dst, size_t src) {
    if (dst == new_axis || src == new_axis) {
      return false;
    }
    if (input_shape[dst] == 1 && indices_shape[dst] == 1) {
      return true;
    }
    if (input_shape[src] == 1 && indices_shape[src] == 1) {
      return true;
    }
    return input_shape[dst] == indices_shape[dst] && input_shape[src] == indices_shape[src] &&
           (!p_indices_strides || (*p_indices_strides)[dst] == indices_shape[src] * (*p_indices_strides)[src]);
  };

  size_t curr = 0;
  for (size_t next = 1; next < rank; ++next) {
    if (CanCoalesce(curr, next)) {
      if (indices_shape[next] != 1 && p_indices_strides) {
        (*p_indices_strides)[curr] = (*p_indices_strides)[next];
      }
      input_shape[curr] *= input_shape[next];
      indices_shape[curr] *= indices_shape[next];
    } else {
      if (next == new_axis) {
        if (input_shape[curr] != 1 || indices_shape[curr] != 1) {
          ++curr;
        }
        new_axis = curr;
      } else {
        ++curr;
      }
      if (curr != next) {
        input_shape[curr] = input_shape[next];
        indices_shape[curr] = indices_shape[next];
        if (p_indices_strides) {
          (*p_indices_strides)[curr] = (*p_indices_strides)[next];
        }
      }
    }
  }

  if (curr > new_axis && input_shape[curr] == 1 && indices_shape[curr] == 1) {
    --curr;
  }

  size_t new_rank = curr + 1;
  args.rank = static_cast<int64_t>(new_rank);
  args.axis = static_cast<int64_t>(new_axis);
  input_shape.resize(new_rank);
  indices_shape.resize(new_rank);
  if (p_indices_strides) {
    p_indices_strides->resize(new_rank);
  }

  TensorPitches masked_input_strides_vec(input_shape);
  args.input_stride_along_axis = masked_input_strides_vec[args.axis];
  args.input_dim_along_axis = input_shape[args.axis];
  masked_input_strides_vec[args.axis] = 0;
  CopyToTArray(args.masked_input_strides, masked_input_strides_vec);

  TensorPitches indices_shape_strides(indices_shape);
  args.indices_fdms.SetSize(static_cast<int32_t>(new_rank));
  for (int32_t i = 0; i < args.indices_fdms.Size(); ++i) {
    args.indices_fdms[i] = fast_divmod(gsl::narrow_cast<int>(indices_shape_strides[static_cast<size_t>(i)]));
  }

  if (p_indices_strides) {
    CopyToTArray(args.indices_strides, *p_indices_strides);
  }
}

ONNX_NAMESPACE::TensorProto_DataType GetElementType(size_t element_size) {
  switch (element_size) {
    case sizeof(int8_t):
      return ONNX_NAMESPACE::TensorProto_DataType_INT8;
    case sizeof(MLFloat16):
      return ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
    case sizeof(float):
      return ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
    case sizeof(double):
      return ONNX_NAMESPACE::TensorProto_DataType_DOUBLE;
    default:
      return ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
  }
}

template <typename T>
Status DispatchGatherElementsImpl(musaStream_t stream, const void* input_data_raw, const void* indices_data_raw,
                                  void* output_data_raw, size_t index_element_size,
                                  const GatherScatterElementsArgs& args) {
  const T* input_data = reinterpret_cast<const T*>(input_data_raw);
  T* output_data = reinterpret_cast<T*>(output_data_raw);
  switch (index_element_size) {
    case sizeof(int32_t):
      GatherElementsImpl<T, int32_t>(stream, input_data, reinterpret_cast<const int32_t*>(indices_data_raw), output_data, args);
      return Status::OK();
    case sizeof(int64_t):
      GatherElementsImpl<T, int64_t>(stream, input_data, reinterpret_cast<const int64_t*>(indices_data_raw), output_data, args);
      return Status::OK();
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Unsupported indices element size by the GatherElements MUSA kernel");
  }
}

}  // namespace

// Ideally both input and indices can support strided tensor. Match CUDA and mark indices only.
#ifdef ENABLE_STRIDED_TENSORS
#define CREATE_GATHER_ELEMENTS_KERNEL_DEF (*KernelDefBuilder::Create()).MayStridedInput(1)
#else
#define CREATE_GATHER_ELEMENTS_KERNEL_DEF (*KernelDefBuilder::Create())
#endif

// Keep existing registration coverage so Layer 1/2/3 stay aligned with musa_execution_provider.cc.
#define REGISTER_MUSA_GATHER_ELEMENTS_TYPED_KERNEL(ver, T)                                                   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(GatherElements, kOnnxDomain, ver, T, kMusaExecutionProvider,                \
                                CREATE_GATHER_ELEMENTS_KERNEL_DEF                                            \
                                    .InputMemoryType(OrtMemTypeCPUInput, 1)                                  \
                                    .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                   \
                                    .TypeConstraint("Tind",                                                  \
                                                    std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(), \
                                                                            DataTypeImpl::GetTensorType<int64_t>()}), \
                                GatherElements<T>);

REGISTER_MUSA_GATHER_ELEMENTS_TYPED_KERNEL(11, uint8_t)
REGISTER_MUSA_GATHER_ELEMENTS_TYPED_KERNEL(11, uint16_t)
REGISTER_MUSA_GATHER_ELEMENTS_TYPED_KERNEL(11, uint32_t)
REGISTER_MUSA_GATHER_ELEMENTS_TYPED_KERNEL(11, uint64_t)
REGISTER_MUSA_GATHER_ELEMENTS_TYPED_KERNEL(11, int8_t)
REGISTER_MUSA_GATHER_ELEMENTS_TYPED_KERNEL(11, int16_t)
REGISTER_MUSA_GATHER_ELEMENTS_TYPED_KERNEL(11, int32_t)
REGISTER_MUSA_GATHER_ELEMENTS_TYPED_KERNEL(11, int64_t)
REGISTER_MUSA_GATHER_ELEMENTS_TYPED_KERNEL(11, MLFloat16)
REGISTER_MUSA_GATHER_ELEMENTS_TYPED_KERNEL(11, float)

#undef CREATE_GATHER_ELEMENTS_KERNEL_DEF
#undef REGISTER_MUSA_GATHER_ELEMENTS_TYPED_KERNEL

template <typename T>
Status GatherElements<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* input_tensor = ctx->Input<Tensor>(0);
  const Tensor* indices_tensor = ctx->Input<Tensor>(1);
  const TensorShape& input_shape = input_tensor->Shape();
  const TensorShape& indices_shape = indices_tensor->Shape();
  const auto input_rank = input_shape.NumDimensions();
  const auto axis = HandleNegativeAxis(axis_, input_rank);

  ORT_RETURN_IF_ERROR(onnxruntime::GatherElements::ValidateInputShapes(input_shape, indices_shape, axis));

  Tensor* output_tensor = ctx->Output(0, indices_shape);
  if (indices_shape.Size() == 0) {
    return Status::OK();
  }

  if (indices_tensor->IsDataType<int32_t>()) {
    ORT_RETURN_IF_ERROR(ValidateIndicesOnCpu<int32_t>(indices_tensor, input_shape, axis));
  } else if (indices_tensor->IsDataType<int64_t>()) {
    ORT_RETURN_IF_ERROR(ValidateIndicesOnCpu<int64_t>(indices_tensor, input_shape, axis));
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Unsupported indices data type for GatherElements");
  }

  GatherScatterElementsArgs args;
  args.input_size = input_shape.Size();
  args.indices_size = indices_shape.Size();

  TensorShapeVector input_shape_vec = input_shape.AsShapeVector();
  TensorShapeVector indices_shape_vec = indices_shape.AsShapeVector();
  TensorShapeVector* p_indices_strides_vec = nullptr;
#ifdef ENABLE_STRIDED_TENSORS
  TensorShapeVector indices_strides_vec;
  if (!indices_tensor->IsContiguous()) {
    indices_strides_vec = ToShapeVector(indices_tensor->Strides());
    p_indices_strides_vec = &indices_strides_vec;
  }
#endif
  CoalesceDimensions(input_shape_vec, indices_shape_vec, p_indices_strides_vec, axis, args);

  const size_t indices_bytes = static_cast<size_t>(indices_shape.Size()) * indices_tensor->DataType()->Size();
  auto indices_buffer = GetScratchBuffer<unsigned char>(indices_bytes, ctx->GetComputeStream());
  MUSA_RETURN_IF_ERROR(musaMemcpyAsync(indices_buffer.get(), indices_tensor->DataRaw(), indices_bytes,
                                       musaMemcpyHostToDevice, Stream(ctx)));
  const void* indices_data = indices_buffer.get();

  switch (GetElementType(input_tensor->DataType()->Size())) {
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      return DispatchGatherElementsImpl<int8_t>(Stream(ctx), input_tensor->DataRaw(), indices_data,
                                                output_tensor->MutableDataRaw(), indices_tensor->DataType()->Size(), args);
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      if constexpr (std::is_same_v<T, MLFloat16>) {
        GatherElementsImplHalf(Stream(ctx), input_tensor->DataRaw(), indices_data,
                               output_tensor->MutableDataRaw(), indices_tensor->DataType()->Size(), args);
        return Status::OK();
      }
      return DispatchGatherElementsImpl<int16_t>(Stream(ctx), input_tensor->DataRaw(), indices_data,
                                                 output_tensor->MutableDataRaw(), indices_tensor->DataType()->Size(), args);
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return DispatchGatherElementsImpl<float>(Stream(ctx), input_tensor->DataRaw(), indices_data,
                                               output_tensor->MutableDataRaw(), indices_tensor->DataType()->Size(), args);
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      return DispatchGatherElementsImpl<double>(Stream(ctx), input_tensor->DataRaw(), indices_data,
                                                output_tensor->MutableDataRaw(), indices_tensor->DataType()->Size(), args);
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Unsupported element size by the GatherElements MUSA kernel");
  }
}

#define REGISTER_MUSA_GATHER_ELEMENTS_TYPED_COMPUTE(T) \
  template Status GatherElements<T>::ComputeInternal(OpKernelContext* ctx) const;

REGISTER_MUSA_GATHER_ELEMENTS_TYPED_COMPUTE(uint8_t)
REGISTER_MUSA_GATHER_ELEMENTS_TYPED_COMPUTE(uint16_t)
REGISTER_MUSA_GATHER_ELEMENTS_TYPED_COMPUTE(uint32_t)
REGISTER_MUSA_GATHER_ELEMENTS_TYPED_COMPUTE(uint64_t)
REGISTER_MUSA_GATHER_ELEMENTS_TYPED_COMPUTE(int8_t)
REGISTER_MUSA_GATHER_ELEMENTS_TYPED_COMPUTE(int16_t)
REGISTER_MUSA_GATHER_ELEMENTS_TYPED_COMPUTE(int32_t)
REGISTER_MUSA_GATHER_ELEMENTS_TYPED_COMPUTE(int64_t)
REGISTER_MUSA_GATHER_ELEMENTS_TYPED_COMPUTE(MLFloat16)
REGISTER_MUSA_GATHER_ELEMENTS_TYPED_COMPUTE(float)

}  // namespace musa
}  // namespace onnxruntime
