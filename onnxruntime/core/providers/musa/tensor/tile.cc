// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/tensor/tile.h"
#include "core/providers/musa/tensor/tile_impl.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/musa_kernel.h"
#include "core/providers/musa/musa_utils.h"
#include "core/providers/musa/musa_execution_provider.h"
#include "core/providers/common.h"
#include <musa_runtime.h>
#include <mudnn.h>

using namespace onnxruntime::common;
namespace onnxruntime {
namespace musa {

namespace {
bool IsTileMemcpy(const TensorShape& input_shape,
                  const int64_t* repeats,
                  size_t rank,
                  bool& is_batched_memcpy,
                  size_t& num_of_elements_per_batch,
                  size_t& num_of_copies_per_batch,
                  size_t& num_of_batch_copies) {
  for (int64_t i = static_cast<int64_t>(rank) - 1; i >= 0; --i) {
    if (repeats[i] != 1) {
      if (input_shape.SizeToDimension(static_cast<size_t>(i)) == 1) {
        num_of_copies_per_batch = 1;
        for (int64_t j = 0; j <= i; ++j) {
          num_of_copies_per_batch *= static_cast<size_t>(repeats[static_cast<size_t>(j)]);
        }
        is_batched_memcpy = false;
        return true;
      } else if (i == 1) {
        num_of_elements_per_batch = static_cast<size_t>(input_shape.SizeFromDimension(1));
        num_of_copies_per_batch = static_cast<size_t>(repeats[static_cast<size_t>(i)]);
        num_of_batch_copies = static_cast<size_t>(repeats[0]);
        is_batched_memcpy = true;
        return true;
      } else {
        break;
      }
    }
  }
  return false;
}
}  // namespace

#define CASE_TILE_MEMCPY(T)                                 \
  TileMemcpyImpl<T>(stream,                                 \
                    reinterpret_cast<const T*>(input_data), \
                    reinterpret_cast<T*>(output_data),      \
                    input_shape.Size(),                     \
                    num_of_copies_per_batch)

#define CASE_TILE_BATCHED_MEMCPY(T)                                \
  TileBatchedMemcpyImpl<T>(stream,                                 \
                           reinterpret_cast<const T*>(input_data), \
                           reinterpret_cast<T*>(output_data),      \
                           num_of_elements_per_batch,              \
                           input_shape.Size(),                     \
                           num_of_batch_copies,                    \
                           num_of_copies_per_batch)

template <typename T>
Status Tile<T>::ComputeInternal(OpKernelContext* ctx) const {
  const auto* input_tensor = ctx->Input<Tensor>(0);
  const auto* repeats_tensor = ctx->Input<Tensor>(1);

  if (!input_tensor || !repeats_tensor) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input tensor is null");
  }

  const auto& input_shape = input_tensor->Shape();
  int32_t rank = static_cast<int32_t>(input_shape.NumDimensions());

  if (repeats_tensor->Shape().NumDimensions() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "'repeat' input tensor must be 1 dimensional");
  }
  if (repeats_tensor->Shape().Size() != rank) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "'repeat' input tensor must have the same length as the 'input' tensor");
  }

  const auto* repeats = repeats_tensor->Data<int64_t>();
  auto output_dims = input_shape.AsShapeVector();

  for (int32_t axis = 0; axis < rank; ++axis) {
    if (repeats[axis] < 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Repeat values must be non-negative");
    }
    output_dims[axis] *= repeats[axis];
  }

  TensorShape output_shape(output_dims);
  Tensor* output_tensor = ctx->Output(0, output_shape);

  if (output_shape.Size() == 0) {
    return Status::OK();
  }

  const void* input_data = input_tensor->DataRaw();
  void* output_data = output_tensor->MutableDataRaw();
  auto* stream = Stream(ctx);

  if (!stream) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to get MUSA stream");
  }

  if (output_shape == input_shape) {
    auto status = musaMemcpyAsync(output_data, input_data,
                                  input_tensor->SizeInBytes(),
                                  musaMemcpyDeviceToDevice, stream);
    if (status != musaSuccess) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "musaMemcpyAsync failed");
    }
    return Status::OK();
  }

  bool is_batched_memcpy = false;
  size_t num_of_elements_per_batch = 1;
  size_t num_of_copies_per_batch = 1;
  size_t num_of_batch_copies = 1;

  if (IsTileMemcpy(input_shape, repeats, static_cast<size_t>(rank),
                   is_batched_memcpy, num_of_elements_per_batch,
                   num_of_copies_per_batch, num_of_batch_copies)) {
    if (!is_batched_memcpy) {
      if constexpr (std::is_same_v<T, float>) {
        CASE_TILE_MEMCPY(float);
      } else if constexpr (std::is_same_v<T, double>) {
        CASE_TILE_MEMCPY(double);
      } else if constexpr (std::is_same_v<T, MLFloat16>) {
        TileMemcpyImpl<uint16_t>(stream,
                                 reinterpret_cast<const uint16_t*>(input_data),
                                 reinterpret_cast<uint16_t*>(output_data),
                                 input_shape.Size(),
                                 num_of_copies_per_batch);
      } else {
        goto generic_kernel;
      }
    } else {
      if constexpr (std::is_same_v<T, float>) {
        CASE_TILE_BATCHED_MEMCPY(float);
      } else if constexpr (std::is_same_v<T, double>) {
        CASE_TILE_BATCHED_MEMCPY(double);
      } else if constexpr (std::is_same_v<T, MLFloat16>) {
        TileBatchedMemcpyImpl<uint16_t>(stream,
                                        reinterpret_cast<const uint16_t*>(input_data),
                                        reinterpret_cast<uint16_t*>(output_data),
                                        num_of_elements_per_batch,
                                        input_shape.Size(),
                                        num_of_batch_copies,
                                        num_of_copies_per_batch);
      } else {
        goto generic_kernel;
      }
    }
    return Status::OK();
  }

generic_kernel:
  if (rank > kMaxTileDimensions) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Tile rank exceeds maximum supported dimensions (",
                           rank, " > ", kMaxTileDimensions, ")");
  }

  TileKernelParams params;
  params.rank = rank;
  params.total_elements = output_shape.Size();

  const auto& input_dims = input_shape.GetDims();

  if (rank > 0) {
    params.input_strides[rank - 1] = 1;
    params.output_strides[rank - 1] = 1;
    params.input_dims[rank - 1] = input_dims[rank - 1];

    for (int i = rank - 2; i >= 0; i--) {
      params.input_strides[i] = params.input_strides[i + 1] * input_dims[i + 1];
      params.output_strides[i] = params.output_strides[i + 1] * output_dims[i + 1];
      params.input_dims[i] = input_dims[i];
    }
  }

  LaunchTileKernel<T>(reinterpret_cast<const T*>(input_data),
                      reinterpret_cast<T*>(output_data), params, stream);

  return Status::OK();
}

#define REGISTER_MUSA_TILE_TYPED_KERNEL(ver, T)                          \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                         \
      Tile, kOnnxDomain, ver, T, kMusaExecutionProvider,                 \
      (*KernelDefBuilder::Create())                                      \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                        \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())         \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()), \
      Tile<T>);

#define REGISTER_MUSA_TILE_VERSIONED_TYPED_KERNEL(startver, endver, T)   \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                               \
      Tile, kOnnxDomain, startver, endver, T, kMusaExecutionProvider,    \
      (*KernelDefBuilder::Create())                                      \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                        \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())         \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()), \
      Tile<T>);

REGISTER_MUSA_TILE_VERSIONED_TYPED_KERNEL(6, 12, float)
REGISTER_MUSA_TILE_VERSIONED_TYPED_KERNEL(6, 12, int32_t)
REGISTER_MUSA_TILE_VERSIONED_TYPED_KERNEL(6, 12, int64_t)
REGISTER_MUSA_TILE_VERSIONED_TYPED_KERNEL(6, 12, MLFloat16)

REGISTER_MUSA_TILE_TYPED_KERNEL(13, float)
REGISTER_MUSA_TILE_TYPED_KERNEL(13, int32_t)
REGISTER_MUSA_TILE_TYPED_KERNEL(13, int64_t)
REGISTER_MUSA_TILE_TYPED_KERNEL(13, MLFloat16)

template class Tile<float>;
template class Tile<double>;
template class Tile<int32_t>;
template class Tile<int64_t>;
template class Tile<MLFloat16>;

}  // namespace musa
}  // namespace onnxruntime
