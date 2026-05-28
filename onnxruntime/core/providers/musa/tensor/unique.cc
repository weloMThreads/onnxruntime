// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/tensor/unique.h"

#include "core/providers/musa/musa_call.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/shared_library/provider_api.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

namespace onnxruntime {
namespace musa {
namespace {

template <typename T>
struct UniqueEntry {
  T value;
  int64_t first_index;
  int64_t count;
};

template <typename T>
Status CopyDeviceToHost(musaStream_t stream, const T* src, int64_t count, std::vector<T>* dst) {
  ORT_RETURN_IF_NOT(dst != nullptr, "Unique: destination host vector is null");
  if (count == 0) {
    return Status::OK();
  }

  MUSA_RETURN_IF_ERROR(musaMemcpyAsync(dst->data(), src, static_cast<size_t>(count) * sizeof(T),
                                       musaMemcpyDeviceToHost, stream));
  MUSA_RETURN_IF_ERROR(musaStreamSynchronize(stream));
  return Status::OK();
}

template <typename T>
Status CopyHostToDevice(musaStream_t stream, const std::vector<T>& src, T* dst) {
  if (src.empty()) {
    return Status::OK();
  }

  MUSA_RETURN_IF_ERROR(musaMemcpyAsync(dst, src.data(), src.size() * sizeof(T),
                                       musaMemcpyHostToDevice, stream));
  return Status::OK();
}

template <typename T>
int64_t FindUniqueIndex(const std::vector<UniqueEntry<T>>& entries, const T& value) {
  for (size_t i = 0; i < entries.size(); ++i) {
    if (entries[i].value == value) {
      return static_cast<int64_t>(i);
    }
  }
  return -1;
}

template <typename T>
void BuildUniqueHostOutputs(const std::vector<T>& input,
                            bool sorted,
                            std::vector<T>* y,
                            std::vector<int64_t>* indices,
                            std::vector<int64_t>* inverse_indices,
                            std::vector<int64_t>* counts) {
  std::vector<UniqueEntry<T>> entries;
  entries.reserve(input.size());
  inverse_indices->resize(input.size());

  for (size_t input_index = 0; input_index < input.size(); ++input_index) {
    const int64_t unique_index = FindUniqueIndex(entries, input[input_index]);
    if (unique_index < 0) {
      entries.push_back(UniqueEntry<T>{input[input_index], static_cast<int64_t>(input_index), 1});
      (*inverse_indices)[input_index] = static_cast<int64_t>(entries.size() - 1);
    } else {
      entries[static_cast<size_t>(unique_index)].count += 1;
      (*inverse_indices)[input_index] = unique_index;
    }
  }

  const int64_t num_unique = static_cast<int64_t>(entries.size());
  y->resize(static_cast<size_t>(num_unique));
  indices->resize(static_cast<size_t>(num_unique));
  counts->resize(static_cast<size_t>(num_unique));

  std::vector<int64_t> output_order(static_cast<size_t>(num_unique));
  std::iota(output_order.begin(), output_order.end(), int64_t{0});
  if (sorted) {
    std::sort(output_order.begin(), output_order.end(), [&entries](int64_t lhs, int64_t rhs) {
      return entries[static_cast<size_t>(lhs)].value < entries[static_cast<size_t>(rhs)].value;
    });
  }

  std::vector<int64_t> unsorted_to_output(static_cast<size_t>(num_unique));
  for (int64_t output_index = 0; output_index < num_unique; ++output_index) {
    const int64_t entry_index = output_order[static_cast<size_t>(output_index)];
    const auto& entry = entries[static_cast<size_t>(entry_index)];

    (*y)[static_cast<size_t>(output_index)] = entry.value;
    (*indices)[static_cast<size_t>(output_index)] = entry.first_index;
    (*counts)[static_cast<size_t>(output_index)] = entry.count;
    unsorted_to_output[static_cast<size_t>(entry_index)] = output_index;
  }

  if (sorted) {
    for (int64_t& inverse_index : *inverse_indices) {
      inverse_index = unsorted_to_output[static_cast<size_t>(inverse_index)];
    }
  }
}

}  // namespace

template <typename T>
Unique<T>::Unique(const OpKernelInfo& info) : MusaKernel(info) {
  if (!info.GetAttr<int64_t>("axis", &axis_).IsOK()) {
    flatten_ = true;
  }
  sorted_ = info.GetAttrOrDefault<int64_t>("sorted", 1) == 1;
}

template <typename T>
Status Unique<T>::ComputeInternal(OpKernelContext* ctx) const {
  const auto* input = ctx->Input<Tensor>(0);
  ORT_RETURN_IF_NOT(input != nullptr, "Unique: input tensor is null");

  const TensorShape& input_shape = input->Shape();
  const auto input_rank = input_shape.NumDimensions();
  ORT_RETURN_IF_NOT(flatten_ || (axis_ == 0 && input_rank <= 1),
                    "Unique: MUSA EP currently supports flattened Unique or axis=0 for 1D input only");

  const int64_t input_count = input_shape.Size();
  std::vector<T> host_input(static_cast<size_t>(input_count));
  ORT_RETURN_IF_ERROR(CopyDeviceToHost(Stream(ctx), input->Data<T>(), input_count, &host_input));

  std::vector<T> host_y;
  std::vector<int64_t> host_indices;
  std::vector<int64_t> host_inverse_indices;
  std::vector<int64_t> host_counts;
  BuildUniqueHostOutputs(host_input, sorted_, &host_y, &host_indices, &host_inverse_indices, &host_counts);

  const int64_t num_unique = static_cast<int64_t>(host_y.size());
  Tensor* output = ctx->Output(0, TensorShapeVector{num_unique});
  ORT_RETURN_IF_NOT(output != nullptr, "Unique: output tensor is null");
  ORT_RETURN_IF_ERROR(CopyHostToDevice(Stream(ctx), host_y, output->MutableData<T>()));

  Tensor* indices = ctx->Output(1, TensorShapeVector{num_unique});
  if (indices != nullptr) {
    ORT_RETURN_IF_ERROR(CopyHostToDevice(Stream(ctx), host_indices, indices->MutableData<int64_t>()));
  }

  Tensor* inverse_indices = ctx->Output(2, TensorShapeVector{input_count});
  if (inverse_indices != nullptr) {
    ORT_RETURN_IF_ERROR(CopyHostToDevice(Stream(ctx), host_inverse_indices, inverse_indices->MutableData<int64_t>()));
  }

  Tensor* counts = ctx->Output(3, TensorShapeVector{num_unique});
  if (counts != nullptr) {
    ORT_RETURN_IF_ERROR(CopyHostToDevice(Stream(ctx), host_counts, counts->MutableData<int64_t>()));
  }

  MUSA_RETURN_IF_ERROR(musaStreamSynchronize(Stream(ctx)));
  return Status::OK();
}

template class Unique<float>;
template class Unique<double>;
template class Unique<int8_t>;
template class Unique<int32_t>;
template class Unique<int64_t>;
template class Unique<MLFloat16>;

#define REGISTER_MUSA_UNIQUE_TYPED_KERNEL(ver, T)                           \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                            \
      Unique, kOnnxDomain, ver, T, kMusaExecutionProvider,                  \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Unique<T>);

REGISTER_MUSA_UNIQUE_TYPED_KERNEL(11, float)
REGISTER_MUSA_UNIQUE_TYPED_KERNEL(11, double)
REGISTER_MUSA_UNIQUE_TYPED_KERNEL(11, int8_t)
REGISTER_MUSA_UNIQUE_TYPED_KERNEL(11, int32_t)
REGISTER_MUSA_UNIQUE_TYPED_KERNEL(11, int64_t)
REGISTER_MUSA_UNIQUE_TYPED_KERNEL(11, MLFloat16)

#undef REGISTER_MUSA_UNIQUE_TYPED_KERNEL

}  // namespace musa
}  // namespace onnxruntime
