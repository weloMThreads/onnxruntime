// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"

#include <musa_runtime.h>
#include <mudnn.h>

#include <string>
#include <vector>

#include "core/providers/common.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/musa/musa_execution_provider.h"
#include "core/providers/musa/musa_kernel.h"
#include "core/providers/musa/musa_utils.h"

using onnxruntime::common::Status;

namespace onnxruntime {
namespace contrib {
namespace musa {

namespace {

void NoOpDelete(void*) {}

bool NormalizeAxis(int64_t axis, size_t rank, int64_t& normalized_axis) {
  const auto rank_i64 = static_cast<int64_t>(rank);
  if (rank == 0 || axis < -rank_i64 || axis >= rank_i64) {
    return false;
  }

  normalized_axis = axis < 0 ? axis + rank_i64 : axis;
  return true;
}

bool CanReshape4DTo3D(const TensorShape& a_shape, const TensorShape& b_shape) {
  if (a_shape.NumDimensions() != 4 || b_shape.NumDimensions() != 4) {
    return false;
  }

  if (a_shape[0] != b_shape[0] || a_shape[1] != b_shape[1]) {
    return false;
  }

  for (size_t i = 0; i < 4; ++i) {
    if (a_shape[i] <= 0 || b_shape[i] <= 0) {
      return false;
    }
  }

  return true;
}

::musa::dnn::Tensor::Format GetTensorFormatForShape(const TensorShape& shape) {
  if (shape.NumDimensions() == 0 || shape.NumDimensions() == 1 || shape.NumDimensions() == 3) {
    return ::musa::dnn::Tensor::Format::NCW;
  }
  if (shape.NumDimensions() == 4) {
    return ::musa::dnn::Tensor::Format::NCHW;
  }
  if (shape.NumDimensions() == 5) {
    return ::musa::dnn::Tensor::Format::NCDHW;
  }
  return ::musa::dnn::Tensor::Format::NCHW;
}

Status SetupMusaTensorFromBuffer(::musa::dnn::Tensor& musa_tensor,
                                 const void* data_ptr,
                                 const TensorShape& shape,
                                 ::musa::dnn::Tensor::Type data_type) {
  auto status = musa_tensor.SetType(data_type);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to set MUSA scratch tensor type, status: ",
                           static_cast<int>(status));
  }

  if (data_ptr == nullptr && shape.Size() > 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "MusaConcatMatMul scratch tensor data pointer is null for non-empty tensor");
  }

  if (data_ptr != nullptr) {
    status = musa_tensor.SetAddr(data_ptr);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set MUSA scratch tensor address, status: ",
                             static_cast<int>(status));
    }
  }

  status = musa_tensor.SetFormat(GetTensorFormatForShape(shape));
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to set MUSA scratch tensor format, status: ",
                           static_cast<int>(status));
  }

  if (shape.NumDimensions() == 0) {
    std::vector<int64_t> scalar_dims = {1};
    std::vector<int64_t> scalar_strides = {1};
    status = musa_tensor.SetNdInfo(static_cast<int>(scalar_dims.size()), scalar_dims.data(), scalar_strides.data());
  } else {
    std::vector<int64_t> dims;
    dims.reserve(shape.NumDimensions());
    for (size_t i = 0; i < shape.NumDimensions(); ++i) {
      dims.push_back(shape[i]);
    }

    const auto strides = onnxruntime::musa::CalculateStrides(shape);
    status = musa_tensor.SetNdInfo(static_cast<int>(dims.size()), dims.data(), strides.data());
  }

  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to set MUSA scratch tensor shape info, status: ",
                           static_cast<int>(status));
  }

  return Status::OK();
}

Status SetupMusaTensorWithReshape(::musa::dnn::Tensor& musa_tensor,
                                  const Tensor& ort_tensor,
                                  ::musa::dnn::Tensor::Type data_type,
                                  onnxruntime::musa::MusaPreparation* preparation,
                                  int target_dims) {
  const auto& original_shape = ort_tensor.Shape();
  const auto& dims = original_shape.GetDims();
  const int num_dims = static_cast<int>(original_shape.NumDimensions());

  if (num_dims == target_dims || target_dims <= 0) {
    return onnxruntime::musa::SetupMusaTensor(musa_tensor, &ort_tensor, data_type, preparation);
  }

  std::vector<int64_t> new_dims;
  if (num_dims == 4 && target_dims == 3) {
    new_dims = {dims[0] * dims[1], dims[2], dims[3]};
  } else {
    return onnxruntime::musa::SetupMusaTensor(musa_tensor, &ort_tensor, data_type, preparation);
  }

  auto status = musa_tensor.SetType(data_type);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to set MUSA tensor type for MusaConcatMatMul reshape, status: ",
                           static_cast<int>(status));
  }

  const void* data_ptr = ort_tensor.DataRaw();
  if (data_ptr != nullptr) {
    status = musa_tensor.SetAddr(data_ptr);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set MUSA tensor address for MusaConcatMatMul reshape, status: ",
                             static_cast<int>(status));
    }
  }

  status = musa_tensor.SetFormat(::musa::dnn::Tensor::Format::NCHW);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to set MUSA tensor format for MusaConcatMatMul reshape, status: ",
                           static_cast<int>(status));
  }

  status = musa_tensor.SetNdInfo(static_cast<int>(new_dims.size()), new_dims.data());
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to set MUSA tensor shape for MusaConcatMatMul reshape, status: ",
                           static_cast<int>(status));
  }

  return Status::OK();
}

Status SetupScratchTensorWithReshape(::musa::dnn::Tensor& musa_tensor,
                                     const void* data_ptr,
                                     const TensorShape& shape,
                                     ::musa::dnn::Tensor::Type data_type,
                                     int target_dims) {
  const int num_dims = static_cast<int>(shape.NumDimensions());
  if (num_dims == target_dims || target_dims <= 0) {
    return SetupMusaTensorFromBuffer(musa_tensor, data_ptr, shape, data_type);
  }

  if (num_dims != 4 || target_dims != 3) {
    return SetupMusaTensorFromBuffer(musa_tensor, data_ptr, shape, data_type);
  }

  std::vector<int64_t> new_dims = {shape[0] * shape[1], shape[2], shape[3]};
  auto status = musa_tensor.SetType(data_type);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to set MUSA scratch tensor type for reshape, status: ",
                           static_cast<int>(status));
  }

  if (data_ptr != nullptr) {
    status = musa_tensor.SetAddr(data_ptr);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set MUSA scratch tensor address for reshape, status: ",
                             static_cast<int>(status));
    }
  }

  status = musa_tensor.SetFormat(::musa::dnn::Tensor::Format::NCHW);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to set MUSA scratch tensor format for reshape, status: ",
                           static_cast<int>(status));
  }

  status = musa_tensor.SetNdInfo(static_cast<int>(new_dims.size()), new_dims.data());
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to set MUSA scratch tensor shape for reshape, status: ",
                           static_cast<int>(status));
  }

  return Status::OK();
}

}  // namespace

template <typename T>
class ConcatMatMul final : public onnxruntime::musa::MusaKernel {
 public:
  explicit ConcatMatMul(const OpKernelInfo& info) : MusaKernel(info) {
    axis_ = info.GetAttrOrDefault<int64_t>("axis", 0);
    concat_input_idx_ = info.GetAttrOrDefault<int64_t>("concat_input_idx", 0);
  }

  Status ComputeInternal(OpKernelContext* ctx) const override {
    const int input_count = ctx->InputCount();
    ORT_RETURN_IF_NOT(input_count >= 3, "MusaConcatMatMul requires at least 2 concat inputs and 1 MatMul input.");
    ORT_RETURN_IF_NOT(concat_input_idx_ == 0 || concat_input_idx_ == 1,
                      "MusaConcatMatMul concat_input_idx must be 0 or 1.");

    const int concat_input_count = input_count - 1;
    std::vector<const Tensor*> concat_inputs;
    concat_inputs.reserve(static_cast<size_t>(concat_input_count));
    for (int i = 0; i < concat_input_count; ++i) {
      const Tensor* input = ctx->Input<Tensor>(i);
      ORT_RETURN_IF_NOT(input != nullptr, "MusaConcatMatMul concat input is null.");
      concat_inputs.push_back(input);
    }

    const Tensor* other_input = ctx->Input<Tensor>(concat_input_count);
    ORT_RETURN_IF_NOT(other_input != nullptr, "MusaConcatMatMul other MatMul input is null.");

    const Tensor* first_concat = concat_inputs.front();
    ORT_RETURN_IF_NOT(first_concat->DataType() == other_input->DataType(),
                      "MusaConcatMatMul input data types must match.");
    ORT_RETURN_IF_NOT(first_concat->Shape().NumDimensions() >= 2,
                      "MusaConcatMatMul requires concat inputs with rank >= 2.");
    ORT_RETURN_IF_NOT(first_concat->Shape().NumDimensions() == other_input->Shape().NumDimensions(),
                      "MusaConcatMatMul currently requires equal input ranks.");

    int64_t normalized_axis = 0;
    ORT_RETURN_IF_NOT(NormalizeAxis(axis_, first_concat->Shape().NumDimensions(), normalized_axis),
                      "MusaConcatMatMul axis is out of range.");

    TensorShapeVector concat_dims = first_concat->Shape().AsShapeVector();
    const size_t axis_index = static_cast<size_t>(normalized_axis);
    concat_dims[axis_index] = 0;
    for (const Tensor* input : concat_inputs) {
      ORT_RETURN_IF_NOT(input->DataType() == first_concat->DataType(),
                        "MusaConcatMatMul concat input data types must match.");
      ORT_RETURN_IF_NOT(input->Shape().NumDimensions() == first_concat->Shape().NumDimensions(),
                        "MusaConcatMatMul concat input ranks must match.");
      const auto& dims = input->Shape().GetDims();
      for (size_t i = 0; i < dims.size(); ++i) {
        if (i == axis_index) {
          continue;
        }
        ORT_RETURN_IF_NOT(dims[i] == concat_dims[i], "MusaConcatMatMul concat non-axis dimensions must match.");
      }
      concat_dims[axis_index] += dims[axis_index];
    }

    TensorShape concat_shape{concat_dims};
    const TensorShape& lhs_shape = concat_input_idx_ == 0 ? concat_shape : other_input->Shape();
    const TensorShape& rhs_shape = concat_input_idx_ == 0 ? other_input->Shape() : concat_shape;

    MatMulComputeHelper helper;
    ORT_RETURN_IF_ERROR(helper.Compute(lhs_shape, rhs_shape, false, false, false, false, false));

    Tensor* output = ctx->Output(0, helper.OutputShape());
    ORT_RETURN_IF_NOT(output != nullptr, "MusaConcatMatMul output tensor is null.");
    if (output->Shape().Size() == 0) {
      return Status::OK();
    }

    const size_t concat_storage = static_cast<size_t>(concat_shape.Size()) * first_concat->DataType()->Size();
    auto concat_buffer = GetScratchBuffer<void>(concat_storage, ctx->GetComputeStream());
    ORT_RETURN_IF_NOT(concat_buffer.get() != nullptr, "Failed to allocate MusaConcatMatMul concat scratch buffer.");

    const auto* ep = static_cast<const MusaExecutionProvider*>(Info().GetExecutionProvider());
    const auto musa_type = onnxruntime::musa::GetMusaDataType<T>();

    onnxruntime::musa::MusaPreparation concat_prepare(ep);
    if (concat_prepare.handle && Stream(ctx)) {
      auto status = concat_prepare.handle->SetStream(Stream(ctx));
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Failed to set MUSA stream for MusaConcatMatMul concat path, status: ",
                               static_cast<int>(status));
      }
    }

    concat_prepare.inputTensors.resize(static_cast<size_t>(concat_input_count));
    concat_prepare.outputTensors.resize(1);
    for (int i = 0; i < concat_input_count; ++i) {
      ORT_RETURN_IF_ERROR(onnxruntime::musa::SetupMusaTensor(
          concat_prepare.inputTensors[static_cast<size_t>(i)], concat_inputs[static_cast<size_t>(i)], musa_type, &concat_prepare));
    }
    ORT_RETURN_IF_ERROR(SetupMusaTensorFromBuffer(concat_prepare.outputTensors[0], concat_buffer.get(), concat_shape, musa_type));

    ::musa::dnn::Concat concat_op;
    auto status = concat_op.SetAxis(static_cast<int>(normalized_axis));
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set MusaConcatMatMul concat axis, status: ",
                             static_cast<int>(status));
    }

    ::musa::dnn::Tensor mutable_concat_output = concat_prepare.outputTensors[0];
    status = concat_op.Run(concat_prepare.GetHandle(), mutable_concat_output,
                           static_cast<int>(concat_prepare.inputTensors.size()), concat_prepare.inputTensors.data());
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "MusaConcatMatMul concat execution failed, status: ",
                             static_cast<int>(status));
    }

    onnxruntime::musa::MusaPreparation matmul_prepare(ep);
    if (matmul_prepare.handle && Stream(ctx)) {
      status = matmul_prepare.handle->SetStream(Stream(ctx));
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Failed to set MUSA stream for MusaConcatMatMul matmul path, status: ",
                               static_cast<int>(status));
      }
    }

    const bool reshape_to_3d = CanReshape4DTo3D(lhs_shape, rhs_shape);
    matmul_prepare.inputTensors.resize(2);
    matmul_prepare.outputTensors.resize(1);
    if (concat_input_idx_ == 0) {
      if (reshape_to_3d) {
        ORT_RETURN_IF_ERROR(SetupScratchTensorWithReshape(matmul_prepare.inputTensors[0], concat_buffer.get(), concat_shape, musa_type, 3));
        ORT_RETURN_IF_ERROR(SetupMusaTensorWithReshape(matmul_prepare.inputTensors[1], *other_input, musa_type, &matmul_prepare, 3));
      } else {
        ORT_RETURN_IF_ERROR(SetupMusaTensorFromBuffer(matmul_prepare.inputTensors[0], concat_buffer.get(), concat_shape, musa_type));
        ORT_RETURN_IF_ERROR(onnxruntime::musa::SetupMusaTensor(matmul_prepare.inputTensors[1], other_input, musa_type, &matmul_prepare));
      }
    } else {
      if (reshape_to_3d) {
        ORT_RETURN_IF_ERROR(SetupMusaTensorWithReshape(matmul_prepare.inputTensors[0], *other_input, musa_type, &matmul_prepare, 3));
        ORT_RETURN_IF_ERROR(SetupScratchTensorWithReshape(matmul_prepare.inputTensors[1], concat_buffer.get(), concat_shape, musa_type, 3));
      } else {
        ORT_RETURN_IF_ERROR(onnxruntime::musa::SetupMusaTensor(matmul_prepare.inputTensors[0], other_input, musa_type, &matmul_prepare));
        ORT_RETURN_IF_ERROR(SetupMusaTensorFromBuffer(matmul_prepare.inputTensors[1], concat_buffer.get(), concat_shape, musa_type));
      }
    }

    if (reshape_to_3d) {
      ORT_RETURN_IF_ERROR(SetupMusaTensorWithReshape(matmul_prepare.outputTensors[0], *output, musa_type, &matmul_prepare, 3));
    } else {
      ORT_RETURN_IF_ERROR(onnxruntime::musa::SetupMusaTensor(matmul_prepare.outputTensors[0], output, musa_type, &matmul_prepare));
    }

    const bool use_batch = lhs_shape.NumDimensions() > 2 || rhs_shape.NumDimensions() > 2;
    std::vector<IAllocatorUniquePtr<void>> workspace_buffers_holder;
    auto* ort_stream = ctx->GetComputeStream();
    auto memory_allocator = [this, ort_stream, &workspace_buffers_holder](size_t size) -> ::musa::dnn::MemoryHandler {
      if (size == 0) {
        return ::musa::dnn::MemoryHandler(nullptr, NoOpDelete);
      }
      auto scratch = this->GetScratchBuffer<void>(size, ort_stream);
      void* ptr = scratch.get();
      workspace_buffers_holder.push_back(std::move(scratch));
      return ::musa::dnn::MemoryHandler(ptr, NoOpDelete);
    };

    if (use_batch) {
      ::musa::dnn::BatchMatMul batch_op;
      ORT_RETURN_IF_NOT(batch_op.SetAlpha(1.0) == ::musa::dnn::Status::SUCCESS, "Failed to set MusaConcatMatMul BatchMatMul alpha.");
      ORT_RETURN_IF_NOT(batch_op.SetBeta(0.0) == ::musa::dnn::Status::SUCCESS, "Failed to set MusaConcatMatMul BatchMatMul beta.");
      ORT_RETURN_IF_NOT(batch_op.SetTranspose(false, false) == ::musa::dnn::Status::SUCCESS,
                        "Failed to set MusaConcatMatMul BatchMatMul transpose.");
      ORT_RETURN_IF_NOT(batch_op.SetComputeMode(::musa::dnn::BatchMatMul::ComputeMode::TENSOR) == ::musa::dnn::Status::SUCCESS,
                        "Failed to set MusaConcatMatMul BatchMatMul compute mode.");
      status = batch_op.Run(matmul_prepare.GetHandle(),
                            const_cast<::musa::dnn::Tensor&>(matmul_prepare.outputTensors[0]),
                            matmul_prepare.inputTensors[0], matmul_prepare.inputTensors[1], memory_allocator);
    } else {
      ::musa::dnn::MatMul matmul_op;
      ORT_RETURN_IF_NOT(matmul_op.SetAlpha(1.0) == ::musa::dnn::Status::SUCCESS, "Failed to set MusaConcatMatMul MatMul alpha.");
      ORT_RETURN_IF_NOT(matmul_op.SetBeta(0.0) == ::musa::dnn::Status::SUCCESS, "Failed to set MusaConcatMatMul MatMul beta.");
      ORT_RETURN_IF_NOT(matmul_op.SetTranspose(false, false) == ::musa::dnn::Status::SUCCESS,
                        "Failed to set MusaConcatMatMul MatMul transpose.");
      ORT_RETURN_IF_NOT(matmul_op.SetComputeMode(::musa::dnn::MatMul::ComputeMode::TENSOR) == ::musa::dnn::Status::SUCCESS,
                        "Failed to set MusaConcatMatMul MatMul compute mode.");
      status = matmul_op.Run(matmul_prepare.GetHandle(),
                             const_cast<::musa::dnn::Tensor&>(matmul_prepare.outputTensors[0]),
                             matmul_prepare.inputTensors[0], matmul_prepare.inputTensors[1], memory_allocator);
    }

    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "MusaConcatMatMul matmul execution failed, status: ",
                             static_cast<int>(status));
    }

    return Status::OK();
  }

 private:
  int64_t axis_ = 0;
  int64_t concat_input_idx_ = 0;
};

#define REGISTER_MUSA_CONCAT_MATMUL_TYPED(T)                     \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                 \
      MusaConcatMatMul, kMSDomain, 1, T, kMusaExecutionProvider, \
      (*KernelDefBuilder::Create())                              \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      ConcatMatMul<T>);

REGISTER_MUSA_CONCAT_MATMUL_TYPED(float)
REGISTER_MUSA_CONCAT_MATMUL_TYPED(MLFloat16)

}  // namespace musa
}  // namespace contrib
}  // namespace onnxruntime
