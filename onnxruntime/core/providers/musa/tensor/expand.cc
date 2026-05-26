// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/tensor/expand.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/musa_call.h"
#include "core/providers/musa/musa_execution_provider.h"
#include <musa_runtime.h>
#include <mudnn.h>

using onnxruntime::common::Status;
namespace onnxruntime {
namespace musa {

template <typename T>
Status Expand<T>::InferOutputShape(const Tensor* input, const Tensor* shape,
                                   TensorShape& output_shape) const {
  // Input validation
  ORT_ENFORCE(shape != nullptr, "Shape tensor cannot be null");
  ORT_ENFORCE(shape->Shape().NumDimensions() == 1,
              "Shape tensor must be 1-dimensional");

  // Get target shape from shape tensor
  // Handle empty shape tensor (scalar to scalar expand)
  const int64_t shape_size = shape->Shape().Size();
  if (shape_size == 0) {
    // Empty shape tensor means expand to scalar - output shape equals input shape
    output_shape = input->Shape();
    return Status::OK();
  }

  const auto* p_shape = shape->Data<int64_t>();
  if (p_shape == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Shape tensor data pointer is null");
  }

  TensorShapeVector target_shape_vec(p_shape, p_shape + shape_size);

  // Validate target shape values - only reject negative values
  // Note: dim==0 is allowed (results in empty output tensor, matching CPU EP behavior)
  for (size_t i = 0; i < target_shape_vec.size(); ++i) {
    int64_t dim = target_shape_vec[i];
    if (dim < 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Expand: negative values in target shape are not allowed. Found " +
                                 std::to_string(dim) + " at position " + std::to_string(i));
    }
  }

  // Compute the actual broadcast output shape using numpy broadcasting rules
  ORT_RETURN_IF_ERROR(ComputeBroadcastShape(input->Shape(), target_shape_vec, output_shape));

  return Status::OK();
}

template <typename T>
Status Expand<T>::ComputeBroadcastShape(const TensorShape& input_shape,
                                        const TensorShapeVector& target_shape_vec,
                                        TensorShape& output_shape) const {
  const auto& input_dims = input_shape.GetDims();
  size_t input_rank = input_dims.size();
  size_t target_rank = target_shape_vec.size();
  size_t output_rank = std::max(input_rank, target_rank);

  TensorShapeVector output_dims(output_rank, 1);

  // Process from right to left (least significant dimension first)
  for (size_t i = 0; i < output_rank; ++i) {
    int64_t input_dim = (i < input_rank) ? input_dims[input_rank - 1 - i] : 1;
    int64_t target_dim = (i < target_rank) ? target_shape_vec[target_rank - 1 - i] : 1;

    // Broadcasting rules:
    // 1. Dimensions with size 1 can be broadcast to any size
    // 2. Dimensions must be equal or one must be 1
    int64_t output_dim;
    if (input_dim == target_dim) {
      output_dim = input_dim;
    } else if (input_dim == 1) {
      output_dim = target_dim;
    } else if (target_dim == 1) {
      output_dim = input_dim;
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Expand: Cannot broadcast dimension " + std::to_string(input_dim) +
                                 " with dimension " + std::to_string(target_dim));
    }

    output_dims[output_rank - 1 - i] = output_dim;
  }

  output_shape = TensorShape(output_dims);
  return Status::OK();
}

template <typename T>
Status Expand<T>::ValidateBroadcast(const TensorShape& input_shape,
                                    const TensorShape& target_shape) const {
  const auto& input_dims = input_shape.GetDims();
  const auto& target_dims = target_shape.GetDims();

  // Get the number of dimensions
  size_t input_ndim = input_dims.size();
  size_t target_ndim = target_dims.size();

  // Target must have at least as many dimensions as input (or equal)
  if (target_ndim < input_ndim) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Expand: target shape " + target_shape.ToString() +
                               " has fewer dimensions than input shape " + input_shape.ToString());
  }

  // Check broadcasting compatibility from right to left
  // Only need to check the overlapping dimensions (rightmost input_ndim dimensions of target)
  size_t start_offset = target_ndim - input_ndim;  // offset in target_dims to align with input_dims

  for (size_t i = 0; i < input_ndim; ++i) {
    int64_t input_dim = input_dims[i];
    int64_t target_dim = target_dims[start_offset + i];

    // Broadcasting rule for expand:
    // - input dimension can be 1 (expands to target dimension)
    // - input dimension equals target dimension (no change)
    // - target dimension can be 1 (input dimension contracts/reshapes)
    if (input_dim != 1 && target_dim != 1 && input_dim != target_dim) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Expand: input dimension " + std::to_string(input_dim) +
                                 " at position " + std::to_string(i) +
                                 " cannot be expanded to target dimension " + std::to_string(target_dim));
    }
  }

  return Status::OK();
}

template <typename T>
Status Expand<T>::Prepare(OpKernelContext* ctx, MusaPreparation& prepare) const {
  // Get input tensors
  const Tensor* input_tensor = ctx->Input<Tensor>(0);
  const Tensor* shape_tensor = ctx->Input<Tensor>(1);

  if (!input_tensor || !shape_tensor) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Expand: input_tensor or shape_tensor is null");
  }

  // Infer output shape
  TensorShape output_shape;
  ORT_RETURN_IF_ERROR(InferOutputShape(input_tensor, shape_tensor, output_shape));

  // Allocate output tensor
  Tensor* output_tensor = ctx->Output(0, output_shape);
  if (!output_tensor) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor");
  }

  // Store basic info in preparation
  prepare.input_a_ptr = input_tensor->DataRaw();
  prepare.output_ptr = output_tensor->MutableDataRaw();
  prepare.output_size = output_shape.Size();
  prepare.input_a_shape = input_tensor->Shape();
  prepare.output_shape = output_shape;

  // For empty output tensor (when any dimension is 0), data pointers may be null
  // This is valid - ComputeInternal will handle the early return
  if (prepare.output_size > 0) {
    if (!prepare.input_a_ptr || !prepare.output_ptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid tensor data pointers");
    }

    // Setup MUSA tensors only for non-empty output
    ORT_TRY {
      // Get MUSA data type
      const auto musaType = GetMusaDataType<T>();

      // Set up input tensor
      prepare.inputTensors.resize(1);
      ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.inputTensors[0], input_tensor,
                                          musaType, &prepare));

      // Set up output tensor
      prepare.outputTensors.resize(1);
      ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.outputTensors[0], output_tensor,
                                          musaType, &prepare));
    }
    ORT_CATCH(const std::exception& e) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to setup MUSA tensors: ", e.what());
    }
  }

  return Status::OK();
}

template <typename T>
Status Expand<T>::ComputeInternal(OpKernelContext* ctx) const {
  const auto* ep = static_cast<const MusaExecutionProvider*>(
      Info().GetExecutionProvider());
  MusaPreparation prepare(ep);
  ORT_RETURN_IF_ERROR(Prepare(ctx, prepare));

  // Handle empty output tensor (when any dimension is 0)
  // This is a valid ONNX scenario, e.g., batch_size=0
  if (prepare.output_size == 0) {
    return Status::OK();
  }

  // Use unified Binary ADD approach for expand operation
  try {
    // Set up MUSA stream for asynchronous execution
    // In EP mode, stream is already set in PerThreadContext constructor
    // In legacy mode, we need to set stream here
    auto* stream = Stream(ctx);
    if (prepare.handle && stream) {
      auto musa_status = prepare.handle->SetStream(stream);
      if (musa_status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Failed to set MUSA stream, status: " +
                                   std::to_string(static_cast<int>(musa_status)));
      }
    }

    // For both scalar and non-scalar cases, use Binary ADD operation approach
    // This avoids muDNN Fill operation's potential issues with integer types

    {
      // Unified approach - use muDNN Binary operation for broadcasting (works for both scalar and non-scalar)

      // Create a zeros tensor with the same shape as output
      ::musa::dnn::Fill zero_fill_op;
      zero_fill_op.SetValue(0.0);  // Fill with zeros

      // Create temporary zero tensor (we'll use output as temporary)
      auto status = zero_fill_op.Run(prepare.GetHandle(),
                                     const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]));
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "muDNN zero fill failed, status: " +
                                   std::to_string(static_cast<int>(status)));
      }

      // Now use Binary ADD operation: input + zeros = input (with broadcasting)
      ::musa::dnn::Binary add_op;
      add_op.SetMode(::musa::dnn::Binary::Mode::ADD);

      status = add_op.Run(prepare.GetHandle(),
                          const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]),  // output =
                          prepare.inputTensors[0],                                     // input
                          prepare.outputTensors[0]);                                   // + zeros (already in output)

      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "muDNN Binary ADD operation failed, status: " +
                                   std::to_string(static_cast<int>(status)));
      }
    }

  } catch (const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Exception in Expand operation: " + std::string(e.what()));
  }

  return Status::OK();
}

// Explicit template instantiations for supported types
template class Expand<float>;
template class Expand<int32_t>;
template class Expand<int64_t>;
template class Expand<MLFloat16>;

// Register kernels with ONNX Runtime
#define REGISTER_MUSA_EXPAND_TYPED_KERNEL(ver, T)                \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                 \
      Expand, kOnnxDomain, ver, T, kMusaExecutionProvider,       \
      (*KernelDefBuilder::Create())                              \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()) \
          .InputMemoryType(OrtMemTypeCPUInput, 1),               \
      Expand<T>);

#define REGISTER_MUSA_EXPAND_VERSIONED_TYPED_KERNEL(name, startver, endver, T) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                     \
      name, kOnnxDomain, startver, endver, T, kMusaExecutionProvider,          \
      (*KernelDefBuilder::Create())                                            \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())               \
          .InputMemoryType(OrtMemTypeCPUInput, 1),                             \
      name<T>);

// Register for different ONNX versions
REGISTER_MUSA_EXPAND_VERSIONED_TYPED_KERNEL(Expand, 8, 12, float)
REGISTER_MUSA_EXPAND_VERSIONED_TYPED_KERNEL(Expand, 8, 12, int32_t)
REGISTER_MUSA_EXPAND_VERSIONED_TYPED_KERNEL(Expand, 8, 12, int64_t)
REGISTER_MUSA_EXPAND_VERSIONED_TYPED_KERNEL(Expand, 8, 12, MLFloat16)

REGISTER_MUSA_EXPAND_TYPED_KERNEL(13, float)
REGISTER_MUSA_EXPAND_TYPED_KERNEL(13, int32_t)
REGISTER_MUSA_EXPAND_TYPED_KERNEL(13, int64_t)
REGISTER_MUSA_EXPAND_TYPED_KERNEL(13, MLFloat16)

}  // namespace musa
}  // namespace onnxruntime
