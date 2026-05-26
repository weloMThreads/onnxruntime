// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/nn/conv_transpose.h"
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/common.h"
#include "core/providers/musa/musa_call.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/musa_execution_provider.h"
#include "core/providers/musa/musa_utils.h"
#include "core/providers/musa/tensor/transpose.h"

#include <musa_runtime.h>
#include <mudnn.h>

#include <string>
#include <vector>

namespace onnxruntime {
namespace musa {

#define REGISTER_MUSA_CONV_TRANSPOSE_TYPED_KERNEL(T, DOMAIN, NHWC)                           \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                    \
      ConvTranspose, DOMAIN, 1, 10, T, kMusaExecutionProvider,                               \
      (*KernelDefBuilder::Create())                                                           \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),                           \
      ConvTranspose<T, NHWC>);                                                                \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                              \
      ConvTranspose, DOMAIN, 11, T, kMusaExecutionProvider,                                  \
      (*KernelDefBuilder::Create())                                                           \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),                           \
      ConvTranspose<T, NHWC>);

// V0.17.4: ConvTranspose1D is handled by metadata-only 1D->2D unsqueeze before
// calling muDNN RunBwdData, so these registrations can run without CPU fallback.
REGISTER_MUSA_CONV_TRANSPOSE_TYPED_KERNEL(float, kOnnxDomain, false)
REGISTER_MUSA_CONV_TRANSPOSE_TYPED_KERNEL(MLFloat16, kOnnxDomain, false)

#ifdef ENABLE_MUSA_NHWC_OPS
REGISTER_MUSA_CONV_TRANSPOSE_TYPED_KERNEL(float, kMSInternalNHWCDomain, true)
REGISTER_MUSA_CONV_TRANSPOSE_TYPED_KERNEL(MLFloat16, kMSInternalNHWCDomain, true)
#endif

namespace {

Status SetMusaTensorFormat(::musa::dnn::Tensor& tensor, int64_t rank, bool channels_last, const char* tensor_name) {
  ::musa::dnn::Tensor::Format format = ::musa::dnn::Tensor::Format::UNKNOWN;
  if (rank == 3) {
    format = channels_last ? ::musa::dnn::Tensor::Format::NHWC
                           : ::musa::dnn::Tensor::Format::NCW;
  } else if (rank == 4) {
    format = channels_last ? ::musa::dnn::Tensor::Format::NHWC
                           : ::musa::dnn::Tensor::Format::NCHW;
  } else if (rank == 5) {
    format = channels_last ? ::musa::dnn::Tensor::Format::NDHWC
                           : ::musa::dnn::Tensor::Format::NCDHW;
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Unsupported ConvTranspose ", tensor_name, " tensor rank: ", rank);
  }

  auto status = tensor.SetFormat(format);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to set ConvTranspose ", tensor_name, " tensor format");
  }

  return Status::OK();
}

Status SetMusaTensorShape(::musa::dnn::Tensor& tensor,
                          const TensorShapeVector& dims,
                          ::musa::dnn::Tensor::Format format,
                          const char* tensor_name) {
  auto status = tensor.SetFormat(format);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to set ConvTranspose ", tensor_name, " tensor format");
  }

  status = tensor.SetNdInfo(static_cast<int>(dims.size()), dims.data());
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to set ConvTranspose ", tensor_name, " tensor shape");
  }

  return Status::OK();
}

TensorShapeVector UnsqueezeConvTranspose1DShape(const TensorShape& shape, bool channels_last) {
  if (channels_last) {
    return {shape[0], 1, shape[1], shape[2]};
  }
  return {shape[0], shape[1], 1, shape[2]};
}

std::vector<int> UnsqueezeSpatial1D(const TensorShapeVector& values, int inserted_dim_value) {
  const int value = values.empty() ? inserted_dim_value : static_cast<int>(values[0]);
  return {inserted_dim_value, value};
}

std::vector<int> UnsqueezeSpatial1D(const ConvTransposeAttributes::ConvPadVector& values,
                                    int inserted_dim_value) {
  const int value = values.empty() ? inserted_dim_value : static_cast<int>(values[0]);
  return {inserted_dim_value, value};
}

Status RunConvTransposeBwdData(::musa::dnn::Convolution& conv_op,
                               ::musa::dnn::Handle& handle,
                               ::musa::dnn::Tensor& y_tensor,
                               ::musa::dnn::Tensor& x_tensor,
                               ::musa::dnn::Tensor& w_tensor,
                               ::musa::dnn::MemoryMaintainer& maintainer) {
  auto status = conv_op.RunBwdData(handle,
                                   y_tensor,
                                   x_tensor,
                                   w_tensor,
                                   ::musa::dnn::Convolution::AlgorithmBwdData::GEMM,
                                   maintainer);
  if (status == ::musa::dnn::Status::SUCCESS) {
    return Status::OK();
  }

  const auto gemm_status = status;
  status = conv_op.RunBwdData(handle,
                              y_tensor,
                              x_tensor,
                              w_tensor,
                              ::musa::dnn::Convolution::AlgorithmBwdData::IMPLICIT_GEMM,
                              maintainer);
  if (status == ::musa::dnn::Status::SUCCESS) {
    return Status::OK();
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                         "mudnn ConvTranspose RunBwdData failed, GEMM status: ",
                         static_cast<int>(gemm_status),
                         ", IMPLICIT_GEMM status: ",
                         static_cast<int>(status));
}

}  // namespace

template <typename T, bool NHWC>
Status ConvTranspose<T, NHWC>::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                                       bool& is_packed, PrePackedWeights* prepacked_weights) {
  is_packed = false;
  ORT_UNUSED_PARAMETER(prepacked_weights);

  if constexpr (NHWC) {
    if (is_nhwc_domain_ && input_idx == 1) {
      const auto orig_shape = tensor.Shape();
      const auto& shape_dims = orig_shape.GetDims();
      const auto rank = shape_dims.size();

      ORT_ENFORCE(rank >= 3, "ConvTranspose weight tensor must have rank >= 3 for NHWC");

      const auto perm = GetNHWCWeightPerm(rank);
      std::vector<int64_t> nhwc_dims;
      nhwc_dims.reserve(rank);
      for (size_t i = 0; i < rank; ++i) {
        nhwc_dims.push_back(shape_dims[perm[i]]);
      }

      W_ = Tensor::Create(tensor.DataType(), TensorShape(nhwc_dims), std::move(alloc));
      ORT_RETURN_IF_ERROR(DoMusaTranspose(tensor.DataRaw(),
                                          W_->MutableDataRaw(),
                                          std::vector<int64_t>(shape_dims.begin(), shape_dims.end()),
                                          perm,
                                          GetMusaDataType<T>(),
                                          nullptr,
                                          Info().GetExecutionProvider()->GetDeviceId()));
      MUSA_CALL(musaStreamSynchronize(nullptr));
      is_packed = true;
    } else if (input_idx == 1) {
      W_already_nhwc = true;
    }
  } else {
    ORT_UNUSED_PARAMETER(tensor);
    ORT_UNUSED_PARAMETER(input_idx);
    ORT_UNUSED_PARAMETER(alloc);
    ORT_UNUSED_PARAMETER(prepacked_weights);
  }

  return Status::OK();
}

template <typename T, bool NHWC>
Status ConvTranspose<T, NHWC>::ComputeInternal(OpKernelContext* context) const {
  constexpr bool channels_last = NHWC;

  const Tensor* X = context->Input<Tensor>(0);
  const Tensor* W = W_ ? W_.get() : context->Input<Tensor>(1);
  if (X == nullptr || W == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "ConvTranspose input X or W is null");
  }

  const Tensor* B = context->Input<Tensor>(2);
  const bool has_bias = (B != nullptr);
  const bool w_in_nhwc = channels_last && (W_ != nullptr || W_already_nhwc);

  ConvTransposeAttributes::Prepare p;
  if constexpr (channels_last) {
    ORT_ENFORCE(w_in_nhwc, "ConvTranspose NHWC requires NHWC-packed weights");
    ORT_RETURN_IF_ERROR(conv_transpose_attrs_.PrepareForCompute(
        context, has_bias, p, false, &W->Shape(), true, false));
  } else {
    ORT_RETURN_IF_ERROR(conv_transpose_attrs_.PrepareForCompute(context, has_bias, p));
  }

  Tensor* Y = p.Y;
  if (Y == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to allocate ConvTranspose output tensor");
  }

  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  const auto* ep = static_cast<const MusaExecutionProvider*>(Info().GetExecutionProvider());
  MusaPreparation prepare(ep);

  auto* ort_stream = context->GetComputeStream();
  auto* musa_stream = Stream(context);
  if (prepare.handle != nullptr && musa_stream != nullptr) {
    auto handle_status = prepare.handle->SetStream(musa_stream);
    if (handle_status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set MUSA stream, status: " +
                                 std::to_string(static_cast<int>(handle_status)));
    }
  }

  const auto musa_type = GetMusaDataType<T>();
  const bool is_1d_unsqueeze = X->Shape().NumDimensions() == 3 && W->Shape().NumDimensions() == 3 &&
                               Y->Shape().NumDimensions() == 3;

  prepare.inputTensors.resize(2);
  prepare.outputTensors.resize(1);

  ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.inputTensors[0], X, musa_type, &prepare));
  ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.inputTensors[1], W, musa_type, &prepare));
  ORT_RETURN_IF_ERROR(SetupMusaTensor(prepare.outputTensors[0], Y, musa_type, &prepare));
  if (is_1d_unsqueeze) {
    const auto x_4d_shape = UnsqueezeConvTranspose1DShape(X->Shape(), channels_last);
    const auto w_4d_shape = UnsqueezeConvTranspose1DShape(W->Shape(), w_in_nhwc);
    const auto y_4d_shape = UnsqueezeConvTranspose1DShape(Y->Shape(), channels_last);
    ORT_RETURN_IF_ERROR(SetMusaTensorShape(prepare.inputTensors[0], x_4d_shape,
                                           channels_last ? ::musa::dnn::Tensor::Format::NHWC
                                                         : ::musa::dnn::Tensor::Format::NCHW,
                                           "input"));
    ORT_RETURN_IF_ERROR(SetMusaTensorShape(prepare.inputTensors[1], w_4d_shape,
                                           w_in_nhwc ? ::musa::dnn::Tensor::Format::NHWC
                                                     : ::musa::dnn::Tensor::Format::NCHW,
                                           "weight"));
    ORT_RETURN_IF_ERROR(SetMusaTensorShape(prepare.outputTensors[0], y_4d_shape,
                                           channels_last ? ::musa::dnn::Tensor::Format::NHWC
                                                         : ::musa::dnn::Tensor::Format::NCHW,
                                           "output"));
  } else {
    ORT_RETURN_IF_ERROR(SetMusaTensorFormat(prepare.inputTensors[0], X->Shape().NumDimensions(), channels_last, "input"));
    ORT_RETURN_IF_ERROR(SetMusaTensorFormat(prepare.inputTensors[1], W->Shape().NumDimensions(), w_in_nhwc, "weight"));
    ORT_RETURN_IF_ERROR(SetMusaTensorFormat(prepare.outputTensors[0], Y->Shape().NumDimensions(), channels_last, "output"));
  }

  ::musa::dnn::Convolution conv_op;

  auto status = conv_op.SetGroups(static_cast<int>(conv_transpose_attrs_.group));
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set ConvTranspose groups");
  }

  size_t spatial_dims = p.kernel_shape.size();
  std::vector<int> pad_ints(spatial_dims, 0);
  std::vector<int> stride_ints(spatial_dims, 1);
  std::vector<int> dilation_ints(spatial_dims, 1);

  if (is_1d_unsqueeze) {
    spatial_dims = 2;
    pad_ints = UnsqueezeSpatial1D(p.pads, 0);
    stride_ints = UnsqueezeSpatial1D(p.strides, 1);
    dilation_ints = UnsqueezeSpatial1D(p.dilations, 1);
  } else {
    for (size_t i = 0; i < spatial_dims; ++i) {
      pad_ints[i] = static_cast<int>(p.pads[i]);
      stride_ints[i] = static_cast<int>(p.strides[i]);
      dilation_ints[i] = static_cast<int>(p.dilations[i]);
    }
  }

  status = conv_op.SetNdInfo(static_cast<int>(spatial_dims),
                             pad_ints.data(),
                             stride_ints.data(),
                             dilation_ints.data());
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set ConvTranspose nd info");
  }

  status = conv_op.SetComputeMode(::musa::dnn::Convolution::ComputeMode::TENSOR);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set ConvTranspose compute mode");
  }

  std::vector<IAllocatorUniquePtr<void>> workspace_buffers_holder;
  auto memory_allocator = [this, ort_stream, &workspace_buffers_holder](size_t size) -> ::musa::dnn::MemoryHandler {
    if (size == 0) {
      return ::musa::dnn::MemoryHandler(nullptr, [](void*) {});
    }

    auto scratch = GetScratchBuffer<void>(size, ort_stream);
    void* ptr = scratch.get();
    workspace_buffers_holder.push_back(std::move(scratch));
    return ::musa::dnn::MemoryHandler(ptr, [](void*) {});
  };
  ::musa::dnn::MemoryMaintainer maintainer = memory_allocator;

  ORT_RETURN_IF_ERROR(RunConvTransposeBwdData(conv_op,
                                              prepare.GetHandle(),
                                              prepare.outputTensors[0],
                                              prepare.inputTensors[0],
                                              prepare.inputTensors[1],
                                              maintainer));

  if (has_bias) {
    ::musa::dnn::Binary bias_add_op;
    status = bias_add_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set mudnn Binary mode for ConvTranspose bias add");
    }

    ::musa::dnn::Tensor bias_tensor;
    status = bias_tensor.SetType(musa_type);
    ORT_RETURN_IF_NOT(status == ::musa::dnn::Status::SUCCESS, "Failed to set ConvTranspose bias tensor type");

    status = bias_tensor.SetAddr(B->DataRaw());
    ORT_RETURN_IF_NOT(status == ::musa::dnn::Status::SUCCESS, "Failed to set ConvTranspose bias tensor address");

    const auto rank = is_1d_unsqueeze ? 4 : Y->Shape().NumDimensions();
    if (is_1d_unsqueeze) {
      status = bias_tensor.SetFormat(channels_last ? ::musa::dnn::Tensor::Format::NHWC
                                                   : ::musa::dnn::Tensor::Format::NCHW);
    } else if (rank == 3 && !channels_last) {
      status = bias_tensor.SetFormat(::musa::dnn::Tensor::Format::NCW);
    } else if ((rank == 3 || rank == 4) && channels_last) {
      status = bias_tensor.SetFormat(::musa::dnn::Tensor::Format::NHWC);
    } else if (rank == 4) {
      status = bias_tensor.SetFormat(::musa::dnn::Tensor::Format::NCHW);
    } else if (rank == 5 && channels_last) {
      status = bias_tensor.SetFormat(::musa::dnn::Tensor::Format::NDHWC);
    } else if (rank == 5) {
      status = bias_tensor.SetFormat(::musa::dnn::Tensor::Format::NCDHW);
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Unsupported ConvTranspose output rank for bias add: ", rank);
    }
    ORT_RETURN_IF_NOT(status == ::musa::dnn::Status::SUCCESS, "Failed to set ConvTranspose bias tensor format");

    std::vector<int64_t> bias_broadcast_shape(rank, 1);
    bias_broadcast_shape[channels_last ? rank - 1 : 1] = B->Shape()[0];
    status = bias_tensor.SetNdInfo(static_cast<int>(bias_broadcast_shape.size()), bias_broadcast_shape.data());
    ORT_RETURN_IF_NOT(status == ::musa::dnn::Status::SUCCESS, "Failed to set ConvTranspose bias tensor shape");

    status = bias_add_op.Run(prepare.GetHandle(),
                             prepare.outputTensors[0],
                             prepare.outputTensors[0],
                             bias_tensor);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "mudnn ConvTranspose bias add failed, status: " +
                                 std::to_string(static_cast<int>(status)));
    }
  }

  return Status::OK();
}

template class ConvTranspose<float, false>;
template class ConvTranspose<MLFloat16, false>;
#ifdef ENABLE_MUSA_NHWC_OPS
template class ConvTranspose<float, true>;
template class ConvTranspose<MLFloat16, true>;
#endif

}  // namespace musa
}  // namespace onnxruntime
