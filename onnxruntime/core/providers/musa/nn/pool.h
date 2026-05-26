// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/musa/musa_kernel.h"
#include "core/providers/musa/musa_execution_provider.h"
#include "core/providers/cpu/nn/pool_base.h"

namespace onnxruntime {
namespace musa {

namespace detail {

template <typename PoolType>
struct PoolOpTraits;

template <>
struct PoolOpTraits<AveragePool> {
  static constexpr bool kHasIndices = false;

  static ::musa::dnn::Pooling::Mode GetMudnnMode(const PoolAttributes& pool_attrs) {
    if (pool_attrs.global_pooling) {
      return ::musa::dnn::Pooling::Mode::GLOBAL_AVGPOOL;
    }

    return pool_attrs.count_include_pad ? ::musa::dnn::Pooling::Mode::AVGPOOL_COUNT_PAD
                                        : ::musa::dnn::Pooling::Mode::AVGPOOL_COUNT_WITHOUT_PAD;
  }
};

template <>
struct PoolOpTraits<MaxPool<1>> {
  static constexpr bool kHasIndices = false;

  static ::musa::dnn::Pooling::Mode GetMudnnMode(const PoolAttributes&) {
    return ::musa::dnn::Pooling::Mode::MAXPOOL;
  }
};

template <>
struct PoolOpTraits<MaxPool<8>> {
  static constexpr bool kHasIndices = true;

  static ::musa::dnn::Pooling::Mode GetMudnnMode(const PoolAttributes&) {
    return ::musa::dnn::Pooling::Mode::MAXPOOL;
  }
};

inline Status ToStatus(::musa::dnn::Status status, const std::string& op_name, const char* step) {
  if (status == ::musa::dnn::Status::SUCCESS) {
    return Status::OK();
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, op_name, ": ", step,
                         " failed with muDNN status ", static_cast<int>(status));
}

inline ::musa::dnn::Tensor::Format GetTensorFormat(size_t rank) {
  switch (rank) {
    case 3:
      return ::musa::dnn::Tensor::Format::NCW;
    case 4:
      return ::musa::dnn::Tensor::Format::NCHW;
    case 5:
      return ::musa::dnn::Tensor::Format::NCDHW;
    default:
      return ::musa::dnn::Tensor::Format::UNKNOWN;
  }
}

inline ::musa::dnn::Tensor::Format GetNhwcTensorFormat(size_t rank) {
  switch (rank) {
    case 4:
      return ::musa::dnn::Tensor::Format::NHWC;
    case 5:
      return ::musa::dnn::Tensor::Format::NDHWC;
    default:
      return ::musa::dnn::Tensor::Format::UNKNOWN;
  }
}

inline std::vector<int64_t> GetContiguousStrides(const std::vector<int64_t>& dims) {
  std::vector<int64_t> strides(dims.size(), 1);
  for (int64_t i = static_cast<int64_t>(dims.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * dims[static_cast<size_t>(i + 1)];
  }
  return strides;
}

template <typename TData>
Status ConfigureTensor(::musa::dnn::Tensor& tensor,
                       const std::vector<int64_t>& dims,
                       ::musa::dnn::Tensor::Format format,
                       const TData* data,
                       const std::string& op_name) {
  const auto strides = GetContiguousStrides(dims);
  ORT_RETURN_IF_ERROR(ToStatus(
      tensor.SetNdInfo(static_cast<int>(dims.size()), dims.data(), strides.data()), op_name, "SetNdInfo"));
  ORT_RETURN_IF_ERROR(ToStatus(
      tensor.SetType(GetMusaDataType<TData>()), op_name, "SetType"));
  ORT_RETURN_IF_ERROR(ToStatus(
      tensor.SetFormat(format), op_name, "SetFormat"));
  ORT_RETURN_IF_ERROR(ToStatus(
      tensor.SetAddr(data), op_name, "SetAddr"));
  return Status::OK();
}

inline Status ConfigureTensor(::musa::dnn::Tensor& tensor,
                              const std::vector<int64_t>& dims,
                              ::musa::dnn::Tensor::Format format,
                              const int64_t* data,
                              const std::string& op_name) {
  const auto strides = GetContiguousStrides(dims);
  ORT_RETURN_IF_ERROR(ToStatus(
      tensor.SetNdInfo(static_cast<int>(dims.size()), dims.data(), strides.data()), op_name, "SetNdInfo"));
  ORT_RETURN_IF_ERROR(ToStatus(
      tensor.SetType(::musa::dnn::Tensor::Type::INT64), op_name, "SetType"));
  ORT_RETURN_IF_ERROR(ToStatus(
      tensor.SetFormat(format), op_name, "SetFormat"));
  ORT_RETURN_IF_ERROR(ToStatus(
      tensor.SetAddr(data), op_name, "SetAddr"));
  return Status::OK();
}

}  // namespace detail

template <typename PoolType>
class PoolTypeSupport final {
 public:
  static constexpr bool kHasIndices = detail::PoolOpTraits<PoolType>::kHasIndices;
};

template <typename T, typename PoolType, bool NHWC = false>
class Pool : public MusaKernel {
 public:
  explicit Pool(const OpKernelInfo& info)
      : MusaKernel(info),
        op_name_(info.GetKernelDef().OpName()),
        pool_attrs_(info, op_name_, info.node().SinceVersion()) {}

  Status ComputeInternal(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    const TensorShape& x_shape = X->Shape();
    const auto x_dims = x_shape.GetDims();

    if (x_shape.NumDimensions() < 3) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input dimension cannot be less than 3.");
    }

    auto kernel_shape = pool_attrs_.kernel_shape;
    auto strides = pool_attrs_.strides;
    TensorShapeVector pads = pool_attrs_.pads;

    if (pool_attrs_.global_pooling) {
      if constexpr (NHWC) {
        kernel_shape.assign(x_dims.begin() + 1, x_dims.end() - 1);
      } else {
        kernel_shape.assign(x_dims.begin() + 2, x_dims.end());
      }
      pads.assign(2 * kernel_shape.size(), 0);
      strides.assign(kernel_shape.size(), 1);
    }

    auto out_channel = NHWC ? x_shape[x_dims.size() - 1] : x_shape[1];
    auto y_dims = pool_attrs_.SetOutputSize(x_shape, out_channel, &pads, NHWC);
    TensorShape y_shape(y_dims);
    Tensor* Y = context->Output(0, y_shape);

    if (y_shape.Size() == 0) {
      return Status::OK();
    }

    auto x_data = X->Data<T>();
    auto y_data = Y->MutableData<T>();

    int64_t* i_data = nullptr;
    if constexpr (PoolTypeSupport<PoolType>::kHasIndices) {
      Tensor* I = context->Output(1, TensorShape(y_dims));
      i_data = (I != nullptr) ? I->MutableData<int64_t>() : nullptr;

      if (!pool_attrs_.default_dilations && i_data != nullptr) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               op_name_, ": non-default dilations with indices are not yet supported.");
      }
    }

    const auto* ep = static_cast<const MusaExecutionProvider*>(Info().GetExecutionProvider());
    MusaPreparation prepare(ep);
    auto& handle = prepare.GetHandle();

    ORT_RETURN_IF_ERROR(detail::ToStatus(handle.SetStream(Stream(context)), op_name_, "SetStream"));

    ::musa::dnn::Pooling pool_op;
    auto mode = detail::PoolOpTraits<PoolType>::GetMudnnMode(pool_attrs_);
    std::vector<int64_t> x_pool_dims(x_dims.begin(), x_dims.end());
    std::vector<int64_t> y_pool_dims(y_dims.begin(), y_dims.end());

    // muDNN Pooling only supports 4D (2D spatial) inputs natively.
    // For 3D inputs (1D spatial), pad [N,C,L]→[N,C,L,1] to make muDNN happy.
    // This handles both global pooling (GlobalAvgPool/GlobalMaxPool on 1D)
    // and regular 1D pooling (MaxPool/AveragePool with 1D kernel).
    const bool needs_1d_to_2d_pad = (x_dims.size() == 3);

    if (needs_1d_to_2d_pad) {
      if constexpr (NHWC) {
        x_pool_dims.insert(x_pool_dims.end() - 1, 1);
        y_pool_dims.insert(y_pool_dims.end() - 1, 1);
      } else {
        x_pool_dims.push_back(1);
        y_pool_dims.push_back(1);
      }
      kernel_shape.push_back(1);
      strides.push_back(1);
      pads.insert(pads.begin() + pads.size() / 2, 0);
      pads.push_back(0);
      // GlobalAveragePool needs mode change (GLOBAL_AVGPOOL → AVGPOOL with kernel=spatial).
      // All other modes (MAXPOOL, AVGPOOL_COUNT_*) stay as-is.
      if (mode == ::musa::dnn::Pooling::Mode::GLOBAL_AVGPOOL) {
        mode = ::musa::dnn::Pooling::Mode::AVGPOOL_COUNT_WITHOUT_PAD;
      }
    }

    ORT_RETURN_IF_ERROR(detail::ToStatus(
        pool_op.SetMode(mode),
        op_name_, "SetMode"));

    const int64_t spatial_dim_count = static_cast<int64_t>(kernel_shape.size());
    if (spatial_dim_count > 3) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, op_name_, ": supports at most 3D pooling operations.");
    }

    std::vector<int> kernel_int(kernel_shape.begin(), kernel_shape.end());
    std::vector<int> stride_int(strides.begin(), strides.end());
    std::vector<int> pad_int(spatial_dim_count);
    for (int64_t i = 0; i < spatial_dim_count; ++i) {
      pad_int[static_cast<size_t>(i)] = static_cast<int>(pads[static_cast<size_t>(i)]);
    }

    std::vector<int> dilation_int;
    if (pool_attrs_.default_dilations || pool_attrs_.global_pooling) {
      // Global pooling and default dilations: all 1s (spatial_dim_count already
      // accounts for 1D→2D padding if applied).
      dilation_int.assign(static_cast<size_t>(spatial_dim_count), 1);
    } else {
      dilation_int.assign(pool_attrs_.dilations.begin(), pool_attrs_.dilations.end());
      if (needs_1d_to_2d_pad) {
        // Extend dilation with trailing 1 for the padded spatial dimension.
        dilation_int.push_back(1);
      }
    }

    if (mode != ::musa::dnn::Pooling::Mode::GLOBAL_AVGPOOL) {
      ORT_RETURN_IF_ERROR(detail::ToStatus(
          pool_op.SetNdInfo(static_cast<int>(spatial_dim_count), kernel_int.data(), pad_int.data(),
                            stride_int.data(), dilation_int.data()),
          op_name_, "SetNdInfo"));
    }

    const auto format = NHWC ? detail::GetNhwcTensorFormat(x_pool_dims.size())
                             : detail::GetTensorFormat(x_pool_dims.size());
    if (format == ::musa::dnn::Tensor::Format::UNKNOWN) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, op_name_, ": unsupported input rank ", x_pool_dims.size(), ".");
    }

    ::musa::dnn::Tensor input_tensor;
    ::musa::dnn::Tensor output_tensor;
    ::musa::dnn::Tensor indices_tensor;

    ORT_RETURN_IF_ERROR(detail::ConfigureTensor(input_tensor, x_pool_dims, format, x_data, op_name_));
    ORT_RETURN_IF_ERROR(detail::ConfigureTensor(output_tensor, y_pool_dims, format, y_data, op_name_));

    if constexpr (PoolTypeSupport<PoolType>::kHasIndices) {
      if (i_data != nullptr) {
        ORT_RETURN_IF_ERROR(detail::ConfigureTensor(indices_tensor, y_pool_dims, format, i_data, op_name_));
      }
    }

    ORT_RETURN_IF_ERROR(detail::ToStatus(
        pool_op.Run(handle, output_tensor, input_tensor, indices_tensor),
        op_name_, "Run"));

    return Status::OK();
  }

 private:
  const std::string op_name_;
  PoolAttributes pool_attrs_;
};

}  // namespace musa
}  // namespace onnxruntime
