// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/nn/conv.h"
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/common.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/musa_execution_provider.h"
#include "core/providers/musa/tensor/transpose.h"
#include <algorithm>
#include <musa_runtime.h>
#include <mudnn.h>
#include <string>
#include <vector>

using onnxruntime::common::Status;
namespace onnxruntime {
namespace musa {

using ConvPadVector = ConvAttributes::ConvPadVector;

template <typename T, bool NHWC>
Status Conv<T, NHWC>::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                              bool& is_packed, PrePackedWeights* /*prepacked_weights*/) {
  is_packed = false;
  
  if constexpr (NHWC) {
    if (input_idx == 1) {
      if (is_nhwc_domain_) {
        // Transpose from {M, C/group, kH, kW} to {M, kH, kW, C/group}
        auto orig_shape = tensor.Shape();
        auto shape_dims = orig_shape.GetDims();
        auto rank = shape_dims.size();
        
        ORT_ENFORCE(rank >= 3, "Weight tensor must have at least 3 dimensions for NHWC");

        auto perm = GetNHWCWeightPerm(rank);
        
        // Calculate NHWC shape
        std::vector<int64_t> nhwc_dims;
        for (size_t i = 0; i < rank; i++) {
          nhwc_dims.push_back(shape_dims[perm[i]]);
        }

        W_ = Tensor::Create(tensor.DataType(), TensorShape(nhwc_dims), std::move(alloc));
        // Use muDNN Permute for conversion
        auto status = DoMusaTranspose(
            tensor.DataRaw(), 
            W_->MutableDataRaw(),
            std::vector<int64_t>(shape_dims.begin(), shape_dims.end()),
            perm,
            GetMusaDataType<T>(),
            nullptr,  // use default stream
            Info().GetExecutionProvider()->GetDeviceId());
        
        if (!status.IsOK()) {
          return status;
        }
        
        musaStreamSynchronize(nullptr);
        is_packed = true;
      } else {
        // NHWC template but not NHWC domain: weight is already in NHWC format
        W_already_nhwc = true;
      }
    }
  } else {
    ORT_UNUSED_PARAMETER(tensor);
    ORT_UNUSED_PARAMETER(input_idx);
    ORT_UNUSED_PARAMETER(alloc);
  }

  return Status::OK();
}

template <typename T>
Status SimpleMusaConvOp(const MusaKernel* kernel,
                        const MusaPreparation& prepare,
                        onnxruntime::Stream* ort_stream,
                        const TensorShapeVector& kernel_shape,
                        const ConvPadVector& pads,
                        const TensorShapeVector& strides,
                        const TensorShapeVector& dilations,
                        int64_t group) {
  // Get tensor data from prepared MUSA tensors
  const T* input_data = reinterpret_cast<const T*>(prepare.input_a_ptr);
  const T* filter_data = reinterpret_cast<const T*>(prepare.input_b_ptr);
  T* output_data = reinterpret_cast<T*>(prepare.output_ptr);

  // Validate prepared tensors
  if (!input_data || !filter_data || !output_data) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid prepared tensor pointers");
  }

  if (prepare.inputTensors.empty() || prepare.inputTensors.size() < 2 || prepare.outputTensors.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid prepared tensor configuration");
  }

  // Use mudnn Convolution class for device computation
  try {
    ::musa::dnn::Convolution conv_op;

    auto status = conv_op.SetGroups(static_cast<int>(group));
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set mudnn Convolution groups");
    }

    // Set convolution parameters (pad, stride, dilation)
    // Convert to int arrays for mudnn
    std::vector<int> pad_ints;
    std::vector<int> stride_ints;
    std::vector<int> dilation_ints;

    size_t spatial_dims = kernel_shape.size();

    // ONNX pads format:
    // 1D: [pad_left, pad_right]
    // 2D: [pad_top, pad_left, pad_bottom, pad_right]
    // 3D: [pad_front, pad_top, pad_left, pad_back, pad_bottom, pad_right]
    // mudnn SetNdInfo expects symmetric padding for each dimension
    if (kernel_shape.size() == 1) {
      // For 1D convolution, ONNX pads are [pad_left, pad_right]
      // mudnn expects symmetric padding [pad], use the first value
      pad_ints = {static_cast<int>(pads[0])};
      stride_ints = {static_cast<int>(strides[0])};
      dilation_ints = {static_cast<int>(dilations[0])};
    } else if (kernel_shape.size() == 2) {
      // For 2D convolution, ONNX pads are [pad_top, pad_left, pad_bottom, pad_right]
      // mudnn expects symmetric padding [pad_h, pad_w], use the first pair
      pad_ints = {static_cast<int>(pads[0]), static_cast<int>(pads[1])};
      stride_ints = {static_cast<int>(strides[0]), static_cast<int>(strides[1])};
      dilation_ints = {static_cast<int>(dilations[0]), static_cast<int>(dilations[1])};
    } else if (kernel_shape.size() == 3) {
      // For 3D convolution, ONNX pads are [pad_front, pad_top, pad_left, pad_back, pad_bottom, pad_right]
      // mudnn expects symmetric padding [pad_d, pad_h, pad_w], use the first triplet
      pad_ints = {static_cast<int>(pads[0]), static_cast<int>(pads[1]), static_cast<int>(pads[2])};
      stride_ints = {static_cast<int>(strides[0]), static_cast<int>(strides[1]), static_cast<int>(strides[2])};
      dilation_ints = {static_cast<int>(dilations[0]), static_cast<int>(dilations[1]), static_cast<int>(dilations[2])};
    } else {
      // For other dimensions, convert all
      for (size_t i = 0; i < kernel_shape.size(); ++i) {
        pad_ints.push_back(static_cast<int>(pads[i]));
        stride_ints.push_back(static_cast<int>(strides[i]));
        dilation_ints.push_back(static_cast<int>(dilations[i]));
      }
    }

    status = conv_op.SetNdInfo(static_cast<int>(spatial_dims),
                               pad_ints.data(),
                               stride_ints.data(),
                               dilation_ints.data());
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set mudnn Convolution parameters");
    }

    status = conv_op.SetComputeMode(::musa::dnn::Convolution::ComputeMode::TENSOR);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to set mudnn Convolution compute mode");
    }

    bool has_bias = (prepare.inputTensors.size() >= 3);

    // Create workspace maintainer using ORT's stream-aware memory allocation
    std::vector<IAllocatorUniquePtr<void>> workspace_buffers_holder;
    auto memory_allocator = [kernel, ort_stream, &workspace_buffers_holder](size_t size) -> ::musa::dnn::MemoryHandler {
      if (size == 0) {
        return ::musa::dnn::MemoryHandler(nullptr, [](void*) {});
      }
      auto scratch = kernel->GetScratchBuffer<void>(size, ort_stream);
      void* ptr = scratch.get();
      workspace_buffers_holder.push_back(std::move(scratch));
      return ::musa::dnn::MemoryHandler(ptr, [](void*) {});
    };
    ::musa::dnn::MemoryMaintainer maintainer = memory_allocator;

    // Run convolution
    bool is_3d_conv = (kernel_shape.size() == 3);
    bool is_2d_conv = (kernel_shape.size() == 2);

    if (has_bias && is_2d_conv) {
      // 2D convolution with bias: use RunFusion
      ::musa::dnn::Convolution::FusedActivationDesc act_desc;
      act_desc.SetMode(::musa::dnn::Convolution::FusedActivationDesc::Mode::IDENTITY);

      ::musa::dnn::Tensor empty_add_tensor;
      status = conv_op.RunFusion(prepare.GetHandle(),
                                const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]),
                                prepare.inputTensors[0],
                                prepare.inputTensors[1],
                                prepare.inputTensors[2],
                                empty_add_tensor,
                                act_desc,
                                ::musa::dnn::Convolution::Algorithm::IMPLICIT_GEMM,
                                maintainer);
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                "mudnn Convolution operation failed, status: " +
                                std::to_string(static_cast<int>(status)));
      }     
    } else {
      // Other cases (1D/3D convolution or no bias): use Run
      status = conv_op.Run(prepare.GetHandle(),
                          const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]),
                          prepare.inputTensors[0],
                          prepare.inputTensors[1],
                          ::musa::dnn::Convolution::Algorithm::IMPLICIT_GEMM,
                          maintainer);
      if (status != ::musa::dnn::Status::SUCCESS) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                "mudnn Convolution operation failed, status: " +
                                    std::to_string(static_cast<int>(status)));
      }

      // If bias exists and this is 3D conv, add it manually
      if (has_bias && is_3d_conv) {
          ::musa::dnn::Binary bias_add_op;
          status = bias_add_op.SetMode(::musa::dnn::Binary::Mode::ADD);
          if (status != ::musa::dnn::Status::SUCCESS) {
              return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                    "Failed to set mudnn Binary mode for bias addition");
          }

          // Create reshaped bias tensor for broadcasting: [1, C, 1, 1, 1]
          ::musa::dnn::Tensor reshaped_bias_tensor;
          auto bias_data_type = GetMusaDataType<T>();

          status = reshaped_bias_tensor.SetType(bias_data_type);
          ORT_RETURN_IF_NOT(status == ::musa::dnn::Status::SUCCESS, "Failed to set reshaped bias tensor type");

          status = reshaped_bias_tensor.SetAddr(prepare.bias_ptr);
          ORT_RETURN_IF_NOT(status == ::musa::dnn::Status::SUCCESS, "Failed to set reshaped bias tensor address");

          status = reshaped_bias_tensor.SetFormat(::musa::dnn::Tensor::Format::NCDHW);
          ORT_RETURN_IF_NOT(status == ::musa::dnn::Status::SUCCESS, "Failed to set reshaped bias tensor format");

          auto output_shape = prepare.output_shape.AsShapeVector();
          std::vector<int64_t> bias_broadcast_shape = {1, output_shape[1], 1, 1, 1};
          status = reshaped_bias_tensor.SetNdInfo(static_cast<int>(bias_broadcast_shape.size()),
                                                  bias_broadcast_shape.data());
          ORT_RETURN_IF_NOT(status == ::musa::dnn::Status::SUCCESS, "Failed to set reshaped bias tensor shape");

          // Binary ADD: output = output + bias (broadcast)
          status = bias_add_op.Run(prepare.GetHandle(),
                                  const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]),
                                  prepare.outputTensors[0],
                                  reshaped_bias_tensor);
          if (status != ::musa::dnn::Status::SUCCESS) {
              return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                    "mudnn Binary bias addition failed, status: " +
                                        std::to_string(static_cast<int>(status)));
          }
      }
    }

  } catch (const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Exception in mudnn Convolution operation: " +
                               std::string(e.what()));
  }

  return Status::OK();
}

template <typename T, bool NHWC>
Status Conv<T, NHWC>::ComputeInternal(OpKernelContext* ctx) const {
  // Debug: Log which template instance is being used
  LOGS_DEFAULT(INFO) << "MUSA Conv::ComputeInternal - NHWC template = " << NHWC
                     << ", is_nhwc_domain_ = " << is_nhwc_domain_
                     << ", node domain = " << this->Node().Domain();
  
  const auto* X = ctx->Input<Tensor>(0);
  if (X == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input tensor X");
  }

  constexpr bool channels_last = NHWC;
  
  // Determine if weights are in NHWC format
  // Note: If W_ is not null, weight was prepacked and removed from graph inputs
  const Tensor* W = nullptr;
  const Tensor* B = nullptr;
  bool w_in_nhwc = false;
  
  if (W_) {
    W = W_.get();
    w_in_nhwc = true;
  } else {
    W = ctx->Input<Tensor>(1);
    if (W == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input tensor W");
    }
    w_in_nhwc = W_already_nhwc;
  }
  B = ctx->Input<Tensor>(2);


  // Validate input shape with layout info
  ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(X->Shape(), W->Shape(), channels_last, w_in_nhwc));

  TensorShapeVector kernel_shape;
  ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(W->Shape(), kernel_shape, w_in_nhwc));

  // Check if this is a 1D convolution (3D tensors)
  bool is_1d_conv = (X->Shape().NumDimensions() == 3 && W->Shape().NumDimensions() == 3);

  // For 1D convolution, convert to 2D
  if (is_1d_conv) {
    return ComputeConv1D(ctx, X, W, B, channels_last, w_in_nhwc);
  }

  ConvPadVector pads(conv_attrs_.pads);
  if (pads.empty()) {
    pads.resize(kernel_shape.size() * 2, 0);
  }
  TensorShapeVector dilations(conv_attrs_.dilations);
  if (dilations.empty()) {
    dilations.resize(kernel_shape.size(), 1);
  }
  TensorShapeVector strides(conv_attrs_.strides);
  if (strides.empty()) {
    strides.resize(kernel_shape.size(), 1);
  }

  // Compute output dimensions
  const int64_t N = X->Shape()[0];
  const int64_t M = W->Shape()[0];
  TensorShapeVector Y_dims;
  
  if constexpr (channels_last) {
    // NHWC: [N, H, W, ..., C]
    Y_dims.push_back(N);
  } else {
    // NCHW: [N, C, H, W, ...]
    Y_dims.push_back(N);
    Y_dims.push_back(M);
  }

  // Get spatial dimensions for output shape calculation
  size_t spatial_dim_start = channels_last ? 1 : 2;
  size_t spatial_dim_end = spatial_dim_start + kernel_shape.size();
  TensorShape spatial_shape = X->Shape().Slice(spatial_dim_start, spatial_dim_end);

  ORT_RETURN_IF_ERROR(conv_attrs_.InferPadsAndOutputShape(spatial_shape, kernel_shape,
                                                          strides, dilations, pads, Y_dims));
  
  if constexpr (channels_last) {
    // NHWC: append C at the end
    Y_dims.push_back(M);
  }

  Tensor* Y = ctx->Output(0, TensorShape(Y_dims));
  if (Y == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor");
  }

  /* Bail out early if the output is going to be empty */
  if (Y->Shape().Size() == 0) return Status::OK();

  /* Prepare MUSA operation */
  const auto* ep = static_cast<const MusaExecutionProvider*>(
      Info().GetExecutionProvider());
  MusaPreparation prepare(ep);

  // Store tensor pointers and shapes
  prepare.input_a_ptr = X->DataRaw();
  prepare.input_b_ptr = W->DataRaw();
  prepare.bias_ptr = B ? B->DataRaw() : nullptr;
  prepare.output_ptr = Y->MutableDataRaw();
  prepare.output_size = Y->Shape().Size();
  prepare.input_a_shape = X->Shape();
  prepare.input_b_shape = W->Shape();
  prepare.output_shape = Y->Shape();

  if (prepare.input_a_ptr == nullptr || prepare.input_b_ptr == nullptr || prepare.output_ptr == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid tensor data pointers");
  }

  // Get MUSA data type
  const auto musaType = GetMusaDataType<T>();

  // Prepare MUSA tensors
  ORT_TRY {
    // Set up MUSA stream for asynchronous execution
    auto* stream = Stream(ctx);
    if (prepare.handle) {
      if (stream) {
        auto status = prepare.handle->SetStream(stream);
        if (status != ::musa::dnn::Status::SUCCESS) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                 "Failed to set MUSA stream, status: " +
                                     std::to_string(static_cast<int>(status)));
        }
      }
    }

    // Initialize tensors vectors - account for optional bias
    size_t num_inputs = (B != nullptr) ? 3 : 2;
    prepare.inputTensors.resize(num_inputs);
    prepare.outputTensors.resize(1);

    // Setup input tensor X with appropriate format
    {
      auto& tensor = prepare.inputTensors[0];
      auto status = tensor.SetType(musaType);
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set input tensor type");
      }
      
      status = tensor.SetAddr(X->DataRaw());
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set input tensor address");
      }
      
      // Set format based on layout
      auto ndims = X->Shape().NumDimensions();
      ::musa::dnn::Tensor::Format format;

      if (ndims == 3 || ndims == 4) { // 1d/2d conv
        format = channels_last ? ::musa::dnn::Tensor::Format::NHWC
                              : ::musa::dnn::Tensor::Format::NCHW;
      } else if (ndims == 5) { // 3d conv
        format = channels_last ? ::musa::dnn::Tensor::Format::NDHWC
                              : ::musa::dnn::Tensor::Format::NCDHW;
      } else {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                              "Unsupported input dimension: " + std::to_string(ndims));
      }

      status = tensor.SetFormat(format);
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set input tensor format");
      }
      
      auto x_dims = X->Shape().AsShapeVector();
      status = tensor.SetNdInfo(static_cast<int>(x_dims.size()), x_dims.data());
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set input tensor shape");
      }
    }

    // Setup filter tensor W with appropriate format
    {
      auto& tensor = prepare.inputTensors[1];
      auto status = tensor.SetType(musaType);
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set weight tensor type");
      }
      
      status = tensor.SetAddr(W->DataRaw());
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set weight tensor address");
      }
      
      // Set format based on layout
      auto ndims = W->Shape().NumDimensions();
      ::musa::dnn::Tensor::Format format;

      if (ndims == 3 || ndims == 4) { // 1d/2d conv
        format = channels_last ? ::musa::dnn::Tensor::Format::NHWC
                              : ::musa::dnn::Tensor::Format::NCHW;
      } else if (ndims == 5) { // 3d conv
        format = channels_last ? ::musa::dnn::Tensor::Format::NDHWC
                              : ::musa::dnn::Tensor::Format::NCDHW;
      } else {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                              "Unsupported input dimension: " + std::to_string(ndims));
      }

      status = tensor.SetFormat(format);
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set weight tensor format");
      }
      
      auto w_dims = W->Shape().AsShapeVector();
      status = tensor.SetNdInfo(static_cast<int>(w_dims.size()), w_dims.data());
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set weight tensor shape");
      }
    }

    // Setup optional bias tensor B
    if (B != nullptr) {
      auto& tensor = prepare.inputTensors[2];
      auto status = tensor.SetType(musaType);
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set bias tensor type");
      }
      
      status = tensor.SetAddr(B->DataRaw());
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set bias tensor address");
      }
      
      // Bias is always 1D [C], use NCW format
      status = tensor.SetFormat(::musa::dnn::Tensor::Format::NCW);
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set bias tensor format");
      }
      
      auto b_dims = B->Shape().AsShapeVector();
      status = tensor.SetNdInfo(static_cast<int>(b_dims.size()), b_dims.data());
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set bias tensor shape");
      }
    }

    // Setup output tensor Y with appropriate format
    {
      auto& tensor = prepare.outputTensors[0];
      auto status = tensor.SetType(musaType);
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set output tensor type");
      }
      
      status = tensor.SetAddr(Y->MutableDataRaw());
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set output tensor address");
      }
      
      // Set format based on layout
      auto ndims = Y->Shape().NumDimensions();
      ::musa::dnn::Tensor::Format format;

      if (ndims == 3 || ndims == 4) { // 1d/2d conv
        format = channels_last ? ::musa::dnn::Tensor::Format::NHWC
                              : ::musa::dnn::Tensor::Format::NCHW;
      } else if (ndims == 5) { // 3d conv
        format = channels_last ? ::musa::dnn::Tensor::Format::NDHWC
                              : ::musa::dnn::Tensor::Format::NCDHW;
      } else {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                              "Unsupported input dimension: " + std::to_string(ndims));
      }

      status = tensor.SetFormat(format);
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set output tensor format");
      }
      
      auto y_dims = Y->Shape().AsShapeVector();
      status = tensor.SetNdInfo(static_cast<int>(y_dims.size()), y_dims.data());
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set output tensor shape");
      }
    }
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  /* Call MUSA device Conv operation using prepared data */
  ORT_RETURN_IF_ERROR(
      SimpleMusaConvOp<T>(this, prepare, ctx->GetComputeStream(), kernel_shape, pads, strides, dilations, 
                              conv_attrs_.group));

  return Status::OK();
}

template <typename T, bool NHWC>
Status Conv<T, NHWC>::ComputeConv1D(OpKernelContext* ctx, const Tensor* X, const Tensor* W, const Tensor* B,
                                    bool x_is_nhwc, bool w_is_nhwc) const {
  // Convert 1D convolution to 2D by adding height=1 dimension
  const auto& X_shape = X->Shape();
  const auto& W_shape = W->Shape();

  // Create 4D shapes by inserting dimension at position 2 (height=1)
  // NCHW: [N, C, L] -> [N, C, 1, L]
  // NHWC: [N, L, C] -> [N, 1, L, C]
  TensorShapeVector X_4d_shape;
  TensorShapeVector W_4d_shape;
  
  if (x_is_nhwc) {
    X_4d_shape = {X_shape[0], 1, X_shape[1], X_shape[2]};  // [N, 1, L, C]
    W_4d_shape = {W_shape[0], 1, W_shape[1], W_shape[2]};  // [M, 1, K, C/g]
  } else {
    X_4d_shape = {X_shape[0], X_shape[1], 1, X_shape[2]};  // [N, C, 1, L]
    W_4d_shape = {W_shape[0], W_shape[1], 1, W_shape[2]};  // [M, C/g, 1, K]
  }

  // Compute kernel shape using original 3D weight shape
  TensorShapeVector kernel_shape_1d;
  ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(W->Shape(), kernel_shape_1d, w_is_nhwc));
  TensorShapeVector kernel_shape_2d = {1, kernel_shape_1d[0]};

  // Convert 1D convolution parameters to 2D
  ConvPadVector pads_2d(conv_attrs_.pads);
  if (pads_2d.empty()) {
    pads_2d.resize(2, 0);
  }
  if (pads_2d.size() == 2) {
    pads_2d = {0, pads_2d[0], 0, pads_2d[1]};
  }

  TensorShapeVector strides_2d(conv_attrs_.strides);
  if (strides_2d.empty()) {
    strides_2d.resize(2, 1);
  }
  if (strides_2d.size() == 1) {
    strides_2d = {1, strides_2d[0]};
  }

  TensorShapeVector dilations_2d(conv_attrs_.dilations);
  if (dilations_2d.empty()) {
    dilations_2d.resize(2, 1);
  }
  if (dilations_2d.size() == 1) {
    dilations_2d = {1, dilations_2d[0]};
  }

  // Compute output shape for 2D convolution
  const int64_t N = X_4d_shape[0];
  const int64_t M = W_4d_shape[0];
  TensorShapeVector Y_4d_dims({N, M});
  TensorShape input_shape_2d = TensorShape(X_4d_shape).Slice(2);
  ORT_RETURN_IF_ERROR(conv_attrs_.InferPadsAndOutputShape(
      input_shape_2d, kernel_shape_2d, strides_2d, dilations_2d, pads_2d, Y_4d_dims));

  // Extract the 1D output length from the 4D shape
  TensorShapeVector Y_1d_dims;
  if (x_is_nhwc) {
    Y_1d_dims = {Y_4d_dims[0], Y_4d_dims[2], Y_4d_dims[3]};  // [N, L_out, M]
  } else {
    Y_1d_dims = {Y_4d_dims[0], Y_4d_dims[1], Y_4d_dims[3]};  // [N, M, L_out]
  }

  // Create 3D output tensor
  Tensor* Y_1d = ctx->Output(0, TensorShape(Y_1d_dims));
  if (Y_1d == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor");
  }

  if (Y_1d->Shape().Size() == 0) return Status::OK();

  /* Prepare MUSA operation with 4D tensors */
  const auto* ep = static_cast<const MusaExecutionProvider*>(
      Info().GetExecutionProvider());
  MusaPreparation prepare(ep);

  prepare.input_a_ptr = X->DataRaw();
  prepare.input_b_ptr = W->DataRaw();
  prepare.bias_ptr = B ? B->DataRaw() : nullptr;
  prepare.output_ptr = Y_1d->MutableDataRaw();
  prepare.output_size = Y_1d->Shape().Size();
  prepare.input_a_shape = TensorShape(X_4d_shape);
  prepare.input_b_shape = TensorShape(W_4d_shape);
  prepare.output_shape = TensorShape(Y_4d_dims);

  if (prepare.input_a_ptr == nullptr || prepare.input_b_ptr == nullptr || prepare.output_ptr == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid tensor data pointers");
  }

  const auto musaType = GetMusaDataType<T>();

  ORT_TRY {
    auto* stream = Stream(ctx);
    if (prepare.handle) {
      if (stream) {
        prepare.handle->SetStream(stream);
      }
    }

    size_t num_inputs = (B != nullptr) ? 3 : 2;
    prepare.inputTensors.resize(num_inputs);
    prepare.outputTensors.resize(1);

    // Setup input tensor X as 4D
    {
      auto& tensor = prepare.inputTensors[0];
      auto status = tensor.SetType(musaType);
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set input tensor type");
      }
      status = tensor.SetAddr(X->DataRaw());
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set input tensor address");
      }
      if (x_is_nhwc) {
        status = tensor.SetFormat(::musa::dnn::Tensor::Format::NHWC);
      } else {
        status = tensor.SetFormat(::musa::dnn::Tensor::Format::NCHW);
      }
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set input tensor format");
      }
      status = tensor.SetNdInfo(static_cast<int>(X_4d_shape.size()), X_4d_shape.data());
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set input tensor shape");
      }
    }

    // Setup filter tensor W as 4D
    {
      auto& tensor = prepare.inputTensors[1];
      auto status = tensor.SetType(musaType);
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set weight tensor type");
      }
      status = tensor.SetAddr(W->DataRaw());
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set weight tensor address");
      }
      if (w_is_nhwc) {
        status = tensor.SetFormat(::musa::dnn::Tensor::Format::NHWC);
      } else {
        status = tensor.SetFormat(::musa::dnn::Tensor::Format::NCHW);
      }
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set weight tensor format");
      }
      status = tensor.SetNdInfo(static_cast<int>(W_4d_shape.size()), W_4d_shape.data());
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set weight tensor shape");
      }
    }

    // Setup bias
    if (B != nullptr) {
      auto& tensor = prepare.inputTensors[2];
      auto status = tensor.SetType(musaType);
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set bias tensor type");
      }
      status = tensor.SetAddr(B->DataRaw());
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set bias tensor address");
      }
      status = tensor.SetFormat(::musa::dnn::Tensor::Format::NCW);
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set bias tensor format");
      }
      auto b_dims = B->Shape().AsShapeVector();
      status = tensor.SetNdInfo(static_cast<int>(b_dims.size()), b_dims.data());
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set bias tensor shape");
      }
    }

    // Setup output tensor Y as 4D
    {
      auto& tensor = prepare.outputTensors[0];
      auto status = tensor.SetType(musaType);
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set output tensor type");
      }
      status = tensor.SetAddr(Y_1d->MutableDataRaw());
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set output tensor address");
      }
      if (x_is_nhwc) {
        status = tensor.SetFormat(::musa::dnn::Tensor::Format::NHWC);
      } else {
        status = tensor.SetFormat(::musa::dnn::Tensor::Format::NCHW);
      }
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set output tensor format");
      }
      status = tensor.SetNdInfo(static_cast<int>(Y_4d_dims.size()), Y_4d_dims.data());
      if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set output tensor shape");
      }
    }
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  ORT_RETURN_IF_ERROR(
      SimpleMusaConvOp<T>(this, prepare, ctx->GetComputeStream(), kernel_shape_2d, pads_2d, strides_2d, 
                              dilations_2d, conv_attrs_.group));

  return Status::OK();
}

// Macro for registering typed kernel
#define REGISTER_MUSA_CONV_TYPED_KERNEL(ver, T, NHWC)                     \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                  \
      Conv, kOnnxDomain, 1, 10, T, kMusaExecutionProvider,                  \
      (*KernelDefBuilder::Create())                                         \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),           \
      Conv<T, NHWC>);                                                       \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                            \
      Conv, kOnnxDomain, ver, T, kMusaExecutionProvider,                    \
      (*KernelDefBuilder::Create())                                         \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),           \
      Conv<T, NHWC>);

// Register NCHW operations for version 11
REGISTER_MUSA_CONV_TYPED_KERNEL(11, float, false)
REGISTER_MUSA_CONV_TYPED_KERNEL(11, MLFloat16, false)

// Register NHWC operations (if enabled)
#ifdef ENABLE_MUSA_NHWC_OPS
// NHWC domain kernels
ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(
    Conv, kMSInternalNHWCDomain, 1, 10, float, kMusaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Conv<float, true>);
ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(
    Conv, kMSInternalNHWCDomain, 1, 10, MLFloat16, kMusaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    Conv<MLFloat16, true>);
ONNX_OPERATOR_TYPED_KERNEL_EX(
    Conv, kMSInternalNHWCDomain, 11, float, kMusaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Conv<float, true>);
ONNX_OPERATOR_TYPED_KERNEL_EX(
    Conv, kMSInternalNHWCDomain, 11, MLFloat16, kMusaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    Conv<MLFloat16, true>);
#endif

// Template instantiation for NCHW
#ifndef DISABLE_CONTRIB_OPS
template class Conv<float, false>;
#ifdef ENABLE_MUSA_NHWC_OPS
template class Conv<float, true>;
template class Conv<MLFloat16, true>;
#endif
#endif

}  // namespace musa
}  // namespace onnxruntime
