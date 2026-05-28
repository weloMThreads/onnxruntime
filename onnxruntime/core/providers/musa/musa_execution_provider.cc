// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include "core/common/parse_string.h"
#include "core/providers/musa/musa_execution_provider.h"
#include "contrib_ops/musa/musa_contrib_kernels.h"
#include "core/providers/musa/musa_data_transfer.h"
#include "core/providers/musa/musa_allocator.h"
#include "core/providers/musa/musa_stream_handle.h"
#include "core/providers/musa/musa_call.h"
#include "core/session/onnxruntime_run_options_config_keys.h"
#include "musa_fwd.h"
#include <mudnn_version.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_set>
#include <vector>

namespace onnxruntime {

// Memcpy kernel implementation for MUSA EP
class Memcpy final : public OpKernel {
 public:
  explicit Memcpy(const OpKernelInfo& info) : OpKernel{info} {}

  Status Compute(OpKernelContext* ctx) const override {
    auto X_type = ctx->InputType(0);
    if (X_type->IsTensorType()) {
      const auto* X = ctx->Input<Tensor>(0);
      ORT_ENFORCE(X != nullptr, "Memcpy: Input tensor is nullptr.");
      Tensor* Y = ctx->Output(0, X->Shape());
      ORT_ENFORCE(Y != nullptr, "Memcpy: Failed to allocate output tensor.");

      // Use DataTransferManager to handle device-aware copying
      auto* data_transfer = Info().GetDataTransferManager().GetDataTransfer(
          X->Location().device, Y->Location().device);

      if (!data_transfer) {
        return Status(common::ONNXRUNTIME, common::FAIL, "Memcpy: No data transfer found for device pair.");
      }

      // Check if we have a compute stream available for async copy
      auto* stream = ctx->GetComputeStream();
      if (stream) {
        // Use async copy when stream is available
        return data_transfer->CopyTensorAsync(*X, *Y, *stream);
      } else {
        // Fall back to sync copy when no stream is available
        return data_transfer->CopyTensor(*X, *Y);
      }
    } else {
      return Status(common::ONNXRUNTIME, common::FAIL, "Memcpy: Unsupported input type.");
    }
  }
};

namespace contrib {
namespace musa {
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSDomain, 1, float, GridSample);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSDomain, 1, MLFloat16, GridSample);
#ifdef ENABLE_MUSA_NHWC_OPS
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 1, float, GridSample);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 1, MLFloat16, GridSample);
#endif
}  // namespace musa
}  // namespace contrib

namespace musa {

// Memcpy operations
ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorAndSequenceTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kMusaExecutionProvider,
    (*KernelDefBuilder::Create())
        .OutputMemoryType(OrtMemTypeCPUOutput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorAndSequenceTensorTypes()),
    Memcpy);

// Memcpy operations (op 1)
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, MemcpyToHost);

// MUSA provider fusion kernels
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSDomain, 1, MusaTokenMixerResidual);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSDomain, 1, MusaReshapeMatMul);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSDomain, 1, MusaFeatureNorm);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSDomain, 1, MusaGelu);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSDomain, 1, MusaPlnCascadeBlock);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSDomain, 1, MusaLayerNormLastDim);

// Conv operations (op 1-10 and op 11) - NCHW
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, float, Conv);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, Conv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, float, Conv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, MLFloat16, Conv);

// ConvTranspose operations (op 1-10 and op 11) - NCHW
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, float,
                                                      ConvTranspose);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16,
                                                      ConvTranspose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, float, ConvTranspose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, MLFloat16, ConvTranspose);

// Conv operations - NHWC (if enabled)
#ifdef ENABLE_MUSA_NHWC_OPS
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 1, 10, float, Conv);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 1, 10, MLFloat16, Conv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 11, float, Conv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 11, MLFloat16, Conv);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 1, 10, float,
                                                      ConvTranspose);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 1, 10,
                                                      MLFloat16, ConvTranspose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 11, float, ConvTranspose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 11, MLFloat16,
                                            ConvTranspose);
#endif

// InstanceNormalization operations (op 1-5, 6)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 5, MLFloat16, InstanceNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 5, float, InstanceNormalization);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, MLFloat16, InstanceNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, float, InstanceNormalization);

// BatchNormalization operations (op 15+ inference)
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 15, MLFloat16, BatchNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 15, float, BatchNormalization);

// LSTM operations (op 7-13, 14)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 13, float, LSTM);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 13, MLFloat16, LSTM);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 13, int32_t, LSTM);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 13, int64_t, LSTM);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 14, float, LSTM);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 14, MLFloat16, LSTM);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 14, int32_t, LSTM);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 14, int64_t, LSTM);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 7, MLFloat16, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 8, 9, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 8, 9, MLFloat16, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 10, 10, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 10, 10, MLFloat16, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 11, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 11, MLFloat16, MaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 12, float, MaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 12, MLFloat16, MaxPool);

// AveragePool operations (op 7-9, 10, and 11)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 9, float, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 9, MLFloat16, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 10, 10, float, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 10, 10, MLFloat16, AveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, float, AveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, MLFloat16, AveragePool);

// GlobalMaxPool operations (op 1)
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, float, GlobalMaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, GlobalMaxPool);

// GlobalAveragePool operations (op 1)
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, float, GlobalAveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, GlobalAveragePool);

#ifdef ENABLE_MUSA_NHWC_OPS
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 1, 7, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 1, 7, MLFloat16, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 8, 9, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 8, 9, MLFloat16, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 10, 10, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 10, 10, MLFloat16, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 11, 11, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 11, 11, MLFloat16, MaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 12, float, MaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 12, MLFloat16, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 7, 9, float, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 7, 9, MLFloat16, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 10, 10, float, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 10, 10, MLFloat16, AveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 11, float, AveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 11, MLFloat16, AveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 1, float, GlobalMaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 1, MLFloat16, GlobalMaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 1, float, GlobalAveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 1, MLFloat16, GlobalAveragePool);
#endif

// Relu operations (op 6-12, 13, and 14)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Relu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, float, Relu);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, Relu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 13, float, Relu);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 14, MLFloat16, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 14, float, Relu);

#ifdef ENABLE_MUSA_NHWC_OPS
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 6, 12, MLFloat16, Relu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 6, 12, float, Relu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 13, 13, MLFloat16, Relu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 13, 13, float, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 14, MLFloat16, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 14, float, Relu);
#endif

// Tanh operations (op 6-12, 13)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Tanh);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, float, Tanh);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Tanh);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, Tanh);

// Sigmoid operations (op 6-12, 13)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Sigmoid);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, float, Sigmoid);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Sigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, Sigmoid);

#ifdef ENABLE_MUSA_NHWC_OPS
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 6, 12, MLFloat16, Sigmoid);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 6, 12, float, Sigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 13, MLFloat16, Sigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 13, float, Sigmoid);
#endif

// LeakyRelu operations (op 6-15, 16)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 15, MLFloat16, LeakyRelu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 15, float, LeakyRelu);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 16, MLFloat16, LeakyRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 16, float, LeakyRelu);

// Log operations (op 6-12, 13)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Log);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, float, Log);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Log);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, Log);

// Softplus operations (op 1+)
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, Softplus);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, float, Softplus);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, BFloat16, Softplus);

// TensorFlow-converted Log1p compatibility op (op 1+)
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, Log1p);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, float, Log1p);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, double, Log1p);

// TensorFlow-converted Expm1 compatibility op (op 1+)
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, Expm1);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, float, Expm1);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, double, Expm1);

// TensorFlow-converted Square/Rsqrt compatibility ops (op 1+)
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, Square);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, float, Square);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, double, Square);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int32_t, Square);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int64_t, Square);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, Rsqrt);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, float, Rsqrt);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, double, Rsqrt);

// TensorFlow-converted ZerosLike compatibility op (op 1+)
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, bool, ZerosLike);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, float, ZerosLike);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, double, ZerosLike);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int32_t, ZerosLike);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int64_t, ZerosLike);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, ZerosLike);

// Exp operations (op 6-12, 13)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Exp);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, float, Exp);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Exp);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, Exp);

// Floor operations (op 6-12, 13)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Floor);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, float, Floor);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, double, Floor);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Floor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, Floor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, double, Floor);

// Ceil operations (op 6-12, 13)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Ceil);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, float, Ceil);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, double, Ceil);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Ceil);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, Ceil);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, double, Ceil);

// Reciprocal operations (op 6-12, 13)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Reciprocal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, float, Reciprocal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, double, Reciprocal);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Reciprocal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, Reciprocal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, double, Reciprocal);

// Sqrt operations (op 6-12, 13)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Sqrt);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, float, Sqrt);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, double, Sqrt);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Sqrt);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, Sqrt);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, double, Sqrt);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, BFloat16, Sqrt);

// IsNaN operations (op 9-12, 13-19, 20)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, IsNaN);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, float, IsNaN);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, double, IsNaN);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 19, MLFloat16, IsNaN);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 19, float, IsNaN);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 19, double, IsNaN);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 20, MLFloat16, IsNaN);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 20, float, IsNaN);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 20, double, IsNaN);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, IsNan);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, float, IsNan);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, double, IsNan);

// Erf operations (op 9-12, 13)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, Erf);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, float, Erf);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Erf);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, Erf);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, BFloat16, Erf);

// Cos operations (op 7-12, 13)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 12, MLFloat16, Cos);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 12, float, Cos);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Cos);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, Cos);

// Sin operations (op 7)
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, MLFloat16, Sin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, float, Sin);

// Softmax operations (op 1-10, 11-12, 13)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, Softmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, float, Softmax);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Softmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, float, Softmax);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Softmax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, Softmax);

// LogSoftmax operations (op 1-10, 11-12, 13)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, LogSoftmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, float, LogSoftmax);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, LogSoftmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, float, LogSoftmax);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, LogSoftmax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, LogSoftmax);

// LayerNormalization operations (op 1-16 legacy contrib schema, op 17 standard ONNX)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
    kMusaExecutionProvider, kOnnxDomain, 1, 16, float_float_float, LayerNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
    kMusaExecutionProvider, kOnnxDomain, 1, 16, MLFloat16_float_MLFloat16, LayerNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
    kMusaExecutionProvider, kOnnxDomain, 1, 16, BFloat16_float_BFloat16, LayerNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 17, float, LayerNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 17, MLFloat16, LayerNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 17, BFloat16, LayerNormalization);

// Abs operations (op 1-5, 6-12, 13)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 5, float, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 5, MLFloat16, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 5, int32_t, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 5, int64_t, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 5, int8_t, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 5, uint8_t, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 5, int16_t, Abs);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, float, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, int32_t, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, int64_t, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, int8_t, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, uint8_t, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, int16_t, Abs);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int8_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint8_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int16_t, Abs);

// Clip operations (op 6-10, 11, 12, 13)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 10, float, Clip_6);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 10, MLFloat16, Clip_6);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 11, Clip);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 12, 12, Clip);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, Clip);

// Sign operations (op 13)
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, Sign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, double, Sign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Sign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Sign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Sign);

// TopK operations (op 11-23, 24)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 23, float, TopK);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 23, double, TopK);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 23, MLFloat16, TopK);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 23, int32_t, TopK);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 23, int64_t, TopK);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 24, float, TopK);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 24, double, TopK);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 24, MLFloat16, TopK);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 24, int32_t, TopK);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 24, int64_t, TopK);

// Neg operations (op 6-12, 13)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, float, Neg);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Neg);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, int32_t, Neg);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, int64_t, Neg);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, int8_t, Neg);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, uint8_t, Neg);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, int16_t, Neg);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, uint16_t, Neg);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, uint32_t, Neg);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, uint64_t, Neg);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int8_t, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint8_t, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int16_t, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint16_t, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint32_t, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint64_t, Neg);

// op 7-13
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 13, int32_t, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 13, int64_t, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 13, MLFloat16, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 13, float, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int32_t, AddV2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int64_t, AddV2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, AddV2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, float, AddV2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int32_t, BiasAdd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int64_t, BiasAdd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, BiasAdd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, float, BiasAdd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int32_t, BiasAddV1);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int64_t, BiasAddV1);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, BiasAddV1);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, float, BiasAddV1);

#ifdef ENABLE_MUSA_NHWC_OPS
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 7, 13, int32_t, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 7, 13, int64_t, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 7, 13, MLFloat16, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 7, 13, float, Add);
#endif

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 13, int32_t, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 13, int64_t, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 13, MLFloat16, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 13, float, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int32_t, SubV2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int64_t, SubV2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, SubV2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, float, SubV2);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 13, int32_t, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 13, int64_t, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 13, MLFloat16, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 13, float, Mul);

#ifdef ENABLE_MUSA_NHWC_OPS
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 7, 13, int32_t, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 7, 13, int64_t, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 7, 13, MLFloat16, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 7, 13, float, Mul);
#endif

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 13, int32_t, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 13, int64_t, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 13, MLFloat16, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 13, float, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, RealDiv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, float, RealDiv);

// TensorFlow compatibility binary math operations
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, DivNoNan);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, float, DivNoNan);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, double, DivNoNan);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int32_t, SquaredDifference);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int64_t, SquaredDifference);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, SquaredDifference);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, float, SquaredDifference);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, double, SquaredDifference);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int32_t, FloorDiv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int64_t, FloorDiv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, FloorDiv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, float, FloorDiv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, double, FloorDiv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int32_t, FloorMod);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int64_t, FloorMod);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, FloorMod);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, float, FloorMod);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, double, FloorMod);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int32_t, Maximum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int64_t, Maximum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, Maximum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, float, Maximum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int32_t, Minimum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int64_t, Minimum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, Minimum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, float, Minimum);

// PRelu operations (opset 7-8, 9-15, 16+)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 8, float, PRelu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, PRelu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 15, float, PRelu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 15, MLFloat16, PRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 16, float, PRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 16, MLFloat16, PRelu);

// Min operations - versioned (op 6-11, op 12, op 13)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 11, int32_t, Min);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 11, int64_t, Min);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 11, MLFloat16, Min);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 11, float, Min);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 12, int32_t, Min);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 12, int64_t, Min);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 12, MLFloat16, Min);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 12, float, Min);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Min);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Min);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Min);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, Min);

// Max operations - versioned (op 6-11, op 12, op 13)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 11, int32_t, Max);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 11, int64_t, Max);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 11, MLFloat16, Max);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 11, float, Max);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 12, int32_t, Max);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 12, int64_t, Max);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 12, MLFloat16, Max);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 12, float, Max);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Max);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Max);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Max);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, Max);

// Sum operations - versioned (op 6-7, op 8-12, op 13+)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 7, int32_t, Sum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 7, int64_t, Sum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 7, MLFloat16, Sum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 7, float, Sum);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 8, 12, int32_t, Sum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 8, 12, int64_t, Sum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 8, 12, MLFloat16, Sum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 8, 12, float, Sum);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Sum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Sum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Sum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, Sum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int32_t, AddN);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int64_t, AddN);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, AddN);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, float, AddN);

// Pow operations - versioned (op 7-11)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 11, int32_t, Pow);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 11, int64_t, Pow);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 11, MLFloat16, Pow);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 11, float, Pow);

// Equal operations - versioned
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 6, bool, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 6, int8_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 6, uint8_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 6, int16_t, Equal);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 6, uint16_t, Equal);  // MUDNN NOT_SUPPORTED
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 6, int32_t, Equal);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 6, uint32_t, Equal);  // MUDNN NOT_SUPPORTED
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 6, int64_t, Equal);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 6, uint64_t, Equal);  // MUDNN NOT_SUPPORTED
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 6, MLFloat16, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 6, float, Equal);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 10, bool, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 10, int8_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 10, uint8_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 10, int16_t, Equal);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 10, uint16_t, Equal);  // MUDNN NOT_SUPPORTED
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 10, int32_t, Equal);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 10, uint32_t, Equal);  // MUDNN NOT_SUPPORTED
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 10, int64_t, Equal);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 10, uint64_t, Equal);  // MUDNN NOT_SUPPORTED
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 10, MLFloat16, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 10, float, Equal);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, bool, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, int8_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, uint8_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, int16_t, Equal);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, uint16_t, Equal);  // MUDNN NOT_SUPPORTED
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, int32_t, Equal);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, uint32_t, Equal);  // MUDNN NOT_SUPPORTED
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, int64_t, Equal);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, uint64_t, Equal);  // MUDNN NOT_SUPPORTED
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, float, Equal);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, bool, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int8_t, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint8_t, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int16_t, Equal);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint16_t, Equal);  // MUDNN NOT_SUPPORTED
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Equal);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint32_t, Equal);  // MUDNN NOT_SUPPORTED
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Equal);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint64_t, Equal);  // MUDNN NOT_SUPPORTED
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, Equal);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, bool, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, int8_t, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, uint8_t, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, int16_t, Equal);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, uint16_t, Equal);  // MUDNN NOT_SUPPORTED
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, int32_t, Equal);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, uint32_t, Equal);  // MUDNN NOT_SUPPORTED
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, int64_t, Equal);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, uint64_t, Equal);  // MUDNN NOT_SUPPORTED
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, MLFloat16, Equal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, float, Equal);

// TensorFlow-converted comparison compatibility ops.
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, bool, NotEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int32_t, NotEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int64_t, NotEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, NotEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, float, NotEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int32_t, GreaterEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int64_t, GreaterEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, GreaterEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, float, GreaterEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int32_t, LessEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int64_t, LessEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, LessEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, float, LessEqual);

// Less operations - versioned
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 8, int8_t, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 8, uint8_t, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 8, int16_t, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 8, int32_t, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 8, int64_t, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 8, float, Less);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, int8_t, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, uint8_t, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, int16_t, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, int32_t, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, int64_t, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, Less);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, float, Less);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int8_t, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint8_t, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int16_t, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, Less);

// Greater operations - versioned
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 8, int8_t, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 8, uint8_t, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 8, int16_t, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 8, int32_t, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 8, int64_t, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 8, float, Greater);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, int8_t, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, uint8_t, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, int16_t, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, int32_t, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, int64_t, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, float, Greater);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int8_t, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint8_t, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int16_t, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, Greater);

// GreaterOrEqual operations - versioned
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 12, 15, bool, GreaterOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 12, 15, int8_t, GreaterOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 12, 15, uint8_t, GreaterOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 12, 15, int16_t, GreaterOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 12, 15, int32_t, GreaterOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 12, 15, int64_t, GreaterOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 12, 15, MLFloat16, GreaterOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 12, 15, float, GreaterOrEqual);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 16, bool, GreaterOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 16, int8_t, GreaterOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 16, uint8_t, GreaterOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 16, int16_t, GreaterOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 16, int32_t, GreaterOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 16, int64_t, GreaterOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 16, MLFloat16, GreaterOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 16, float, GreaterOrEqual);

// LessOrEqual operations - versioned
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 12, 15, bool, LessOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 12, 15, int8_t, LessOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 12, 15, uint8_t, LessOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 12, 15, int16_t, LessOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 12, 15, int32_t, LessOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 12, 15, int64_t, LessOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 12, 15, MLFloat16, LessOrEqual);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 12, 15, float, LessOrEqual);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 16, bool, LessOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 16, int8_t, LessOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 16, uint8_t, LessOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 16, int16_t, LessOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 16, int32_t, LessOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 16, int64_t, LessOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 16, MLFloat16, LessOrEqual);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 16, float, LessOrEqual);

// Gemm operations (op 11-12, 13)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, float, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Gemm);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, Gemm);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Gemm);

// MatMul operations - versioned (op 1-8) - Float types
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 8, MLFloat16, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 8, float, MatMul);

// MatMul operations - versioned (op 9-12) - Float types
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, float, MatMul);

// MatMul operations (op 13) - Float types
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, MatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, MatMul);

// Fill operations (op 1)
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, float, Fill);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, double, Fill);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, Fill);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int32_t, Fill);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int64_t, Fill);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, bool, Fill);

// ConstantOfShape operations (op 9)
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, ConstantOfShape);

// Random generator operations
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, RandomUniformLike);

// OneHot operations (op 9-10, 11)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 10, int32_t_float_int64_t, OneHot);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 10, int32_t_int32_t_int64_t, OneHot);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 10, int32_t_int64_t_int64_t, OneHot);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 10, int64_t_float_int64_t, OneHot);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 10, int64_t_int32_t_int64_t, OneHot);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 10, int64_t_int64_t_int64_t, OneHot);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, int32_t_float_int64_t, OneHot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, int32_t_int32_t_int64_t, OneHot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, int32_t_int64_t_int64_t, OneHot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, int32_t_MLFloat16_int64_t, OneHot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, int32_t_uint8_t_int64_t, OneHot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, int32_t_int8_t_int64_t, OneHot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, int64_t_float_int64_t, OneHot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, int64_t_int32_t_int64_t, OneHot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, int64_t_int64_t_int64_t, OneHot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, int64_t_MLFloat16_int64_t, OneHot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, int64_t_uint8_t_int64_t, OneHot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, int64_t_int8_t_int64_t, OneHot);

// Range operations (op 11)
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, float, Range);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, int32_t, Range);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, int64_t, Range);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, int16_t, Range);

// op 14
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain,
                                            14, int32_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain,
                                            14, int64_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain,
                                            14, MLFloat16, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain,
                                            14, float, Add);

#ifdef ENABLE_MUSA_NHWC_OPS
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain,
                                            14, int32_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain,
                                            14, int64_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain,
                                            14, MLFloat16, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain,
                                            14, float, Add);
#endif

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain,
                                            14, int32_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain,
                                            14, int64_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain,
                                            14, MLFloat16, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain,
                                            14, float, Sub);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain,
                                            14, int32_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain,
                                            14, int64_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain,
                                            14, MLFloat16, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain,
                                            14, float, Mul);

#ifdef ENABLE_MUSA_NHWC_OPS
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain,
                                            14, int32_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain,
                                            14, int64_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain,
                                            14, MLFloat16, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain,
                                            14, float, Mul);
#endif

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain,
                                            14, int32_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain,
                                            14, int64_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain,
                                            14, MLFloat16, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain,
                                            14, float, Div);

// Pow operations - current versions (op 12+)
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 12, int32_t, Pow);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 12, int64_t, Pow);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 12, MLFloat16, Pow);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 12, float, Pow);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Pow);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Pow);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Pow);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, Pow);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 14, int32_t, Pow);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 14, int64_t, Pow);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 14, MLFloat16, Pow);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 14, float, Pow);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 15, int32_t, Pow);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 15, int64_t, Pow);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 15, MLFloat16, Pow);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 15, float, Pow);

// Mod operations (op 10-12, 13) - only for MUSA-supported types
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 10, 12, int32_t, Mod);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 10, 12, int64_t, Mod);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 10, 12, uint32_t, Mod);  // Disabled: muDNN Binary doesn't support unsigned types
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 10, 12, uint64_t, Mod);  // Disabled: muDNN Binary doesn't support unsigned types
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 10, 12, float, Mod);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 10, 12, MLFloat16, Mod);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Mod);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Mod);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint32_t, Mod);  // Disabled: muDNN Binary doesn't support unsigned types
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint64_t, Mod);  // Disabled: muDNN Binary doesn't support unsigned types
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, Mod);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Mod);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, int8_t, BitwiseAnd);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, uint8_t, BitwiseAnd);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, int16_t, BitwiseAnd);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, uint16_t, BitwiseAnd);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, int32_t, BitwiseAnd);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, uint32_t, BitwiseAnd);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, int64_t, BitwiseAnd);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, uint64_t, BitwiseAnd);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, int8_t, BitwiseOr);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, uint8_t, BitwiseOr);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, int16_t, BitwiseOr);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, uint16_t, BitwiseOr);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, int32_t, BitwiseOr);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, uint32_t, BitwiseOr);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, int64_t, BitwiseOr);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, uint64_t, BitwiseOr);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, int8_t, BitwiseXor);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, uint8_t, BitwiseXor);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, int16_t, BitwiseXor);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, uint16_t, BitwiseXor);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, int32_t, BitwiseXor);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, uint32_t, BitwiseXor);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, int64_t, BitwiseXor);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, uint64_t, BitwiseXor);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, int8_t, BitwiseNot);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, uint8_t, BitwiseNot);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, int16_t, BitwiseNot);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, uint16_t, BitwiseNot);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, int32_t, BitwiseNot);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, uint32_t, BitwiseNot);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, int64_t, BitwiseNot);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, uint64_t, BitwiseNot);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, int8_t, BitwiseAnd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, uint8_t, BitwiseAnd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, int16_t, BitwiseAnd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, uint16_t, BitwiseAnd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, int32_t, BitwiseAnd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, uint32_t, BitwiseAnd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, int64_t, BitwiseAnd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, uint64_t, BitwiseAnd);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, int8_t, BitwiseOr);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, uint8_t, BitwiseOr);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, int16_t, BitwiseOr);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, uint16_t, BitwiseOr);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, int32_t, BitwiseOr);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, uint32_t, BitwiseOr);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, int64_t, BitwiseOr);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, uint64_t, BitwiseOr);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, int8_t, BitwiseXor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, uint8_t, BitwiseXor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, int16_t, BitwiseXor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, uint16_t, BitwiseXor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, int32_t, BitwiseXor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, uint32_t, BitwiseXor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, int64_t, BitwiseXor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, uint64_t, BitwiseXor);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, int8_t, BitwiseNot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, uint8_t, BitwiseNot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, int16_t, BitwiseNot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, uint16_t, BitwiseNot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, int32_t, BitwiseNot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, uint32_t, BitwiseNot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, int64_t, BitwiseNot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, uint64_t, BitwiseNot);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 21, MLFloat16, Round);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 21, float, Round);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 22, MLFloat16, Round);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 22, float, Round);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain,
                                            14, float, Gemm);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain,
                                            14, MLFloat16, Gemm);

// CumSum operations (op 11-13, 14)
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 13, CumSum);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 14, CumSum);

// Logical unary operations
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, bool, Not);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, bool, And);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, bool, Or);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, bool, Xor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, bool, LogicalNot);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, bool, LogicalAnd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, bool, LogicalOr);

// ReduceMax operations (op 1-17, 18)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, MLFloat16, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, float, ReduceMax);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, MLFloat16, ReduceMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, float, ReduceMax);

// ReduceMin operations (op 1-17, 18)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, MLFloat16, ReduceMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, float, ReduceMin);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, MLFloat16, ReduceMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, float, ReduceMin);

// ReduceSum operations (op 1-12, 13)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 12, MLFloat16, ReduceSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 12, float, ReduceSum);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, ReduceSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, ReduceSum);

// ReduceSumSquare operations (op 1-17, 18)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, MLFloat16, ReduceSumSquare);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, float, ReduceSumSquare);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, MLFloat16, ReduceSumSquare);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, float, ReduceSumSquare);

// ReduceMean operations (op 1-12, 13)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 12, MLFloat16, ReduceMean);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 12, float, ReduceMean);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, ReduceMean);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, ReduceMean);

// ReduceProd operations (op 1-17, 18)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, MLFloat16, ReduceProd);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, float, ReduceProd);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, int32_t, ReduceProd);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, int64_t, ReduceProd);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, MLFloat16, ReduceProd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, float, ReduceProd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, int32_t, ReduceProd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, int64_t, ReduceProd);

// ReduceL2 operations (op 1-17 uses axes attribute, 18+ supports axes input)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, MLFloat16, ReduceL2);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 17, float, ReduceL2);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, MLFloat16, ReduceL2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, float, ReduceL2);

// ArgMax operations (op 1-10, 11-12, 13) - Float types
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ArgMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, float, ArgMax);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, ArgMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, float, ArgMax);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, ArgMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, ArgMax);

// ArgMax operations (op 1-10, 11-12, 13) - Integer types
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, int32_t, ArgMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, int64_t, ArgMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, int8_t, ArgMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, uint8_t, ArgMax);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, int32_t, ArgMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, int64_t, ArgMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, int8_t, ArgMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, uint8_t, ArgMax);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int32_t, ArgMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int64_t, ArgMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int8_t, ArgMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint8_t, ArgMax);

// ArgMin operations (op 1-10, 11-12, 13) - Float types
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ArgMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, float, ArgMin);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, ArgMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, float, ArgMin);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, ArgMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, ArgMin);

// ArgMin operations (op 1-10, 11-12, 13) - Integer types
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, int32_t, ArgMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, int64_t, ArgMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, int8_t, ArgMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, uint8_t, ArgMin);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, int32_t, ArgMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, int64_t, ArgMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, int8_t, ArgMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, uint8_t, ArgMin);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int32_t, ArgMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int64_t, ArgMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int8_t, ArgMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint8_t, ArgMin);

// Transpose operations (op 1-12, 13+)
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, ReverseV2);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, InvertPermutation);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 12, uint8_t, Transpose);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 12, uint16_t, Transpose);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 12, uint32_t, Transpose);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 12, uint64_t, Transpose);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 12, int8_t, Transpose);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 12, int16_t, Transpose);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 12, int32_t, Transpose);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 12, int64_t, Transpose);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 12, MLFloat16, Transpose);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 12, float, Transpose);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 12, bool, Transpose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint8_t, Transpose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint16_t, Transpose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint32_t, Transpose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint64_t, Transpose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int8_t, Transpose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int16_t, Transpose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Transpose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Transpose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Transpose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, Transpose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, bool, Transpose);

// Resize operations (op 10, 11-12, 13-17, 18+)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 10, 10, float, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 10, 10, MLFloat16, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, float, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 17, float, Resize);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 17, MLFloat16, Resize);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, float, Resize);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, MLFloat16, Resize);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 16, float, GridSample);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 16, MLFloat16, GridSample);
#ifdef ENABLE_MUSA_NHWC_OPS
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 16, float, GridSample);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 16, MLFloat16, GridSample);
#endif

// Gather operations (op 1-10, 11-12, 13)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, uint8_t, Gather);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, uint16_t, Gather);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, uint32_t, Gather);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, uint64_t, Gather);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, int8_t, Gather);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, int16_t, Gather);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, int32_t, Gather);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, int64_t, Gather);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, Gather);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, float, Gather);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, double, Gather);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, bool, Gather);

// Slice operations (op 1-9, 10, 11-12, 13)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 9, float, Slice);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 9, MLFloat16, Slice);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 9, int32_t, Slice);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 9, int64_t, Slice);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 10, 10, float, Slice);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 10, 10, MLFloat16, Slice);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 10, 10, int32_t, Slice);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 10, 10, int64_t, Slice);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, float, Slice);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Slice);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, int32_t, Slice);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, int64_t, Slice);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, Slice);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Slice);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Slice);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Slice);

// Split operations (op 2-10, 11-12, 13-17, 18)
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 2, 10, Split);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, SplitV);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, Split);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 17, Split);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, Split);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, uint8_t, Gather);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, uint16_t, Gather);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, uint32_t, Gather);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, uint64_t, Gather);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, int8_t, Gather);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, int16_t, Gather);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, int32_t, Gather);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, int64_t, Gather);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Gather);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, float, Gather);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, double, Gather);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, bool, Gather);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint8_t, Gather);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint16_t, Gather);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint32_t, Gather);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint64_t, Gather);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int8_t, Gather);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int16_t, Gather);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Gather);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Gather);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Gather);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, Gather);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, double, Gather);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, bool, Gather);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int32_t, GatherV2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int64_t, GatherV2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, GatherV2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, float, GatherV2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, double, GatherV2);

// GatherElements operations (op 11+)
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, uint8_t, GatherElements);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, uint16_t, GatherElements);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, uint32_t, GatherElements);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, uint64_t, GatherElements);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, int8_t, GatherElements);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, int16_t, GatherElements);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, int32_t, GatherElements);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, int64_t, GatherElements);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, MLFloat16, GatherElements);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, float, GatherElements);

// GatherND operations (op 11, 12, 13+)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 11, int64_t, GatherND);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 12, 12, int64_t, GatherND);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int64_t, GatherND);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int64_t, GatherNd);

// ScatterND operations (op 11-12, 13-15, 16-17, 18+)
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, ScatterND);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 15, ScatterND);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 16, 17, ScatterND);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, ScatterND);

// NonZero operations (op 9-12, 13+)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, bool, NonZero);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, uint8_t, NonZero);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, int32_t, NonZero);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, int64_t, NonZero);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, float, NonZero);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, bool, NonZero);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint8_t, NonZero);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int32_t, NonZero);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int64_t, NonZero);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, NonZero);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, double, NonZero);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, NonZero);

// Unique operations (op 11+)
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, float, Unique);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, double, Unique);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, int8_t, Unique);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, int32_t, Unique);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, int64_t, Unique);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, MLFloat16, Unique);

// Tile operations (op 6-12, 13)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, float, Tile);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, int32_t, Tile);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, int64_t, Tile);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Tile);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, Tile);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Tile);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Tile);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Tile);

// Unsqueeze operations (op 1-10, 11-12, 13)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, uint8_t, Unsqueeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, uint16_t, Unsqueeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, uint32_t, Unsqueeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, uint64_t, Unsqueeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, int8_t, Unsqueeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, int16_t, Unsqueeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, int32_t, Unsqueeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, int64_t, Unsqueeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, Unsqueeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, float, Unsqueeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, bool, Unsqueeze);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, uint8_t, Unsqueeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, uint16_t, Unsqueeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, uint32_t, Unsqueeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, uint64_t, Unsqueeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, int8_t, Unsqueeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, int16_t, Unsqueeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, int32_t, Unsqueeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, int64_t, Unsqueeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Unsqueeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, float, Unsqueeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, bool, Unsqueeze);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint8_t, Unsqueeze);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint16_t, Unsqueeze);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint32_t, Unsqueeze);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint64_t, Unsqueeze);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int8_t, Unsqueeze);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int16_t, Unsqueeze);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Unsqueeze);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Unsqueeze);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Unsqueeze);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, Unsqueeze);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, bool, Unsqueeze);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, float, ExpandDims);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int32_t, ExpandDims);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int64_t, ExpandDims);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, ExpandDims);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, bool, ExpandDims);

// Squeeze operations (op 1-10, 11-12, 13)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, uint8_t, Squeeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, uint16_t, Squeeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, uint32_t, Squeeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, uint64_t, Squeeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, int8_t, Squeeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, int16_t, Squeeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, int32_t, Squeeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, int64_t, Squeeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, Squeeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, float, Squeeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, bool, Squeeze);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, uint8_t, Squeeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, uint16_t, Squeeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, uint32_t, Squeeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, uint64_t, Squeeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, int8_t, Squeeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, int16_t, Squeeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, int32_t, Squeeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, int64_t, Squeeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Squeeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, float, Squeeze);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, bool, Squeeze);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint8_t, Squeeze);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint16_t, Squeeze);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint32_t, Squeeze);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint64_t, Squeeze);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int8_t, Squeeze);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int16_t, Squeeze);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Squeeze);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Squeeze);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Squeeze);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, Squeeze);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, bool, Squeeze);

// Concat operations (op 4-10, 11-12, and 13)
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, ConcatV2);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 4, 10, uint8_t, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 4, 10, uint16_t, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 4, 10, uint32_t, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 4, 10, uint64_t, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 4, 10, int8_t, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 4, 10, int16_t, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 4, 10, int32_t, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 4, 10, int64_t, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 4, 10, MLFloat16, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 4, 10, float, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 4, 10, bool, Concat);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, uint8_t, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, uint16_t, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, uint32_t, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, uint64_t, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, int8_t, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, int16_t, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, int32_t, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, int64_t, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, float, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, bool, Concat);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint8_t, Concat);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint16_t, Concat);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint32_t, Concat);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, uint64_t, Concat);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int8_t, Concat);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int16_t, Concat);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Concat);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Concat);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Concat);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, Concat);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, bool, Concat);

#ifdef ENABLE_MUSA_NHWC_OPS
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 4, 10, uint8_t, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 4, 10, uint16_t, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 4, 10, uint32_t, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 4, 10, uint64_t, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 4, 10, int8_t, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 4, 10, int16_t, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 4, 10, int32_t, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 4, 10, int64_t, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 4, 10, MLFloat16, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 4, 10, float, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 4, 10, bool, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 11, 12, uint8_t, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 11, 12, uint16_t, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 11, 12, uint32_t, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 11, 12, uint64_t, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 11, 12, int8_t, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 11, 12, int16_t, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 11, 12, int32_t, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 11, 12, int64_t, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 11, 12, MLFloat16, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 11, 12, float, Concat);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 11, 12, bool, Concat);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 13, uint8_t, Concat);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 13, uint16_t, Concat);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 13, uint32_t, Concat);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 13, uint64_t, Concat);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 13, int8_t, Concat);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 13, int16_t, Concat);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 13, int32_t, Concat);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 13, int64_t, Concat);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 13, MLFloat16, Concat);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 13, float, Concat);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 13, bool, Concat);
#endif

// Pad operations (op 2-10, 11-17, 18)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 2, 10, float, Pad);
// V0.17.5/V0.17.6.1: muDNN 3.1+ supports Pad fp16, conditionally registered.
// Older SDKs (e.g. M1000 SDK 4.1.2 / muDNN 2.9.x) auto-fallback to CPU EP, same behavior as V0.17.1.
#if defined(MUDNN_VERSION) && (MUDNN_VERSION >= 3100)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 2, 10, MLFloat16, Pad);
#endif

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 17, float, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 17, float, PadV2);
#if defined(MUDNN_VERSION) && (MUDNN_VERSION >= 3100)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 17, MLFloat16, Pad);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 17, MLFloat16, PadV2);
#endif

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, float, Pad);
#if defined(MUDNN_VERSION) && (MUDNN_VERSION >= 3100)
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, MLFloat16, Pad);
#endif

// Flatten operations (op 1-8, 9-10, 11-12, 13)
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 8, Flatten);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 10, Flatten);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, Flatten);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, Flatten);

// Reshape operations (op 1-4, 5-12, 13, 14-18, 19)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 4, int32_t, Reshape);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 4, int64_t, Reshape);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 4, MLFloat16, Reshape);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 4, float, Reshape);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 4, bool, Reshape);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 5, 12, int32_t, Reshape);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 5, 12, int64_t, Reshape);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 5, 12, MLFloat16, Reshape);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 5, 12, float, Reshape);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 5, 12, bool, Reshape);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 13, int32_t, Reshape);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 13, int64_t, Reshape);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, Reshape);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 13, float, Reshape);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 13, bool, Reshape);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 14, 18, int32_t, Reshape);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 14, 18, int64_t, Reshape);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 14, 18, MLFloat16, Reshape);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 14, 18, float, Reshape);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 14, 18, bool, Reshape);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 14, 18, string, Reshape);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, int32_t, Reshape);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, int64_t, Reshape);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, MLFloat16, Reshape);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, float, Reshape);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, bool, Reshape);

// Shape operations (multi-version support)
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 12, Shape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 14, Shape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 15, 18, Shape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, 20, Shape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 21, 22, Shape);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 23, Shape);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, ShapeN);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, BroadcastGradientArgs);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, ConcatOffset);

// Size operations (multi-version support)
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 12, Size);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 20, Size);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 21, 22, Size);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 23, 24, Size);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 25, Size);

// Where operations (op 9-15, 16)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 15, float, Where);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 15, MLFloat16, Where);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 15, int32_t, Where);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 15, int64_t, Where);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 16, float, Where);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 16, MLFloat16, Where);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 16, int32_t, Where);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 16, int64_t, Where);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, float, Select);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, double, Select);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, Select);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int32_t, Select);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int64_t, Select);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, float, SelectV2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, double, SelectV2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, SelectV2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int32_t, SelectV2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int64_t, SelectV2);

// Identity operations (tensor subset)
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 12, Identity);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, StopGradient);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, Snapshot);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 13, Identity);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 14, 18, Identity);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, 20, Identity);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 21, 22, Identity);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 23, 24, Identity);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 25, Identity);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, IdentityN);

// Dropout inference no-op operations
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 7, 9, Dropout);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 10, 11, Dropout);

// Cast operations (version 6-8)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 8, int32_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 8, int64_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 8, MLFloat16, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 8, float, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 8, int8_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 8, int16_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 8, uint8_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 8, uint16_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 8, uint32_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 8, uint64_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 8, bool, Cast);

// Cast operations (version 9-12)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, int32_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, int64_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, float, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, int8_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, int16_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, uint8_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, uint16_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, uint32_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, uint64_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 12, bool, Cast);

// Cast operations (version 13-18)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 18, int32_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 18, int64_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 18, MLFloat16, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 18, float, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 18, int8_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 18, int16_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 18, uint8_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 18, uint16_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 18, uint32_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 18, uint64_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 18, bool, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 18, string, Cast);

// Cast operations (version 19+)
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, int32_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, int64_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, MLFloat16, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, float, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, int8_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, int16_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, uint8_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, uint16_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, uint32_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, uint64_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, bool, Cast);

// Expand operations (version 8-12 and 13)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 8, 12, float, Expand);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 8, 12, int32_t, Expand);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 8, 12, int64_t, Expand);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 8, 12, MLFloat16, Expand);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, Expand);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Expand);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Expand);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Expand);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, float, BroadcastTo);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int32_t, BroadcastTo);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, int64_t, BroadcastTo);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, BroadcastTo);

// Tile operations (op 6-12 and op 13)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, float, Tile);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, int32_t, Tile);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, int64_t, Tile);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Tile);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, Tile);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Tile);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Tile);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Tile);

// If operations (control flow)
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, If);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, If);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 18, If);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, If);

Status RegisterMusaKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      // Register Memcpy operations for data transfer
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,

      // Register MUSA EP fused kernels
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSDomain, 1, MusaTokenMixerResidual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSDomain, 1, MusaReshapeMatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSDomain, 1, MusaFeatureNorm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSDomain, 1, MusaGelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSDomain, 1, MusaPlnCascadeBlock)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSDomain, 1, MusaLayerNormLastDim)>,

      // Register Conv operators with float type (NCHW)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain,
                                                                            1, 10, float, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain,
                                                                            1, 10, MLFloat16, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain,
                                                                  11, float, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain,
                                                                  11, MLFloat16, Conv)>,

      // Register ConvTranspose operators with float type (NCHW)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain,
                                                                            1, 10, float, ConvTranspose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain,
                                                                            1, 10, MLFloat16, ConvTranspose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain,
                                                                  11, float, ConvTranspose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain,
                                                                  11, MLFloat16, ConvTranspose)>,

#ifdef ENABLE_MUSA_NHWC_OPS
      // Register Conv operators (NHWC)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 1, 10, float, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 1, 10, MLFloat16, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 11, float, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain, 11, MLFloat16, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider,
                                                                            kMSInternalNHWCDomain, 1, 10, float,
                                                                            ConvTranspose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider,
                                                                            kMSInternalNHWCDomain, 1, 10, MLFloat16,
                                                                            ConvTranspose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain,
                                                                  11, float, ConvTranspose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kMSInternalNHWCDomain,
                                                                  11, MLFloat16, ConvTranspose)>,
#endif

      // Register InstanceNormalization operators
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain,
                                                                            1, 5, MLFloat16, InstanceNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain,
                                                                            1, 5, float, InstanceNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain,
                                                                  6, MLFloat16, InstanceNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain,
                                                                  6, float, InstanceNormalization)>,

      // Register BatchNormalization operators
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain,
                                                                  15, MLFloat16, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain,
                                                                  15, float, BatchNormalization)>,

      // Register LSTM operators (temporarily disabled due to weight format issues)
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain,
      //                                                                       7, 13, float, LSTM)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain,
      //                                                             14, float, LSTM)>,

      // Register MaxPool operators
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 7, MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 8, 9, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 8, 9, MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 10, 10, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 10, 10, MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 11, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 11, MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 12, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 12, MLFloat16, MaxPool)>,

      // Register AveragePool operators
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 9, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 9, MLFloat16, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 10, 10, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 10, 10, MLFloat16, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, MLFloat16, AveragePool)>,

      // Register GlobalMaxPool operators
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, float, GlobalMaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, GlobalMaxPool)>,

      // Register GlobalAveragePool operators
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, float, GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, GlobalAveragePool)>,

#ifdef ENABLE_MUSA_NHWC_OPS
      // Register Pool operators (NHWC)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 1, 7, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 1, 7, MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 8, 9, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 8, 9, MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 10, 10, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 10, 10, MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 11, 11, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 11, 11, MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 12, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 12, MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 7, 9, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 7, 9, MLFloat16, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 10, 10, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 10, 10, MLFloat16, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 11, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 11, MLFloat16, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 1, float, GlobalMaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 1, MLFloat16, GlobalMaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 1, float, GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 1, MLFloat16, GlobalAveragePool)>,
#endif

      // Register Relu operators (version 6-12)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, float, Relu)>,

      // Register Relu operators (version 13)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, 13, float, Relu)>,

      // Register Relu operators (version 14)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, MLFloat16, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, float, Relu)>,

#ifdef ENABLE_MUSA_NHWC_OPS
      // Register Relu operators (NHWC)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 6, 12, MLFloat16, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 6, 12, float, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 13, 13, MLFloat16, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 13, 13, float, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 14, MLFloat16, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 14, float, Relu)>,
#endif

      // Register Tanh operators (version 6-12)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Tanh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, float, Tanh)>,

      // Register Tanh operators (version 13)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Tanh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, Tanh)>,

      // Register Sigmoid operators (version 6-12)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Sigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, float, Sigmoid)>,

      // Register Sigmoid operators (version 13)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Sigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, Sigmoid)>,

#ifdef ENABLE_MUSA_NHWC_OPS
      // Register Sigmoid operators (NHWC)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 6, 12, MLFloat16, Sigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 6, 12, float, Sigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 13, MLFloat16, Sigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 13, float, Sigmoid)>,
#endif

      // Register LeakyRelu operators (version 6-15)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 15, MLFloat16, LeakyRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 15, float, LeakyRelu)>,

      // Register LeakyRelu operators (version 16)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 16, MLFloat16, LeakyRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 16, float, LeakyRelu)>,

      // Register Log operators (version 6-12)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Log)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, float, Log)>,

      // Register Log operators (version 13)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Log)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, Log)>,

      // Register Softplus operators (version 1+)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, Softplus)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, float, Softplus)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, BFloat16, Softplus)>,

      // Register TensorFlow-converted Log1p compatibility operators
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, Log1p)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, float, Log1p)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, double, Log1p)>,

      // Register TensorFlow-converted Expm1 compatibility operators
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, Expm1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, float, Expm1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, double, Expm1)>,

      // Register TensorFlow-converted Square/Rsqrt compatibility operators
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, Square)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, float, Square)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, double, Square)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int32_t, Square)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int64_t, Square)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, Rsqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, float, Rsqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, double, Rsqrt)>,

      // Register TensorFlow-converted ZerosLike compatibility operators
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, bool, ZerosLike)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, float, ZerosLike)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, double, ZerosLike)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int32_t, ZerosLike)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int64_t, ZerosLike)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, ZerosLike)>,

      // Register Exp operators (version 6-12)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Exp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, float, Exp)>,

      // Register Exp operators (version 13)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Exp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, Exp)>,

      // Register Floor operators (version 6-12)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Floor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, float, Floor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, double, Floor)>,

      // Register Floor operators (version 13)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Floor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, Floor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, double, Floor)>,

      // Register Ceil operators (version 6-12)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Ceil)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, float, Ceil)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, double, Ceil)>,

      // Register Ceil operators (version 13)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Ceil)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, Ceil)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, double, Ceil)>,

      // Register Reciprocal operators (version 6-12)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Reciprocal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, float, Reciprocal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, double, Reciprocal)>,

      // Register Reciprocal operators (version 13)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Reciprocal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, Reciprocal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, double, Reciprocal)>,

      // Register Sqrt operators (version 6-12)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Sqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, float, Sqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, double, Sqrt)>,

      // Register Sqrt operators (version 13)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Sqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, Sqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, double, Sqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, BFloat16, Sqrt)>,

      // Register IsNaN operators (version 9-12)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, IsNaN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, float, IsNaN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, double, IsNaN)>,

      // Register IsNaN operators (version 13-19)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, 19, MLFloat16, IsNaN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, 19, float, IsNaN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, 19, double, IsNaN)>,

      // Register IsNaN operators (version 20)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 20, MLFloat16, IsNaN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 20, float, IsNaN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 20, double, IsNaN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, IsNan)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, float, IsNan)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, double, IsNan)>,

      // Register Erf operators (version 9-12)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, Erf)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, float, Erf)>,

      // Register Erf operators (version 13)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Erf)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, Erf)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, BFloat16, Erf)>,

      // Register Cos operators (version 7-12)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 12, MLFloat16, Cos)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 12, float, Cos)>,

      // Register Cos operators (version 13)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Cos)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, Cos)>,

      // Register Sin operators (version 7)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, MLFloat16, Sin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, float, Sin)>,

      // Register Softmax operators (version 1-10)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, float, Softmax)>,

      // Register Softmax operators (version 11-12)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, float, Softmax)>,

      // Register Softmax operators (version 13)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, Softmax)>,

      // Register LogSoftmax operators (version 1-10)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, LogSoftmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, float, LogSoftmax)>,

      // Register LogSoftmax operators (version 11-12)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, LogSoftmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, float, LogSoftmax)>,

      // Register LogSoftmax operators (version 13)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, LogSoftmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, LogSoftmax)>,

      // Register LayerNormalization operators (version 1-16 legacy contrib schema, version 17 standard ONNX)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 16, float_float_float, LayerNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 16, MLFloat16_float_MLFloat16, LayerNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 16, BFloat16_float_BFloat16, LayerNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 17, float, LayerNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 17, MLFloat16, LayerNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 17, BFloat16, LayerNormalization)>,

      // Register Abs operators (version 1-5)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 5, float, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 5, MLFloat16, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 5, int32_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 5, int64_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 5, int8_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 5, uint8_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 5, int16_t, Abs)>,

      // Register Abs operators (version 6-12)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, float, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, int32_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, int64_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, int8_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, uint8_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, int16_t, Abs)>,

      // Register Abs operators (version 13)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int8_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, uint8_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int16_t, Abs)>,

      // Register Clip operators (version 6-10)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 10, float, Clip_6)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 10, MLFloat16, Clip_6)>,

      // Register Clip operators (version 11-13)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 11, Clip)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 12, 12, Clip)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, Clip)>,

      // Register Sign operators (version 13)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, Sign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, double, Sign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Sign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Sign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Sign)>,

      // Register TopK operators (versions 11-23, 24)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 23, float, TopK)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 23, double, TopK)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 23, MLFloat16, TopK)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 23, int32_t, TopK)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 23, int64_t, TopK)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 24, float, TopK)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 24, double, TopK)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 24, MLFloat16, TopK)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 24, int32_t, TopK)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 24, int64_t, TopK)>,

      // Register Neg operators (version 6-12)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, float, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, int32_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, int64_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, int8_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, uint8_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, int16_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, uint16_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, uint32_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, uint64_t, Neg)>,

      // Register Neg operators (version 13)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int8_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, uint8_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int16_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, uint16_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, uint32_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, uint64_t, Neg)>,

      // Register Gemm operators for versions 11-12
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, float, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Gemm)>,

      // Register Gemm operator for version 13
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Gemm)>,

      // Register Fill operators
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, float, Fill)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, double, Fill)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, Fill)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int32_t, Fill)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int64_t, Fill)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, bool, Fill)>,

      // Register ConstantOfShape operator
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, ConstantOfShape)>,

      // Register RandomUniformLike operator
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, RandomUniformLike)>,

      //   // Register OneHot operators
      //   BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
      //       kMusaExecutionProvider, kOnnxDomain, 9, 10, int32_t_float_int64_t, OneHot)>,
      //   BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
      //       kMusaExecutionProvider, kOnnxDomain, 9, 10, int32_t_int32_t_int64_t, OneHot)>,
      //   BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
      //       kMusaExecutionProvider, kOnnxDomain, 9, 10, int32_t_int64_t_int64_t, OneHot)>,
      //   BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
      //       kMusaExecutionProvider, kOnnxDomain, 9, 10, int64_t_float_int64_t, OneHot)>,
      //   BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
      //       kMusaExecutionProvider, kOnnxDomain, 9, 10, int64_t_int32_t_int64_t, OneHot)>,
      //   BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
      //       kMusaExecutionProvider, kOnnxDomain, 9, 10, int64_t_int64_t_int64_t, OneHot)>,

      //   BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
      //       kMusaExecutionProvider, kOnnxDomain, 11, int32_t_float_int64_t, OneHot)>,
      //   BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
      //       kMusaExecutionProvider, kOnnxDomain, 11, int32_t_int32_t_int64_t, OneHot)>,
      //   BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
      //       kMusaExecutionProvider, kOnnxDomain, 11, int32_t_int64_t_int64_t, OneHot)>,
      //   BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
      //       kMusaExecutionProvider, kOnnxDomain, 11, int32_t_MLFloat16_int64_t, OneHot)>,
      //   BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
      //       kMusaExecutionProvider, kOnnxDomain, 11, int32_t_uint8_t_int64_t, OneHot)>,
      //   BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
      //       kMusaExecutionProvider, kOnnxDomain, 11, int32_t_int8_t_int64_t, OneHot)>,
      //   BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
      //       kMusaExecutionProvider, kOnnxDomain, 11, int64_t_float_int64_t, OneHot)>,
      //   BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
      //       kMusaExecutionProvider, kOnnxDomain, 11, int64_t_int32_t_int64_t, OneHot)>,
      //   BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
      //       kMusaExecutionProvider, kOnnxDomain, 11, int64_t_int64_t_int64_t, OneHot)>,
      //   BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
      //       kMusaExecutionProvider, kOnnxDomain, 11, int64_t_MLFloat16_int64_t, OneHot)>,
      //   BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
      //       kMusaExecutionProvider, kOnnxDomain, 11, int64_t_uint8_t_int64_t, OneHot)>,
      //   BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
      //       kMusaExecutionProvider, kOnnxDomain, 11, int64_t_int8_t_int64_t, OneHot)>,

      // Register Range operators
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, float, Range)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, int32_t, Range)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, int64_t, Range)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, int16_t, Range)>,

      // Register versioned Add operators (7-13)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 13, int32_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 13, int64_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 13, MLFloat16, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 13, float, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int32_t, AddV2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int64_t, AddV2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, AddV2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, float, AddV2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int32_t, BiasAdd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int64_t, BiasAdd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, BiasAdd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, float, BiasAdd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int32_t, BiasAddV1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int64_t, BiasAddV1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, BiasAddV1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, float, BiasAddV1)>,

#ifdef ENABLE_MUSA_NHWC_OPS
      // Register versioned Add operators (NHWC, 7-13)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 7, 13, int32_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 7, 13, int64_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 7, 13, MLFloat16, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 7, 13, float, Add)>,
#endif

      // Register versioned Sub operators (7-13)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 13, int32_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 13, int64_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 13, MLFloat16, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 13, float, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int32_t, SubV2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int64_t, SubV2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, SubV2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, float, SubV2)>,

      // Register versioned Mul operators (7-13)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 13, int32_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 13, int64_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 13, MLFloat16, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 13, float, Mul)>,

#ifdef ENABLE_MUSA_NHWC_OPS
      // Register versioned Mul operators (NHWC, 7-13)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 7, 13, int32_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 7, 13, int64_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 7, 13, MLFloat16, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 7, 13, float, Mul)>,
#endif

      // Register versioned Div operators (7-13)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 13, int32_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 13, int64_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 13, MLFloat16, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 13, float, Div)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 8, float, PRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, PRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 15, float, PRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 15, MLFloat16, PRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 16, float, PRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 16, MLFloat16, PRelu)>,

      // Register versioned Min operators (6-11)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 11, int32_t, Min)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 11, int64_t, Min)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 11, MLFloat16, Min)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 11, float, Min)>,

      // Register Min operators (version 12)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 12, int32_t, Min)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 12, int64_t, Min)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 12, MLFloat16, Min)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 12, float, Min)>,

      // Register Min operators (version 13)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Min)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Min)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Min)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, Min)>,

      // Register versioned Max operators (6-11)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 11, int32_t, Max)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 11, int64_t, Max)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 11, MLFloat16, Max)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 11, float, Max)>,

      // Register Max operators (version 12)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 12, int32_t, Max)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 12, int64_t, Max)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 12, MLFloat16, Max)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 12, float, Max)>,

      // Register Max operators (version 13)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Max)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Max)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Max)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, Max)>,

      // Register versioned Sum operators (6-7, 8-12, 13+)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 7, int32_t, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 7, int64_t, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 7, MLFloat16, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 7, float, Sum)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 8, 12, int32_t, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 8, 12, int64_t, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 8, 12, MLFloat16, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 8, 12, float, Sum)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int32_t, AddN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int64_t, AddN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, AddN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, float, AddN)>,

      // Register versioned Pow operators (7-11)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 11, int32_t, Pow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 11, int64_t, Pow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 11, MLFloat16, Pow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 11, float, Pow)>,

      // Register versioned Equal operators (1-6)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 6, bool, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 6, int8_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 6, uint8_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 6, int16_t, Equal)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
      //     kMusaExecutionProvider, kOnnxDomain, 1, 6, uint16_t, Equal)>,  // MUDNN NOT_SUPPORTED
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 6, int32_t, Equal)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
      //     kMusaExecutionProvider, kOnnxDomain, 1, 6, uint32_t, Equal)>,  // MUDNN NOT_SUPPORTED
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 6, int64_t, Equal)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
      //     kMusaExecutionProvider, kOnnxDomain, 1, 6, uint64_t, Equal)>,  // MUDNN NOT_SUPPORTED
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 6, MLFloat16, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 6, float, Equal)>,

      // Register versioned Equal operators (7-10)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 10, bool, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 10, int8_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 10, uint8_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 10, int16_t, Equal)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
      //     kMusaExecutionProvider, kOnnxDomain, 7, 10, uint16_t, Equal)>,  // MUDNN NOT_SUPPORTED
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 10, int32_t, Equal)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
      //     kMusaExecutionProvider, kOnnxDomain, 7, 10, uint32_t, Equal)>,  // MUDNN NOT_SUPPORTED
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 10, int64_t, Equal)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
      //     kMusaExecutionProvider, kOnnxDomain, 7, 10, uint64_t, Equal)>,  // MUDNN NOT_SUPPORTED
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 10, MLFloat16, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 10, float, Equal)>,

      // Register versioned Equal operators (11-12)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, bool, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, int8_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, uint8_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, int16_t, Equal)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
      //     kMusaExecutionProvider, kOnnxDomain, 11, 12, uint16_t, Equal)>,  // MUDNN NOT_SUPPORTED
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, int32_t, Equal)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
      //     kMusaExecutionProvider, kOnnxDomain, 11, 12, uint32_t, Equal)>,  // MUDNN NOT_SUPPORTED
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, int64_t, Equal)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
      //     kMusaExecutionProvider, kOnnxDomain, 11, 12, uint64_t, Equal)>,  // MUDNN NOT_SUPPORTED
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, float, Equal)>,

      // Register Equal operators for version 13
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, bool, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int8_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, uint8_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int16_t, Equal)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
      //     kMusaExecutionProvider, kOnnxDomain, 13, uint16_t, Equal)>,  // MUDNN NOT_SUPPORTED
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Equal)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
      //     kMusaExecutionProvider, kOnnxDomain, 13, uint32_t, Equal)>,  // MUDNN NOT_SUPPORTED
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Equal)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
      //     kMusaExecutionProvider, kOnnxDomain, 13, uint64_t, Equal)>,  // MUDNN NOT_SUPPORTED
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, Equal)>,

      // Register Equal operators for version 19
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 19, bool, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 19, int8_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 19, uint8_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 19, int16_t, Equal)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
      //     kMusaExecutionProvider, kOnnxDomain, 19, uint16_t, Equal)>,  // MUDNN NOT_SUPPORTED
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 19, int32_t, Equal)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
      //     kMusaExecutionProvider, kOnnxDomain, 19, uint32_t, Equal)>,  // MUDNN NOT_SUPPORTED
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 19, int64_t, Equal)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
      //     kMusaExecutionProvider, kOnnxDomain, 19, uint64_t, Equal)>,  // MUDNN NOT_SUPPORTED
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 19, MLFloat16, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 19, float, Equal)>,

      // Register TensorFlow-converted comparison compatibility operators
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, bool, NotEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int32_t, NotEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int64_t, NotEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, NotEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, float, NotEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int32_t, GreaterEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int64_t, GreaterEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, GreaterEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, float, GreaterEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int32_t, LessEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int64_t, LessEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, LessEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, float, LessEqual)>,

      // Register versioned Less operators (7-8)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 8, int8_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 8, uint8_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 8, int16_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 8, int32_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 8, int64_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 8, float, Less)>,

      // Register versioned Less operators (9-12)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, int8_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, uint8_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, int16_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, int32_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, int64_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, float, Less)>,

      // Register Less operators for version 13
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int8_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, uint8_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int16_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, Less)>,

      // Register versioned Greater operators (7-8)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 8, int8_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 8, uint8_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 8, int16_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 8, int32_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 8, int64_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 8, float, Greater)>,

      // Register versioned Greater operators (9-12)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, int8_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, uint8_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, int16_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, int32_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, int64_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, float, Greater)>,

      // Register Greater operators for version 13
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int8_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, uint8_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int16_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, Greater)>,

      // Register GreaterOrEqual operators for version 12-15
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 12, 15, bool, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 12, 15, int8_t, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 12, 15, uint8_t, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 12, 15, int16_t, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 12, 15, int32_t, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 12, 15, int64_t, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 12, 15, MLFloat16, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 12, 15, float, GreaterOrEqual)>,

      // Register GreaterOrEqual operators for version 16
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 16, bool, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 16, int8_t, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 16, uint8_t, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 16, int16_t, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 16, int32_t, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 16, int64_t, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 16, MLFloat16, GreaterOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 16, float, GreaterOrEqual)>,

      // Register LessOrEqual operators for version 12-15
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 12, 15, bool, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 12, 15, int8_t, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 12, 15, uint8_t, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 12, 15, int16_t, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 12, 15, int32_t, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 12, 15, int64_t, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 12, 15, MLFloat16, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 12, 15, float, LessOrEqual)>,

      // Register LessOrEqual operators for version 16
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 16, bool, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 16, int8_t, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 16, uint8_t, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 16, int16_t, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 16, int32_t, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 16, int64_t, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 16, MLFloat16, LessOrEqual)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 16, float, LessOrEqual)>,

      // Register Add operator for version 14
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, int32_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, int64_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, MLFloat16, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, float, Add)>,

#ifdef ENABLE_MUSA_NHWC_OPS
      // Register Add operator for version 14 (NHWC)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 14, int32_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 14, int64_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 14, MLFloat16, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 14, float, Add)>,
#endif

      // Register Sub operator for version 14
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, int32_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, int64_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, MLFloat16, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, float, Sub)>,

      // Register Mul operator for version 14
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, int32_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, int64_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, MLFloat16, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, float, Mul)>,

      // Register Bitwise operators for model compatibility before official opset 18 schema.
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, int8_t, BitwiseAnd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, uint8_t, BitwiseAnd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, int16_t, BitwiseAnd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, uint16_t, BitwiseAnd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, int32_t, BitwiseAnd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, uint32_t, BitwiseAnd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, int64_t, BitwiseAnd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, uint64_t, BitwiseAnd)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, int8_t, BitwiseOr)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, uint8_t, BitwiseOr)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, int16_t, BitwiseOr)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, uint16_t, BitwiseOr)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, int32_t, BitwiseOr)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, uint32_t, BitwiseOr)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, int64_t, BitwiseOr)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, uint64_t, BitwiseOr)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, int8_t, BitwiseXor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, uint8_t, BitwiseXor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, int16_t, BitwiseXor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, uint16_t, BitwiseXor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, int32_t, BitwiseXor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, uint32_t, BitwiseXor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, int64_t, BitwiseXor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, uint64_t, BitwiseXor)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, int8_t, BitwiseNot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, uint8_t, BitwiseNot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, int16_t, BitwiseNot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, uint16_t, BitwiseNot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, int32_t, BitwiseNot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, uint32_t, BitwiseNot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, int64_t, BitwiseNot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, uint64_t, BitwiseNot)>,

      // Register BitwiseAnd operators for version 18+
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, int8_t, BitwiseAnd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, uint8_t, BitwiseAnd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, int16_t, BitwiseAnd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, uint16_t, BitwiseAnd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, int32_t, BitwiseAnd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, uint32_t, BitwiseAnd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, int64_t, BitwiseAnd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, uint64_t, BitwiseAnd)>,

      // Register BitwiseOr operators for version 18+
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, int8_t, BitwiseOr)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, uint8_t, BitwiseOr)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, int16_t, BitwiseOr)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, uint16_t, BitwiseOr)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, int32_t, BitwiseOr)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, uint32_t, BitwiseOr)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, int64_t, BitwiseOr)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, uint64_t, BitwiseOr)>,

      // Register BitwiseXor operators for version 18+
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, int8_t, BitwiseXor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, uint8_t, BitwiseXor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, int16_t, BitwiseXor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, uint16_t, BitwiseXor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, int32_t, BitwiseXor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, uint32_t, BitwiseXor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, int64_t, BitwiseXor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, uint64_t, BitwiseXor)>,

      // Register BitwiseNot operators for version 18+
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, int8_t, BitwiseNot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, uint8_t, BitwiseNot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, int16_t, BitwiseNot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, uint16_t, BitwiseNot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, int32_t, BitwiseNot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, uint32_t, BitwiseNot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, int64_t, BitwiseNot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, uint64_t, BitwiseNot)>,

      // Register Round operators for versions 11-21
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 21, MLFloat16, Round)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 21, float, Round)>,

      // Register Round operators for version 22+
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 22, MLFloat16, Round)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 22, float, Round)>,

#ifdef ENABLE_MUSA_NHWC_OPS
      // Register Mul operator for version 14 (NHWC)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 14, int32_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 14, int64_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 14, MLFloat16, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 14, float, Mul)>,
#endif

      // Register Div operator for version 14
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, int32_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, int64_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, MLFloat16, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, float, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, RealDiv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, float, RealDiv)>,

      // Register TensorFlow compatibility binary math operators
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, DivNoNan)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, float, DivNoNan)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, double, DivNoNan)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int32_t, SquaredDifference)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int64_t, SquaredDifference)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, SquaredDifference)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, float, SquaredDifference)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, double, SquaredDifference)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int32_t, FloorDiv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int64_t, FloorDiv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, FloorDiv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, float, FloorDiv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, double, FloorDiv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int32_t, FloorMod)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int64_t, FloorMod)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, FloorMod)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, float, FloorMod)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, double, FloorMod)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int32_t, Maximum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int64_t, Maximum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, Maximum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, float, Maximum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int32_t, Minimum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int64_t, Minimum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, Minimum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, float, Minimum)>,

      // Register Pow operators for version 12+
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 12, int32_t, Pow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 12, int64_t, Pow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 12, MLFloat16, Pow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 12, float, Pow)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Pow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Pow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Pow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, Pow)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, int32_t, Pow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, int64_t, Pow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, MLFloat16, Pow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, float, Pow)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 15, int32_t, Pow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 15, int64_t, Pow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 15, MLFloat16, Pow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 15, float, Pow)>,

      // Register versioned Mod operators (10-12) - only for MUSA-supported types
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 10, 12, int32_t, Mod)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 10, 12, int64_t, Mod)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
      //     kMusaExecutionProvider, kOnnxDomain, 10, 12, uint32_t, Mod)>,  // Disabled: muDNN Binary doesn't support unsigned types
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
      //     kMusaExecutionProvider, kOnnxDomain, 10, 12, uint64_t, Mod)>,  // Disabled: muDNN Binary doesn't support unsigned types
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 10, 12, float, Mod)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 10, 12, MLFloat16, Mod)>,

      // Register Mod operators for version 13+ - only for MUSA-supported types
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Mod)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Mod)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
      //     kMusaExecutionProvider, kOnnxDomain, 13, uint32_t, Mod)>,  // Disabled: muDNN Binary doesn't support unsigned types
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
      //     kMusaExecutionProvider, kOnnxDomain, 13, uint64_t, Mod)>,  // Disabled: muDNN Binary doesn't support unsigned types
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, Mod)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Mod)>,

      // Register Gemm operator for version 14
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, float, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, MLFloat16, Gemm)>,

      // Register CumSum operators
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 13, CumSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, CumSum)>,

      // Register Not operator
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, bool, Not)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, bool, And)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, bool, Or)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, bool, Xor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, bool, LogicalNot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, bool, LogicalAnd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, bool, LogicalOr)>,

      // Register MatMul operators - versioned (1-8) - Float types
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 8, MLFloat16, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 8, float, MatMul)>,

      // Register MatMul operators - versioned (9-12) - Float types
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, float, MatMul)>,

      // Register MatMul operators (version 13) - Float types
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, MatMul)>,

      // Register ReduceMax operators (version 1-17)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, MLFloat16, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, float, ReduceMax)>,

      // Register ReduceMax operators (version 18)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, MLFloat16, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, float, ReduceMax)>,

      // // Register ReduceMin operators (version 1-17)
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
      //     kMusaExecutionProvider, kOnnxDomain, 1, 17, MLFloat16, ReduceMin)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
      //     kMusaExecutionProvider, kOnnxDomain, 1, 17, float, ReduceMin)>,

      // // Register ReduceMin operators (version 18)
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
      //     kMusaExecutionProvider, kOnnxDomain, 18, MLFloat16, ReduceMin)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
      //     kMusaExecutionProvider, kOnnxDomain, 18, float, ReduceMin)>,

      // Register ReduceSum operators (version 1-12)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 12, MLFloat16, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 12, float, ReduceSum)>,

      // Register ReduceSum operators (version 13)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, ReduceSum)>,

      // Register ReduceSumSquare operators (version 1-17)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, MLFloat16, ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, float, ReduceSumSquare)>,

      // Register ReduceSumSquare operators (version 18)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, MLFloat16, ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, float, ReduceSumSquare)>,

      // Register ReduceMean operators (version 1-12)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 12, MLFloat16, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 12, float, ReduceMean)>,

      // Register ReduceMean operators (version 13)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, ReduceMean)>,

      // Register ReduceProd operators (version 1-17)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, MLFloat16, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, float, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, int32_t, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, int64_t, ReduceProd)>,

      // Register ReduceProd operators (version 18)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, MLFloat16, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, float, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, int32_t, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, int64_t, ReduceProd)>,

      // Register ReduceL2 operators (version 1-17)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, MLFloat16, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 17, float, ReduceL2)>,

      // Register ReduceL2 operators (version 18)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, MLFloat16, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, float, ReduceL2)>,

      // Register ArgMax operators (version 1-10) - Float types
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, float, ArgMax)>,

      // Register ArgMax operators (version 1-10) - Integer types
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, int32_t, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, int64_t, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, int8_t, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, uint8_t, ArgMax)>,

      // Register ArgMax operators (version 11-12) - Float types
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, float, ArgMax)>,

      // Register ArgMax operators (version 11-12) - Integer types
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, int32_t, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, int64_t, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, int8_t, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, uint8_t, ArgMax)>,

      // Register ArgMax operators (version 13) - Float types
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, ArgMax)>,

      // Register ArgMax operators (version 13) - Integer types
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int32_t, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int64_t, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int8_t, ArgMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, uint8_t, ArgMax)>,

      // Register ArgMin operators (version 1-10) - Float types
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, float, ArgMin)>,

      // Register ArgMin operators (version 1-10) - Integer types
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, int32_t, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, int64_t, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, int8_t, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, uint8_t, ArgMin)>,

      // Register ArgMin operators (version 11-12) - Float types
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, float, ArgMin)>,

      // Register ArgMin operators (version 11-12) - Integer types
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, int32_t, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, int64_t, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, int8_t, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, uint8_t, ArgMin)>,

      // Register ArgMin operators (version 13) - Float types
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, ArgMin)>,

      // Register ArgMin operators (version 13) - Integer types
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int32_t, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int64_t, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int8_t, ArgMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, uint8_t, ArgMin)>,

      // Register Transpose operators (version 1-12)
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, ReverseV2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, InvertPermutation)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 12, uint8_t, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 12, uint16_t, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 12, uint32_t, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 12, uint64_t, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 12, int8_t, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 12, int16_t, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 12, int32_t, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 12, int64_t, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 12, MLFloat16, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 12, float, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 12, bool, Transpose)>,
      // Register Transpose operators (version 13+)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, uint8_t, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, uint16_t, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, uint32_t, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, uint64_t, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int8_t, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int16_t, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, bool, Transpose)>,

      // Register Resize operators (version 10)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 10, 10, float, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 10, 10, MLFloat16, Resize)>,

      // Register Resize operators (version 11-12)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, float, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Resize)>,

      // Register Resize operators (version 13-17)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, 17, float, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, 17, MLFloat16, Resize)>,

      // Register Resize operators (version 18+)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, float, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, MLFloat16, Resize)>,

      // Register GridSample operators
      onnxruntime::contrib::musa::BuildKernelCreateInfo<onnxruntime::contrib::musa::ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSDomain, 1, float, GridSample)>,
      onnxruntime::contrib::musa::BuildKernelCreateInfo<onnxruntime::contrib::musa::ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSDomain, 1, MLFloat16, GridSample)>,
#ifdef ENABLE_MUSA_NHWC_OPS
      onnxruntime::contrib::musa::BuildKernelCreateInfo<onnxruntime::contrib::musa::ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 1, float, GridSample)>,
      onnxruntime::contrib::musa::BuildKernelCreateInfo<onnxruntime::contrib::musa::ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 1, MLFloat16, GridSample)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 16, float, GridSample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 16, MLFloat16, GridSample)>,
#ifdef ENABLE_MUSA_NHWC_OPS
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 16, float, GridSample)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 16, MLFloat16, GridSample)>,
#endif

      //   Register Gather operators (version 1-10)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, uint8_t, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, uint16_t, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, uint32_t, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, uint64_t, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, int8_t, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, int16_t, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, int32_t, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, int64_t, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, float, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, double, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, bool, Gather)>,

      // Register Gather operators (version 11-12)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, uint8_t, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, uint16_t, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, uint32_t, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, uint64_t, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, int8_t, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, int16_t, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, int32_t, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, int64_t, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, float, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, double, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, bool, Gather)>,

      // Register Gather operators (version 13)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, uint8_t, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, uint16_t, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, uint32_t, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, uint64_t, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int8_t, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int16_t, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, double, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, bool, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int32_t, GatherV2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int64_t, GatherV2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, GatherV2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, float, GatherV2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, double, GatherV2)>,

      // Register GatherElements operators (version 11+)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, uint8_t, GatherElements)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, uint16_t, GatherElements)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, uint32_t, GatherElements)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, uint64_t, GatherElements)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, int8_t, GatherElements)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, int16_t, GatherElements)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, int32_t, GatherElements)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, int64_t, GatherElements)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, MLFloat16, GatherElements)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, float, GatherElements)>,

      // Register GatherND operators (versions 11, 12, 13+)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 11, int64_t, GatherND)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 12, 12, int64_t, GatherND)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int64_t, GatherND)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int64_t, GatherNd)>,

      // Register ScatterND operators (versions 11-12, 13-15, 16-17, 18+)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, ScatterND)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, 15, ScatterND)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 16, 17, ScatterND)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, ScatterND)>,

      // Register NonZero operators (versions 9-12, 13+)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, bool, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, uint8_t, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, int32_t, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, int64_t, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, float, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, bool, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, uint8_t, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int32_t, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int64_t, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, double, NonZero)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, NonZero)>,

      // Register Unique operators (version 11+)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, float, Unique)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, double, Unique)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, int8_t, Unique)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, int32_t, Unique)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, int64_t, Unique)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, MLFloat16, Unique)>,

      // Register Unsqueeze operators (version 1-10)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, uint8_t, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, uint16_t, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, uint32_t, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, uint64_t, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, int8_t, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, int16_t, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, int32_t, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, int64_t, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, float, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, bool, Unsqueeze)>,

      // Register Unsqueeze operators (version 11-12)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, uint8_t, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, uint16_t, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, uint32_t, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, uint64_t, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, int8_t, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, int16_t, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, int32_t, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, int64_t, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, float, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, bool, Unsqueeze)>,

      // Register Unsqueeze operators (version 13+)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, uint8_t, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, uint16_t, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, uint32_t, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, uint64_t, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int8_t, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int16_t, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, bool, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, float, ExpandDims)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int32_t, ExpandDims)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int64_t, ExpandDims)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, ExpandDims)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, bool, ExpandDims)>,

      // Register Squeeze operators (version 1-10)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, uint8_t, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, uint16_t, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, uint32_t, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, uint64_t, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, int8_t, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, int16_t, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, int32_t, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, int64_t, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, float, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 10, bool, Squeeze)>,

      // Register Squeeze operators (version 11-12)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, uint8_t, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, uint16_t, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, uint32_t, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, uint64_t, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, int8_t, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, int16_t, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, int32_t, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, int64_t, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, float, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, bool, Squeeze)>,

      // Register Squeeze operators (version 13)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, uint8_t, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, uint16_t, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, uint32_t, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, uint64_t, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int8_t, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int16_t, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, bool, Squeeze)>,

      // Register Slice operators (version 1-9)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 9, float, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 9, MLFloat16, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 9, int32_t, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 9, int64_t, Slice)>,

      // Register Slice operators (version 10)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 10, 10, float, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 10, 10, MLFloat16, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 10, 10, int32_t, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 10, 10, int64_t, Slice)>,

      // Register Slice operators (version 11-12)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, float, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, int32_t, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, int64_t, Slice)>,

      // Register Slice operators (version 13)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Slice)>,

      // Register Split operators (version 2-10, 11-12, 13-17, 18)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 2, 10, Split)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, SplitV)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, Split)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, 17, Split)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, Split)>,

      // Register Tile operators (version 6-12, 13)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, float, Tile)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, int32_t, Tile)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, int64_t, Tile)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Tile)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, Tile)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Tile)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Tile)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Tile)>,

      // Register Concat operators (version 4-10)
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, ConcatV2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 4, 10, uint8_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 4, 10, uint16_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 4, 10, uint32_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 4, 10, uint64_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 4, 10, int8_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 4, 10, int16_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 4, 10, int32_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 4, 10, int64_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 4, 10, MLFloat16, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 4, 10, float, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 4, 10, bool, Concat)>,

#ifdef ENABLE_MUSA_NHWC_OPS
      // Register Concat operators (NHWC, version 4-10)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 4, 10, uint8_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 4, 10, uint16_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 4, 10, uint32_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 4, 10, uint64_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 4, 10, int8_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 4, 10, int16_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 4, 10, int32_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 4, 10, int64_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 4, 10, MLFloat16, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 4, 10, float, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 4, 10, bool, Concat)>,
#endif

      // Register Concat operators (version 11-12)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, uint8_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, uint16_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, uint32_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, uint64_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, int8_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, int16_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, int32_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, int64_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, float, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, bool, Concat)>,

#ifdef ENABLE_MUSA_NHWC_OPS
      // Register Concat operators (NHWC, version 11-12)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 11, 12, uint8_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 11, 12, uint16_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 11, 12, uint32_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 11, 12, uint64_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 11, 12, int8_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 11, 12, int16_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 11, 12, int32_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 11, 12, int64_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 11, 12, MLFloat16, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 11, 12, float, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 11, 12, bool, Concat)>,
#endif

      // Register Concat operators (version 13)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, uint8_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, uint16_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, uint32_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, uint64_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int8_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int16_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, bool, Concat)>,

#ifdef ENABLE_MUSA_NHWC_OPS
      // Register Concat operators (NHWC, version 13)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 13, uint8_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 13, uint16_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 13, uint32_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 13, uint64_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 13, int8_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 13, int16_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 13, int32_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 13, int64_t, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 13, MLFloat16, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 13, float, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kMSInternalNHWCDomain, 13, bool, Concat)>,
#endif

      // TODO(yichao.hu): fix Pad Kernel for 4D padding limitation
      // Register Pad operators (version 2-10)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 2, 10, float, Pad)>,
      // V0.17.5/V0.17.6.1: muDNN 3.1+ supports Pad fp16, conditionally registered.
      // Older SDKs (e.g. M1000 SDK 4.1.2 / muDNN 2.9.x) auto-fallback to CPU EP, same behavior as V0.17.1.
#if defined(MUDNN_VERSION) && (MUDNN_VERSION >= 3100)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 2, 10, MLFloat16, Pad)>,
#endif

      // Register Pad operators (version 11-17)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 17, float, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 17, float, PadV2)>,
#if defined(MUDNN_VERSION) && (MUDNN_VERSION >= 3100)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 17, MLFloat16, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 17, MLFloat16, PadV2)>,
#endif

      // Register Pad operators (version 18)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, float, Pad)>,
#if defined(MUDNN_VERSION) && (MUDNN_VERSION >= 3100)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 18, MLFloat16, Pad)>,
#endif

      // Register Flatten operators (version 1-8, 9-10, 11-12, 13)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 8, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 10, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 12, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, Flatten)>,

      // Register Reshape operators (version 1-4)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 4, int32_t, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 4, int64_t, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 4, MLFloat16, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 4, float, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 4, bool, Reshape)>,

      // Register Reshape operators (version 5-12)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 5, 12, int32_t, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 5, 12, int64_t, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 5, 12, MLFloat16, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 5, 12, float, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 5, 12, bool, Reshape)>,

      // Register Reshape operators (version 13)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, 13, int32_t, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, 13, int64_t, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, 13, float, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, 13, bool, Reshape)>,

      // Register Reshape operators (version 14-18)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, 18, int32_t, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, 18, int64_t, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, 18, MLFloat16, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, 18, float, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, 18, bool, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, 18, string, Reshape)>,

      // Register Reshape operators (version 19)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 19, int32_t, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 19, int64_t, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 19, MLFloat16, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 19, float, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 19, bool, Reshape)>,

      // Register Cast operators (version 6-8)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 8, int32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 8, int64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 8, MLFloat16, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 8, float, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 8, int8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 8, int16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 8, uint8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 8, uint16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 8, uint32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 8, uint64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 8, bool, Cast)>,

      // Register Cast operators (version 9-12)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, int32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, int64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, float, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, int8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, int16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, uint8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, uint16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, uint32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, uint64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 12, bool, Cast)>,

      // Register Cast operators (version 13-18)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, 18, int32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, 18, int64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, 18, MLFloat16, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, 18, float, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, 18, int8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, 18, int16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, 18, uint8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, 18, uint16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, 18, uint32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, 18, uint64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, 18, bool, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, 18, string, Cast)>,

      // Register Cast operators (version 19+)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 19, int32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 19, int64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 19, MLFloat16, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 19, float, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 19, int8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 19, int16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 19, uint8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 19, uint16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 19, uint32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 19, uint64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 19, bool, Cast)>,

      // Register Expand operators (version 8-12)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 8, 12, float, Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 8, 12, int32_t, Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 8, 12, int64_t, Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 8, 12, MLFloat16, Expand)>,

      // Register Expand operators (version 13)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, float, BroadcastTo)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int32_t, BroadcastTo)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int64_t, BroadcastTo)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, BroadcastTo)>,

      // Register Shape operators (multi-version support with CPU reuse)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 12, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, 14, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 15, 18, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 19, 20, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 21, 22, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 23, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, ShapeN)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, BroadcastGradientArgs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, ConcatOffset)>,

      // Register Size operators (multi-version support with CPU output scalar)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 12, Size)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, 20, Size)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 21, 22, Size)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 23, 24, Size)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 25, Size)>,

      // Register Where operators (version 9-15) - All supported types
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 15, float, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 15, MLFloat16, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 15, int32_t, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, 15, int64_t, Where)>,

      // Register Where operators (version 16) - All supported types
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 16, float, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 16, MLFloat16, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 16, int32_t, Where)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 16, int64_t, Where)>,

      // Register TensorFlow Select compatibility operators
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, float, Select)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, double, Select)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, Select)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int32_t, Select)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int64_t, Select)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, float, SelectV2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, double, SelectV2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, MLFloat16, SelectV2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int32_t, SelectV2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, int64_t, SelectV2)>,

      // Register Identity operators (tensor subset)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, 12, Identity)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, StopGradient)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, Snapshot)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, 13, Identity)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, 18, Identity)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 19, 20, Identity)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 21, 22, Identity)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 23, 24, Identity)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 25, Identity)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 1, IdentityN)>,

      // Register Dropout inference no-op operators
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 7, 9, Dropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 10, 11, Dropout)>,

      // Cast operators are now registered directly in cast_op.cc

      // Register Tile operators (version 6-12)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, float, Tile)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, int32_t, Tile)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, int64_t, Tile)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Tile)>,

      // Register Tile operators (version 13)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, float, Tile)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int32_t, Tile)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, int64_t, Tile)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Tile)>,

      // If control flow operators  - TEMPORARILY DISABLED for fallback testing
      //   BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 10, If)>,
      //   BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 12, If)>,
      //   BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 18, If)>,
      //   BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, If)>,

  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }
  return Status::OK();
}
}  // namespace musa

// Static kernel registry shared across all MUSA execution provider instances
static std::shared_ptr<KernelRegistry> s_kernel_registry;

namespace musa_ep {
void InitializeRegistry() {
  // Initialize MUSA device if needed
  // MUSA_CALL_THROW(musaInit(nullptr));

  s_kernel_registry = KernelRegistry::Create();
  ORT_THROW_IF_ERROR(musa::RegisterMusaKernels(*s_kernel_registry));
  ORT_THROW_IF_ERROR(contrib::musa::RegisterMusaContribKernels(*s_kernel_registry));
}

void DeleteRegistry() {
  s_kernel_registry.reset();

  // Finalize MUSA if needed
  // MUSA_CALL_THROW(musaFinalize());
}
}  // namespace musa_ep

std::shared_ptr<KernelRegistry>
MusaExecutionProvider::GetKernelRegistry() const {
  return s_kernel_registry;
}

std::unique_ptr<IDataTransfer> MusaExecutionProvider::GetDataTransfer() const {
  return std::make_unique<MusaDataTransfer>();
}

std::vector<AllocatorPtr> MusaExecutionProvider::CreatePreferredAllocators() {
  // Create MUSA device allocator
  AllocatorCreationInfo device_info(
      [](OrtDevice::DeviceId device_id) {
        return std::make_unique<MusaAllocator>(device_id, "MUSA");
      },
      info_.device_id,
      true  // use_arena
  );

  // Create MUSA pinned memory allocator for efficient host-device transfers
  AllocatorCreationInfo pinned_info(
      [](OrtDevice::DeviceId device_id) {
        return std::make_unique<MusaPinnedAllocator>(device_id, "MUSA_PINNED");
      },
      DEFAULT_CPU_ALLOCATOR_DEVICE_ID);

  return std::vector<AllocatorPtr>{
      CreateAllocator(device_info),
      CreateAllocator(pinned_info)};
}

// ============================================================================
// PerThreadContext Implementation
// ============================================================================

MusaExecutionProvider::PerThreadContext::PerThreadContext(
    OrtDevice::DeviceId device_id, musaStream_t stream, bool use_tf32, bool use_bf16) {
  // Set device (executed once per thread on first call)
  MUSA_CALL_THROW(musaSetDevice(device_id));

  // Create mudnn Handle bound to this EP's device.
  // The default muDNN Handle constructor binds to device 0 on SDK 4.3.x, so
  // current-device state alone is not enough for multi-device sessions.
  mudnn_handle_ = std::make_unique<::musa::dnn::Handle>(device_id);
  if (!mudnn_handle_) {
    ORT_THROW("Failed to create MUSA DNN handle");
  }

  // Critical: Associate stream with handle
  // Must call SetStream to ensure handle uses our managed stream,
  // otherwise it will use handle's internal default stream, causing cross-thread interference
  auto status = mudnn_handle_->SetStream(stream);
  if (status != ::musa::dnn::Status::SUCCESS) {
    ORT_THROW("Failed to set stream for MUSA DNN handle, status: " +
              std::to_string(static_cast<int>(status)));
  }

  // Default to strict FP32 for correctness-focused CPU vs MUSA parity.
  // Users can opt into TF32 through use_tf32=1. BF16 fast math uses explicit
  // muBLAS paths and should not also enable muDNN TF32 on FP32 ops.
  status = mudnn_handle_->SetAllowTF32(use_tf32 && !use_bf16);
  if (status != ::musa::dnn::Status::SUCCESS) {
    ORT_THROW("Failed to set TF32/BF16 mode for MUSA DNN handle, status: " +
              std::to_string(static_cast<int>(status)));
  }

  musa_graph_.SetStream(stream);
}

MusaExecutionProvider::PerThreadContext::~PerThreadContext() {
  // mudnn_handle_ is automatically cleaned up by unique_ptr
  // Note: Destruction order is important - handle must be cleaned up before stream is destroyed
  // This is guaranteed by PerThreadContext's shared_ptr lifetime and EP destruction order
}

bool MusaExecutionProvider::PerThreadContext::IsGraphCaptureAllowed(
    MusaGraphAnnotation_t musa_graph_annotation_id) const {
  if (!IsGraphCaptureAllowedOnRun(musa_graph_annotation_id)) {
    return false;
  }

  auto it = graph_id_to_run_count_.find(musa_graph_annotation_id);
  if (it == graph_id_to_run_count_.end()) {
    return false;
  }

  return it->second >= min_num_runs_before_musa_graph_capture_;
}

bool MusaExecutionProvider::PerThreadContext::IsGraphCaptureAllowedOnRun(
    MusaGraphAnnotation_t musa_graph_annotation_id) const {
  return musa_graph_.IsGraphCaptureAllowedOnRun(musa_graph_annotation_id);
}

MusaGraphAnnotation_t MusaExecutionProvider::PerThreadContext::GetMusaGraphAnnotationId(
    const RunOptions& run_options) const {
  auto graph_annotation_str =
      run_options.GetConfigOptions().GetConfigEntry(kOrtRunOptionsConfigCudaGraphAnnotation);
  MusaGraphAnnotation_t musa_graph_annotation_id = kMusaGraphAnnotationDefault;
  if (graph_annotation_str.has_value()) {
    ORT_ENFORCE(TryParseStringWithClassicLocale<int>(*graph_annotation_str, musa_graph_annotation_id),
                "Failed to parse the musa graph annotation id: ", *graph_annotation_str);
  }

  return musa_graph_annotation_id;
}

void MusaExecutionProvider::PerThreadContext::CaptureBegin(MusaGraphAnnotation_t musa_graph_annotation_id) {
  musa_graph_.CaptureBegin(musa_graph_annotation_id);
}

void MusaExecutionProvider::PerThreadContext::CaptureEnd(MusaGraphAnnotation_t musa_graph_annotation_id) {
  musa_graph_.CaptureEnd(musa_graph_annotation_id);
}

bool MusaExecutionProvider::PerThreadContext::IsGraphCaptured(MusaGraphAnnotation_t musa_graph_annotation_id) const {
  return musa_graph_.IsGraphCaptured(musa_graph_annotation_id);
}

Status MusaExecutionProvider::PerThreadContext::ReplayGraph(MusaGraphAnnotation_t musa_graph_annotation_id) {
  return musa_graph_.Replay(musa_graph_annotation_id);
}

void MusaExecutionProvider::PerThreadContext::IncrementRegularRunCountBeforeGraphCapture(
    MusaGraphAnnotation_t musa_graph_annotation_id) {
  ++graph_id_to_run_count_[musa_graph_annotation_id];
}

// ============================================================================
// GetPerThreadContext / ReleasePerThreadContext Implementation
// ============================================================================

MusaExecutionProvider::PerThreadContext& MusaExecutionProvider::GetPerThreadContext() const {
  // G5: Always set thread-local current device to this EP's device_id before any context use.
  // mudnn handle cached in PerThreadContext is bound to the device at creation time. When reusing
  // a cached context, current device may have been changed by another EP/session on the same thread.
  MUSA_CALL_THROW(musaSetDevice(info_.device_id));

  const auto& per_thread_context_cache = PerThreadContextCache();

  // Try to use cached context
  auto cached_context_it = per_thread_context_cache->find(this);
  if (cached_context_it != per_thread_context_cache->end()) {
    auto cached_context = cached_context_it->second.lock();
    ORT_ENFORCE(cached_context);
    return *cached_context;
  }

  // Get or create new context
  std::shared_ptr<PerThreadContext> context;
  {
    std::lock_guard<std::mutex> lock(context_state_.mutex);

    if (context_state_.retired_context_pool.empty()) {
      // Create new context
      context = std::make_shared<PerThreadContext>(info_.device_id, stream_, info_.use_tf32, info_.use_bf16);
    } else {
      // Reuse existing context
      context = context_state_.retired_context_pool.back();
      context_state_.retired_context_pool.pop_back();
    }

    // Insert into active_contexts
    const auto active_contexts_insert_result = context_state_.active_contexts.insert(context);
    ORT_ENFORCE(active_contexts_insert_result.second);

    // Record caches that need to be updated on destruction
    ORT_IGNORE_RETURN_VALUE(context_state_.caches_to_update_on_destruction.insert(per_thread_context_cache));
  }

  // Update thread local cache
  per_thread_context_cache->insert(std::make_pair(this, context));

  return *context;
}

void MusaExecutionProvider::ReleasePerThreadContext() const {
  const auto& per_thread_context_cache = PerThreadContextCache();

  auto cached_context_it = per_thread_context_cache->find(this);
  ORT_ENFORCE(cached_context_it != per_thread_context_cache->end());
  auto cached_context = cached_context_it->second.lock();
  ORT_ENFORCE(cached_context);

  {
    std::lock_guard<std::mutex> lock(context_state_.mutex);
    context_state_.active_contexts.erase(cached_context);
    context_state_.retired_context_pool.push_back(cached_context);
  }

  per_thread_context_cache->erase(cached_context_it);
}

// ============================================================================
// OnRunStart / OnRunEnd Implementation
// ============================================================================

Status MusaExecutionProvider::OnRunStart(const onnxruntime::RunOptions& run_options) {
  auto musa_graph_annotation_id = GetPerThreadContext().GetMusaGraphAnnotationId(run_options);
  if (IsGraphCaptureEnabled() && !GetPerThreadContext().IsGraphCaptured(musa_graph_annotation_id) &&
      GetPerThreadContext().IsGraphCaptureAllowed(musa_graph_annotation_id)) {
    LOGS(*GetLogger(), INFO) << "Capturing the musa graph for this model";
    GetPerThreadContext().CaptureBegin(musa_graph_annotation_id);
  }
  return Status::OK();
}

Status MusaExecutionProvider::OnRunEnd(bool sync_stream, const onnxruntime::RunOptions& run_options) {
  auto musa_graph_annotation_id = GetPerThreadContext().GetMusaGraphAnnotationId(run_options);
  if (IsGraphCaptureEnabled() && !GetPerThreadContext().IsGraphCaptured(musa_graph_annotation_id)) {
    if (GetPerThreadContext().IsGraphCaptureAllowed(musa_graph_annotation_id)) {
      GetPerThreadContext().CaptureEnd(musa_graph_annotation_id);
      // MUSA work issued to a capturing stream doesn't actually run on the GPU,
      // so run the captured graph here to actually execute the work.
      ORT_RETURN_IF_ERROR(GetPerThreadContext().ReplayGraph(musa_graph_annotation_id));
    } else {
      GetPerThreadContext().IncrementRegularRunCountBeforeGraphCapture(musa_graph_annotation_id);
    }
  }

  if (sync_stream) {
    MUSA_RETURN_IF_ERROR(musaStreamSynchronize(static_cast<musaStream_t>(stream_)));
  }

  if (!IsGraphCaptureEnabled() &&
      PerThreadContextCache()->find(this) != PerThreadContextCache()->end()) {
    ReleasePerThreadContext();
  }

  return Status::OK();
}

bool MusaExecutionProvider::IsGraphCaptureEnabled() const {
  return enable_musa_graph_;
}

bool MusaExecutionProvider::IsGraphCaptured(int graph_annotation_id) const {
  return GetPerThreadContext().IsGraphCaptured(graph_annotation_id);
}

Status MusaExecutionProvider::ReplayGraph(int graph_annotation_id) {
  return GetPerThreadContext().ReplayGraph(graph_annotation_id);
}

// ============================================================================
// Constructor / Destructor
// ============================================================================

MusaExecutionProvider::MusaExecutionProvider(
    const MusaExecutionProviderInfo& info)
    : IExecutionProvider{kMusaExecutionProvider, OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NONE, info.device_id)}, info_(info) {
  InitProviderOrtApi();
  enable_musa_graph_ = info.enable_musa_graph;

  int device_count = 0;
  MUSA_CALL_THROW(musaGetDeviceCount(&device_count));
  if (info_.device_id < 0 || info_.device_id >= device_count) {
    ORT_THROW("Invalid MUSA device ID: ", info_.device_id,
              ", must be in [0, ", device_count, ")");
  }

  // Set device
  MUSA_CALL_THROW(musaSetDevice(info_.device_id));

  // Create EP-level unified stream
  if (use_ep_level_unified_stream_) {
    MUSA_CALL_THROW(musaStreamCreateWithFlags(&stream_, musaStreamNonBlocking));
  }
}

MusaExecutionProvider::~MusaExecutionProvider() {
  // Clean up all references in thread local caches
  {
    std::lock_guard<std::mutex> lock(context_state_.mutex);
    for (const auto& weak_cache : context_state_.caches_to_update_on_destruction) {
      if (auto cache = weak_cache.lock()) {
        cache->erase(this);
      }
    }
  }

  // Clean up stream
  if (stream_ != nullptr) {
    // Ignore errors as device may be unavailable during destruction
    ORT_IGNORE_RETURN_VALUE(musaStreamDestroy(stream_));
    stream_ = nullptr;
  }
}

void MusaExecutionProvider::RegisterStreamHandlers(IStreamCommandHandleRegistry& stream_handle_registry,
                                                   AllocatorMap& allocator_map) const {
  RegisterMusaStreamHandles(stream_handle_registry,
                            OrtDevice::GPU,
                            stream_,
                            use_ep_level_unified_stream_);
}


namespace {

constexpr const char* kMusaTokenMixerResidualOpName = "MusaTokenMixerResidual";
constexpr const char* kMusaReshapeMatMulOpName = "MusaReshapeMatMul";
constexpr const char* kMusaFeatureNormOpName = "MusaFeatureNorm";
constexpr const char* kMusaGeluOpName = "MusaGelu";
constexpr const char* kMusaPlnCascadeBlockOpName = "MusaPlnCascadeBlock";
constexpr const char* kMusaLayerNormLastDimOpName = "MusaLayerNormLastDim";


bool ReadScalarFloatInitializer(const GraphViewer& graph_viewer,
                                const std::string& name,
                                float& value) {
  const auto* tensor = graph_viewer.GetConstantInitializer(name, true);
  if (tensor == nullptr || tensor->data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    return false;
  }
  const auto& raw = tensor->raw_data();
  if (raw.size() != sizeof(float)) {
    return false;
  }
  std::memcpy(&value, raw.data(), sizeof(float));
  return true;
}

bool IsScalarFloatInitializerNear(const GraphViewer& graph_viewer,
                                  const std::string& name,
                                  float expected,
                                  float tolerance = 1e-4f) {
  float value = 0.0f;
  return ReadScalarFloatInitializer(graph_viewer, name, value) && std::fabs(value - expected) <= tolerance;
}

bool ReadIntShapeInitializer(const GraphViewer& graph_viewer,
                             const std::string& name,
                             std::vector<int64_t>& values) {
  const auto* tensor = graph_viewer.GetConstantInitializer(name, true);
  if (tensor == nullptr) {
    return false;
  }

  const auto data_type = tensor->data_type();
  const auto& raw = tensor->raw_data();

  values.clear();
  if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
    if (!raw.empty()) {
      if (raw.size() % sizeof(int64_t) != 0) {
        return false;
      }
      values.resize(raw.size() / sizeof(int64_t));
      std::memcpy(values.data(), raw.data(), raw.size());
      return true;
    }

    return false;
  }

  if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT32) {
    if (!raw.empty()) {
      if (raw.size() % sizeof(int32_t) != 0) {
        return false;
      }
      const size_t count = raw.size() / sizeof(int32_t);
      values.resize(count);
      for (size_t i = 0; i < count; ++i) {
        int32_t value = 0;
        std::memcpy(&value, raw.data() + i * sizeof(int32_t), sizeof(int32_t));
        values[i] = static_cast<int64_t>(value);
      }
      return true;
    }

    return false;
  }

  return false;
}

bool GetNodeInputName(const Node& node, size_t index, std::string& name) {
  const auto inputs = node.InputDefs();
  if (index >= inputs.size() || inputs[index] == nullptr) {
    return false;
  }
  name = inputs[index]->Name();
  return true;
}

bool GetNodeOutputName(const Node& node, size_t index, std::string& name) {
  const auto outputs = node.OutputDefs();
  if (index >= outputs.size() || outputs[index] == nullptr) {
    return false;
  }
  name = outputs[index]->Name();
  return true;
}

int CountConsumers(const GraphViewer& graph_viewer, const std::string& value_name) {
  int count = 0;
  for (auto& node_index : graph_viewer.GetNodesInTopologicalOrder()) {
    const auto* node = graph_viewer.GetNode(node_index);
    if (node == nullptr) {
      continue;
    }

    for (const auto* input : node->InputDefs()) {
      if (input != nullptr && input->Name() == value_name) {
        ++count;
      }
    }
  }
  return count;
}

bool HasSingleConsumer(const GraphViewer& graph_viewer, const std::string& value_name) {
  return CountConsumers(graph_viewer, value_name) == 1;
}

bool HasElementType(const NodeArg* node_arg, int32_t elem_type) {
  if (node_arg == nullptr) {
    return false;
  }
  const auto* type_proto = node_arg->TypeAsProto();
  return type_proto != nullptr && type_proto->has_tensor_type() &&
         type_proto->tensor_type().has_elem_type() &&
         type_proto->tensor_type().elem_type() == elem_type;
}

bool IsSupportedTokenMixerType(const NodeArg* node_arg) {
  return HasElementType(node_arg, ONNX_NAMESPACE::TensorProto_DataType_FLOAT) ||
         HasElementType(node_arg, ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
}

bool IsPermute0213(const Node& transpose) {
  if (transpose.OpType() != "Transpose") {
    return false;
  }
  const auto& attrs = transpose.GetAttributes();
  if (attrs.count("perm") == 0) {
    return false;
  }
  const auto& perm = attrs.at("perm");
  return perm.ints_size() == 4 && perm.ints(0) == 0 && perm.ints(1) == 2 &&
         perm.ints(2) == 1 && perm.ints(3) == 3;
}

bool GetIntAttribute(const Node& node, const std::string& name, int64_t& value) {
  const auto& attrs = node.GetAttributes();
  if (attrs.count(name) == 0 || attrs.at(name).type() != ONNX_NAMESPACE::AttributeProto_AttributeType_INT) {
    return false;
  }
  value = attrs.at(name).i();
  return true;
}

bool IsCastTo(const Node& node, int64_t to_type) {
  if (node.OpType() != "Cast") {
    return false;
  }
  int64_t value = 0;
  return GetIntAttribute(node, "to", value) && value == to_type;
}

int32_t GetElementType(const NodeArg* node_arg) {
  if (node_arg == nullptr) {
    return ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
  }
  const auto* type_proto = node_arg->TypeAsProto();
  if (type_proto == nullptr || !type_proto->has_tensor_type() ||
      !type_proto->tensor_type().has_elem_type()) {
    return ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
  }
  return type_proto->tensor_type().elem_type();
}

void AddIntAttribute(NodeAttributes& attributes,
                     const std::string& name,
                     int64_t value) {
  auto attr = ONNX_NAMESPACE::AttributeProto::Create();
  attr->set_name(name);
  attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr->set_i(value);
  attributes[name] = *attr;
}

void AddFloatAttribute(NodeAttributes& attributes,
                       const std::string& name,
                       float value) {
  auto attr = ONNX_NAMESPACE::AttributeProto::Create();
  attr->set_name(name);
  attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
  attr->set_f(value);
  attributes[name] = *attr;
}

bool GetFloatAttribute(const Node& node, const std::string& name, float& value) {
  const auto& attrs = node.GetAttributes();
  if (attrs.count(name) == 0 || attrs.at(name).type() != ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT) {
    return false;
  }
  value = attrs.at(name).f();
  return true;
}

bool HasIntListAttribute(const Node& node, const std::string& name, std::initializer_list<int64_t> expected) {
  const auto& attrs = node.GetAttributes();
  if (attrs.count(name) == 0 || attrs.at(name).type() != ONNX_NAMESPACE::AttributeProto_AttributeType_INTS) {
    return false;
  }
  const auto& attr = attrs.at(name);
  if (static_cast<size_t>(attr.ints_size()) != expected.size()) {
    return false;
  }
  size_t i = 0;
  for (int64_t value : expected) {
    if (attr.ints(static_cast<int>(i++)) != value) {
      return false;
    }
  }
  return true;
}

bool HasIntAttributeValue(const Node& node, const std::string& name, int64_t expected) {
  int64_t value = 0;
  return GetIntAttribute(node, name, value) && value == expected;
}

bool IsPermute021(const Node& transpose) {
  return transpose.OpType() == "Transpose" && HasIntListAttribute(transpose, "perm", {0, 2, 1});
}

bool IsFloatInitializer(const GraphViewer& graph_viewer, const std::string& name) {
  const auto* tensor = graph_viewer.GetConstantInitializer(name, true);
  return tensor != nullptr && tensor->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
}

const Node* ProducerOfInput(const GraphViewer& graph_viewer, const Node& node, size_t input_index, std::string& input_name) {
  if (!GetNodeInputName(node, input_index, input_name)) {
    return nullptr;
  }
  return graph_viewer.GetProducerNode(input_name);
}

bool OutputHasExternalConsumer(const GraphViewer& graph_viewer,
                               const std::string& value_name,
                               const std::unordered_set<NodeIndex>& subgraph_nodes,
                               const std::string& allowed_external_output) {
  if (value_name == allowed_external_output) {
    return false;
  }
  for (auto& node_index : graph_viewer.GetNodesInTopologicalOrder()) {
    const auto* node = graph_viewer.GetNode(node_index);
    if (node == nullptr || subgraph_nodes.count(node->Index()) != 0) {
      continue;
    }
    for (const auto* input : node->InputDefs()) {
      if (input != nullptr && input->Name() == value_name) {
        return true;
      }
    }
  }
  return false;
}

bool CollectProducerSubgraph(const GraphViewer& graph_viewer,
                             const std::string& value_name,
                             const std::unordered_set<std::string>& boundary_inputs,
                             std::unordered_set<NodeIndex>& node_indices) {
  if (boundary_inputs.count(value_name) != 0 || graph_viewer.GetConstantInitializer(value_name, true) != nullptr) {
    return true;
  }
  const Node* producer = graph_viewer.GetProducerNode(value_name);
  if (producer == nullptr) {
    return false;
  }
  if (node_indices.insert(producer->Index()).second) {
    for (const auto* input : producer->InputDefs()) {
      if (input != nullptr && !CollectProducerSubgraph(graph_viewer, input->Name(), boundary_inputs, node_indices)) {
        return false;
      }
    }
  }
  return true;
}

bool ValidateNoExternalConsumers(const GraphViewer& graph_viewer,
                                 const std::unordered_set<NodeIndex>& node_indices,
                                 const std::string& final_output) {
  for (NodeIndex node_index : node_indices) {
    const auto* node = graph_viewer.GetNode(node_index);
    if (node == nullptr) {
      return false;
    }
    for (const auto* output : node->OutputDefs()) {
      if (output != nullptr && OutputHasExternalConsumer(graph_viewer, output->Name(), node_indices, final_output)) {
        return false;
      }
    }
  }
  return true;
}

bool IsFeatureNormMean(const Node& reduce_mean, const std::string& bn_input_name) {
  if (reduce_mean.OpType() != "ReduceMean" || reduce_mean.InputDefs().empty()) {
    return false;
  }
  std::string input_name;
  return GetNodeInputName(reduce_mean, 0, input_name) && input_name == bn_input_name &&
         HasIntListAttribute(reduce_mean, "axes", {0, 2, 3}) &&
         HasIntAttributeValue(reduce_mean, "keepdims", 1);
}

bool IsFeatureNormVariance(const GraphViewer& graph_viewer,
                           const Node& div,
                           const std::string& bn_input_name,
                           const std::string& reduce_mean_output) {
  if (div.OpType() != "Div" || div.InputDefs().size() != 2) {
    return false;
  }
  std::string sumsq_name;
  const Node* sumsq = ProducerOfInput(graph_viewer, div, 0, sumsq_name);
  if (sumsq == nullptr || sumsq->OpType() != "ReduceSumSquare" ||
      !HasIntListAttribute(*sumsq, "axes", {0, 2, 3}) ||
      !HasIntAttributeValue(*sumsq, "keepdims", 0)) {
    return false;
  }
  std::string sub_name;
  const Node* sub = ProducerOfInput(graph_viewer, *sumsq, 0, sub_name);
  if (sub == nullptr || sub->OpType() != "Sub" || sub->InputDefs().size() != 2) {
    return false;
  }
  std::string sub0;
  std::string sub1;
  return GetNodeInputName(*sub, 0, sub0) && GetNodeInputName(*sub, 1, sub1) &&
         ((sub0 == bn_input_name && sub1 == reduce_mean_output) ||
          (sub1 == bn_input_name && sub0 == reduce_mean_output));
}

bool IsReduceMeanAxis1Keepdims0(const Node& reduce_mean) {
  return reduce_mean.OpType() == "ReduceMean" &&
         HasIntListAttribute(reduce_mean, "axes", {1}) &&
         HasIntAttributeValue(reduce_mean, "keepdims", 0);
}

bool TryGetInitializerInput(const GraphViewer& graph_viewer,
                            const Node& node,
                            std::string& initializer_name,
                            std::string& other_input_name) {
  if (node.InputDefs().size() != 2) {
    return false;
  }
  std::string input0;
  std::string input1;
  if (!GetNodeInputName(node, 0, input0) || !GetNodeInputName(node, 1, input1)) {
    return false;
  }
  if (IsFloatInitializer(graph_viewer, input0)) {
    initializer_name = input0;
    other_input_name = input1;
    return true;
  }
  if (IsFloatInitializer(graph_viewer, input1)) {
    initializer_name = input1;
    other_input_name = input0;
    return true;
  }
  return false;
}

std::unique_ptr<ComputeCapability> TryCreateLayerNormLastDimCapability(
    const GraphViewer& graph_viewer,
    const Node& final_add) {
  if (final_add.OpType() != "Add" || final_add.InputDefs().size() != 2 ||
      final_add.OutputDefs().size() != 1) {
    return nullptr;
  }

  std::string final_output;
  if (!GetNodeOutputName(final_add, 0, final_output)) {
    return nullptr;
  }

  std::string beta_name;
  std::string gamma_mul_output;
  if (!TryGetInitializerInput(graph_viewer, final_add, beta_name, gamma_mul_output)) {
    return nullptr;
  }

  const Node* gamma_mul = graph_viewer.GetProducerNode(gamma_mul_output);
  if (gamma_mul == nullptr || gamma_mul->OpType() != "Mul") {
    return nullptr;
  }

  std::string gamma_name;
  std::string div_output;
  if (!TryGetInitializerInput(graph_viewer, *gamma_mul, gamma_name, div_output)) {
    return nullptr;
  }

  const Node* div = graph_viewer.GetProducerNode(div_output);
  if (div == nullptr || div->OpType() != "Div" || div->InputDefs().size() != 2) {
    return nullptr;
  }

  std::string sub_output;
  std::string clip_output;
  if (!GetNodeInputName(*div, 0, sub_output) || !GetNodeInputName(*div, 1, clip_output)) {
    return nullptr;
  }

  const Node* sub = graph_viewer.GetProducerNode(sub_output);
  const Node* clip_max = graph_viewer.GetProducerNode(clip_output);
  if (sub == nullptr || sub->OpType() != "Sub" || sub->InputDefs().size() != 2 ||
      clip_max == nullptr || clip_max->OpType() != "Max") {
    return nullptr;
  }

  std::string clip_min_output;
  std::string clip_min_name;
  if (!TryGetInitializerInput(graph_viewer, *clip_max, clip_min_name, clip_min_output)) {
    return nullptr;
  }
  float clip_min_value = 0.0f;
  if (!ReadScalarFloatInitializer(graph_viewer, clip_min_name, clip_min_value)) {
    return nullptr;
  }

  const Node* clip_min = graph_viewer.GetProducerNode(clip_min_output);
  if (clip_min == nullptr || clip_min->OpType() != "Min") {
    return nullptr;
  }

  std::string sqrt_output;
  std::string clip_max_name;
  if (!TryGetInitializerInput(graph_viewer, *clip_min, clip_max_name, sqrt_output)) {
    return nullptr;
  }
  float clip_max_value = 0.0f;
  if (!ReadScalarFloatInitializer(graph_viewer, clip_max_name, clip_max_value)) {
    return nullptr;
  }

  const Node* sqrt = graph_viewer.GetProducerNode(sqrt_output);
  if (sqrt == nullptr || sqrt->OpType() != "Sqrt" || sqrt->InputDefs().size() != 1) {
    return nullptr;
  }

  std::string var_unsqueeze_output;
  if (!GetNodeInputName(*sqrt, 0, var_unsqueeze_output)) {
    return nullptr;
  }
  const Node* var_unsqueeze = graph_viewer.GetProducerNode(var_unsqueeze_output);
  if (var_unsqueeze == nullptr || var_unsqueeze->OpType() != "Unsqueeze" ||
      var_unsqueeze->InputDefs().empty()) {
    return nullptr;
  }

  std::string var_mean_output;
  if (!GetNodeInputName(*var_unsqueeze, 0, var_mean_output)) {
    return nullptr;
  }
  const Node* var_mean = graph_viewer.GetProducerNode(var_mean_output);
  if (var_mean == nullptr || !IsReduceMeanAxis1Keepdims0(*var_mean) ||
      var_mean->InputDefs().empty()) {
    return nullptr;
  }

  std::string square_output;
  if (!GetNodeInputName(*var_mean, 0, square_output)) {
    return nullptr;
  }
  const Node* square = graph_viewer.GetProducerNode(square_output);
  if (square == nullptr || square->OpType() != "Mul" || square->InputDefs().size() != 2) {
    return nullptr;
  }

  std::string square_input0;
  std::string square_input1;
  if (!GetNodeInputName(*square, 0, square_input0) ||
      !GetNodeInputName(*square, 1, square_input1) ||
      square_input0 != sub_output || square_input1 != sub_output) {
    return nullptr;
  }

  std::string sub_input0;
  std::string sub_input1;
  if (!GetNodeInputName(*sub, 0, sub_input0) || !GetNodeInputName(*sub, 1, sub_input1)) {
    return nullptr;
  }

  const Node* mean_unsqueeze = graph_viewer.GetProducerNode(sub_input1);
  if (mean_unsqueeze == nullptr || mean_unsqueeze->OpType() != "Unsqueeze" ||
      mean_unsqueeze->InputDefs().empty()) {
    return nullptr;
  }

  std::string mean_output;
  if (!GetNodeInputName(*mean_unsqueeze, 0, mean_output)) {
    return nullptr;
  }
  const Node* mean = graph_viewer.GetProducerNode(mean_output);
  if (mean == nullptr || !IsReduceMeanAxis1Keepdims0(*mean) ||
      mean->InputDefs().empty()) {
    return nullptr;
  }

  std::string input_name;
  if (!GetNodeInputName(*mean, 0, input_name) || input_name != sub_input0) {
    return nullptr;
  }

  const auto input_type = GetElementType(mean->InputDefs()[0]);
  if (input_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    return nullptr;
  }

  std::unordered_set<std::string> boundary_inputs = {input_name, gamma_name, beta_name};
  std::unordered_set<NodeIndex> node_index_set;
  if (!CollectProducerSubgraph(graph_viewer, final_output, boundary_inputs, node_index_set) ||
      !ValidateNoExternalConsumers(graph_viewer, node_index_set, final_output)) {
    return nullptr;
  }

  std::vector<NodeIndex> node_indices(node_index_set.begin(), node_index_set.end());
  std::sort(node_indices.begin(), node_indices.end());

  auto sub_graph = IndexedSubGraph::Create();
  for (auto node_index : node_indices) {
    sub_graph->Nodes().push_back(node_index);
  }

  auto meta_def = IndexedSubGraph_MetaDef::Create();
  meta_def->name() = kMusaLayerNormLastDimOpName;
  meta_def->domain() = kMSDomain;
  meta_def->since_version() = 1;
  meta_def->inputs().push_back(input_name);
  meta_def->inputs().push_back(gamma_name);
  meta_def->inputs().push_back(beta_name);
  meta_def->outputs().push_back(final_output);
  AddFloatAttribute(meta_def->attributes(), "clip_min", clip_min_value);
  AddFloatAttribute(meta_def->attributes(), "clip_max", clip_max_value);
  sub_graph->SetMetaDef(std::move(meta_def));
  return ComputeCapability::Create(std::move(sub_graph));
}


bool TryMatchMulByScalar(const GraphViewer& graph_viewer,
                         const Node& mul,
                         float expected_scalar,
                         std::string& data_input_name) {
  if (mul.OpType() != "Mul" || mul.InputDefs().size() != 2) {
    return false;
  }
  std::string input0;
  std::string input1;
  if (!GetNodeInputName(mul, 0, input0) || !GetNodeInputName(mul, 1, input1)) {
    return false;
  }
  if (IsScalarFloatInitializerNear(graph_viewer, input0, expected_scalar)) {
    data_input_name = input1;
    return true;
  }
  if (IsScalarFloatInitializerNear(graph_viewer, input1, expected_scalar)) {
    data_input_name = input0;
    return true;
  }
  return false;
}

bool TryMatchGeluFactor(const GraphViewer& graph_viewer,
                        const Node& add,
                        const std::string& expected_input_name,
                        const Node*& erf,
                        const Node*& scale_mul) {
  if (add.OpType() != "Add" || add.InputDefs().size() != 2) {
    return false;
  }
  std::string add_input0;
  std::string add_input1;
  if (!GetNodeInputName(add, 0, add_input0) || !GetNodeInputName(add, 1, add_input1)) {
    return false;
  }
  std::string erf_output;
  if (IsScalarFloatInitializerNear(graph_viewer, add_input0, 1.0f)) {
    erf_output = add_input1;
  } else if (IsScalarFloatInitializerNear(graph_viewer, add_input1, 1.0f)) {
    erf_output = add_input0;
  } else {
    return false;
  }

  erf = graph_viewer.GetProducerNode(erf_output);
  if (erf == nullptr || erf->OpType() != "Erf" || erf->InputDefs().size() != 1) {
    return false;
  }

  std::string scale_output;
  if (!GetNodeInputName(*erf, 0, scale_output)) {
    return false;
  }
  scale_mul = graph_viewer.GetProducerNode(scale_output);
  if (scale_mul == nullptr) {
    return false;
  }

  std::string scaled_input;
  return TryMatchMulByScalar(graph_viewer, *scale_mul, 0.70710678f, scaled_input) &&
         scaled_input == expected_input_name;
}

std::unique_ptr<ComputeCapability> TryCreateGeluCapability(
    const GraphViewer& graph_viewer,
    const Node& final_mul) {
  if (final_mul.OpType() != "Mul" || final_mul.InputDefs().size() != 2 || final_mul.OutputDefs().size() != 1) {
    return nullptr;
  }
  std::string final_input0;
  std::string final_input1;
  std::string final_output;
  if (!GetNodeInputName(final_mul, 0, final_input0) || !GetNodeInputName(final_mul, 1, final_input1) ||
      !GetNodeOutputName(final_mul, 0, final_output)) {
    return nullptr;
  }

  const Node* half_mul = nullptr;
  const Node* factor_add = nullptr;
  std::string gelu_input;
  auto try_order = [&](const std::string& half_output, const std::string& factor_output) -> bool {
    half_mul = graph_viewer.GetProducerNode(half_output);
    factor_add = graph_viewer.GetProducerNode(factor_output);
    return half_mul != nullptr && factor_add != nullptr &&
           TryMatchMulByScalar(graph_viewer, *half_mul, 0.5f, gelu_input);
  };
  if (!try_order(final_input0, final_input1) && !try_order(final_input1, final_input0)) {
    return nullptr;
  }

  const Node* erf = nullptr;
  const Node* scale_mul = nullptr;
  if (!TryMatchGeluFactor(graph_viewer, *factor_add, gelu_input, erf, scale_mul)) {
    return nullptr;
  }

  const auto half_outputs = half_mul->OutputDefs();
  const auto factor_outputs = factor_add->OutputDefs();
  const auto erf_outputs = erf->OutputDefs();
  const auto scale_outputs = scale_mul->OutputDefs();
  if (half_outputs.empty() || factor_outputs.empty() || erf_outputs.empty() || scale_outputs.empty() ||
      !HasSingleConsumer(graph_viewer, half_outputs[0]->Name()) ||
      !HasSingleConsumer(graph_viewer, factor_outputs[0]->Name()) ||
      !HasSingleConsumer(graph_viewer, erf_outputs[0]->Name()) ||
      !HasSingleConsumer(graph_viewer, scale_outputs[0]->Name())) {
    return nullptr;
  }

  const auto input_type = GetElementType(half_mul->InputDefs()[0]->Name() == gelu_input
                                             ? half_mul->InputDefs()[0]
                                             : half_mul->InputDefs()[1]);
  if (input_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    return nullptr;
  }

  std::vector<NodeIndex> node_indices = {
      scale_mul->Index(), erf->Index(), factor_add->Index(), half_mul->Index(), final_mul.Index()};
  std::sort(node_indices.begin(), node_indices.end());
  node_indices.erase(std::unique(node_indices.begin(), node_indices.end()), node_indices.end());

  auto sub_graph = IndexedSubGraph::Create();
  for (auto node_index : node_indices) {
    sub_graph->Nodes().push_back(node_index);
  }

  auto meta_def = IndexedSubGraph_MetaDef::Create();
  meta_def->name() = kMusaGeluOpName;
  meta_def->domain() = kMSDomain;
  meta_def->since_version() = 1;
  meta_def->inputs().push_back(gelu_input);
  meta_def->outputs().push_back(final_output);
  sub_graph->SetMetaDef(std::move(meta_def));
  return ComputeCapability::Create(std::move(sub_graph));
}

std::unique_ptr<ComputeCapability> TryCreateFeatureNormCapability(
    const GraphViewer& graph_viewer,
    const Node& final_add) {
  if (final_add.OpType() != "Add" || final_add.InputDefs().size() != 2 || final_add.OutputDefs().size() != 1) {
    return nullptr;
  }

  std::string final_output;
  if (!GetNodeOutputName(final_add, 0, final_output)) {
    return nullptr;
  }

  std::string add_input0;
  std::string add_input1;
  if (!GetNodeInputName(final_add, 0, add_input0) || !GetNodeInputName(final_add, 1, add_input1)) {
    return nullptr;
  }

  const Node* gamma_mul = nullptr;
  std::string beta_name;
  if (IsFloatInitializer(graph_viewer, add_input0)) {
    beta_name = add_input0;
    gamma_mul = graph_viewer.GetProducerNode(add_input1);
  } else if (IsFloatInitializer(graph_viewer, add_input1)) {
    beta_name = add_input1;
    gamma_mul = graph_viewer.GetProducerNode(add_input0);
  } else {
    return nullptr;
  }
  if (gamma_mul == nullptr || gamma_mul->OpType() != "Mul" || gamma_mul->InputDefs().size() != 2) {
    return nullptr;
  }

  std::string mul_input0;
  std::string mul_input1;
  if (!GetNodeInputName(*gamma_mul, 0, mul_input0) || !GetNodeInputName(*gamma_mul, 1, mul_input1)) {
    return nullptr;
  }

  const Node* final_reshape = nullptr;
  std::string gamma_name;
  if (IsFloatInitializer(graph_viewer, mul_input0)) {
    gamma_name = mul_input0;
    final_reshape = graph_viewer.GetProducerNode(mul_input1);
  } else if (IsFloatInitializer(graph_viewer, mul_input1)) {
    gamma_name = mul_input1;
    final_reshape = graph_viewer.GetProducerNode(mul_input0);
  } else {
    return nullptr;
  }
  if (final_reshape == nullptr || final_reshape->OpType() != "Reshape" || final_reshape->InputDefs().size() != 2) {
    return nullptr;
  }

  std::string bn_output;
  const Node* batch_norm = ProducerOfInput(graph_viewer, *final_reshape, 0, bn_output);
  if (batch_norm == nullptr || batch_norm->OpType() != "BatchNormalization" || batch_norm->InputDefs().size() < 5) {
    return nullptr;
  }
  int64_t training_mode = 0;
  if (GetIntAttribute(*batch_norm, "training_mode", training_mode) && training_mode != 0) {
    return nullptr;
  }
  float epsilon = 1e-5f;
  (void)GetFloatAttribute(*batch_norm, "epsilon", epsilon);

  std::string bn_input_name;
  const Node* input_reshape = ProducerOfInput(graph_viewer, *batch_norm, 0, bn_input_name);
  if (input_reshape == nullptr || input_reshape->OpType() != "Reshape" || input_reshape->InputDefs().size() != 2) {
    return nullptr;
  }

  std::string original_input;
  if (!GetNodeInputName(*input_reshape, 0, original_input)) {
    return nullptr;
  }

  std::string mean_name;
  const Node* squeeze_mean = ProducerOfInput(graph_viewer, *batch_norm, 3, mean_name);
  if (squeeze_mean == nullptr || squeeze_mean->OpType() != "Squeeze" || squeeze_mean->InputDefs().empty()) {
    return nullptr;
  }
  std::string reduce_mean_output;
  const Node* reduce_mean = ProducerOfInput(graph_viewer, *squeeze_mean, 0, reduce_mean_output);
  if (reduce_mean == nullptr || !IsFeatureNormMean(*reduce_mean, bn_input_name)) {
    return nullptr;
  }

  std::string var_name;
  const Node* var_div = ProducerOfInput(graph_viewer, *batch_norm, 4, var_name);
  if (var_div == nullptr || !IsFeatureNormVariance(graph_viewer, *var_div, bn_input_name, reduce_mean_output)) {
    return nullptr;
  }

  const auto input_type = GetElementType(input_reshape->InputDefs()[0]);
  if (input_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    return nullptr;
  }

  std::unordered_set<std::string> boundary_inputs = {original_input, gamma_name, beta_name};
  std::unordered_set<NodeIndex> node_index_set;
  if (!CollectProducerSubgraph(graph_viewer, final_output, boundary_inputs, node_index_set) ||
      !ValidateNoExternalConsumers(graph_viewer, node_index_set, final_output)) {
    return nullptr;
  }

  std::vector<NodeIndex> node_indices(node_index_set.begin(), node_index_set.end());
  std::sort(node_indices.begin(), node_indices.end());

  auto sub_graph = IndexedSubGraph::Create();
  for (auto node_index : node_indices) {
    sub_graph->Nodes().push_back(node_index);
  }

  auto meta_def = IndexedSubGraph_MetaDef::Create();
  meta_def->name() = kMusaFeatureNormOpName;
  meta_def->domain() = kMSDomain;
  meta_def->since_version() = 1;
  meta_def->inputs().push_back(original_input);
  meta_def->inputs().push_back(gamma_name);
  meta_def->inputs().push_back(beta_name);
  meta_def->outputs().push_back(final_output);
  AddFloatAttribute(meta_def->attributes(), "epsilon", epsilon);
  sub_graph->SetMetaDef(std::move(meta_def));
  return ComputeCapability::Create(std::move(sub_graph));
}

std::unique_ptr<ComputeCapability> TryCreateReshapeMatMulCapability(
    const GraphViewer& graph_viewer,
    const Node& final_reshape) {
  if (final_reshape.OpType() != "Reshape" || final_reshape.InputDefs().size() != 2 ||
      final_reshape.OutputDefs().size() != 1) {
    return nullptr;
  }

  std::string matmul_output;
  std::string final_shape_name;
  std::string final_output;
  if (!GetNodeInputName(final_reshape, 0, matmul_output) ||
      !GetNodeInputName(final_reshape, 1, final_shape_name) ||
      !GetNodeOutputName(final_reshape, 0, final_output)) {
    return nullptr;
  }

  const Node* matmul = graph_viewer.GetProducerNode(matmul_output);
  const Node* cast_to_i64 = graph_viewer.GetProducerNode(final_shape_name);
  if (matmul == nullptr || cast_to_i64 == nullptr || matmul->OpType() != "MatMul" ||
      !IsCastTo(*cast_to_i64, ONNX_NAMESPACE::TensorProto_DataType_INT64)) {
    return nullptr;
  }

  std::string reshape0_output;
  std::string weight_name;
  if (!GetNodeInputName(*matmul, 0, reshape0_output) || !GetNodeInputName(*matmul, 1, weight_name)) {
    return nullptr;
  }

  const Node* reshape0 = graph_viewer.GetProducerNode(reshape0_output);
  if (reshape0 == nullptr || reshape0->OpType() != "Reshape") {
    return nullptr;
  }

  std::string input_name;
  std::string reshape0_shape_name;
  if (!GetNodeInputName(*reshape0, 0, input_name) || !GetNodeInputName(*reshape0, 1, reshape0_shape_name)) {
    return nullptr;
  }

  std::vector<int64_t> reshape0_shape;
  if (!ReadIntShapeInitializer(graph_viewer, reshape0_shape_name, reshape0_shape) ||
      reshape0_shape.size() != 2 || reshape0_shape[0] != -1 || reshape0_shape[1] <= 0) {
    return nullptr;
  }

  std::string concat_output;
  if (!GetNodeInputName(*cast_to_i64, 0, concat_output)) {
    return nullptr;
  }
  const Node* concat = graph_viewer.GetProducerNode(concat_output);
  int64_t concat_axis = -1;
  if (concat == nullptr || concat->OpType() != "Concat" ||
      !GetIntAttribute(*concat, "axis", concat_axis) || concat_axis != 0 ||
      concat->InputDefs().size() != 2) {
    return nullptr;
  }

  std::string gather_output;
  std::string concat_const_name;
  if (!GetNodeInputName(*concat, 0, gather_output) || !GetNodeInputName(*concat, 1, concat_const_name)) {
    return nullptr;
  }

  std::vector<int64_t> concat_const;
  if (!ReadIntShapeInitializer(graph_viewer, concat_const_name, concat_const) ||
      concat_const.size() != 1 || concat_const[0] <= 0) {
    return nullptr;
  }

  const Node* gather = graph_viewer.GetProducerNode(gather_output);
  int64_t gather_axis = -1;
  if (gather == nullptr || gather->OpType() != "Gather" ||
      !GetIntAttribute(*gather, "axis", gather_axis) || gather_axis != 0 ||
      gather->InputDefs().size() != 2) {
    return nullptr;
  }

  std::string cast32_output;
  std::string gather_indices_name;
  if (!GetNodeInputName(*gather, 0, cast32_output) || !GetNodeInputName(*gather, 1, gather_indices_name)) {
    return nullptr;
  }

  std::vector<int64_t> gather_indices;
  if (!ReadIntShapeInitializer(graph_viewer, gather_indices_name, gather_indices) ||
      gather_indices.size() != 2 || gather_indices[0] != 0 || gather_indices[1] != 1) {
    return nullptr;
  }

  const Node* cast_to_i32 = graph_viewer.GetProducerNode(cast32_output);
  if (cast_to_i32 == nullptr || !IsCastTo(*cast_to_i32, ONNX_NAMESPACE::TensorProto_DataType_INT32)) {
    return nullptr;
  }

  std::string shape_output;
  if (!GetNodeInputName(*cast_to_i32, 0, shape_output)) {
    return nullptr;
  }
  const Node* shape = graph_viewer.GetProducerNode(shape_output);
  if (shape == nullptr || shape->OpType() != "Shape" || shape->InputDefs().size() != 1) {
    return nullptr;
  }

  std::string shape_input;
  if (!GetNodeInputName(*shape, 0, shape_input) || shape_input != input_name) {
    return nullptr;
  }

  const auto* weight_tensor = graph_viewer.GetConstantInitializer(weight_name, true);
  if (weight_tensor == nullptr || weight_tensor->dims_size() != 2 ||
      weight_tensor->dims()[0] != reshape0_shape[1] || weight_tensor->dims()[1] != concat_const[0]) {
    return nullptr;
  }

  const auto reshape0_outputs = reshape0->OutputDefs();
  const auto matmul_outputs = matmul->OutputDefs();
  const auto shape_outputs = shape->OutputDefs();
  const auto cast32_outputs = cast_to_i32->OutputDefs();
  const auto gather_outputs = gather->OutputDefs();
  const auto concat_outputs = concat->OutputDefs();
  const auto cast64_outputs = cast_to_i64->OutputDefs();
  if (reshape0_outputs.empty() || matmul_outputs.empty() || shape_outputs.empty() ||
      cast32_outputs.empty() || gather_outputs.empty() || concat_outputs.empty() || cast64_outputs.empty() ||
      !HasSingleConsumer(graph_viewer, reshape0_outputs[0]->Name()) ||
      !HasSingleConsumer(graph_viewer, matmul_outputs[0]->Name()) ||
      !HasSingleConsumer(graph_viewer, shape_outputs[0]->Name()) ||
      !HasSingleConsumer(graph_viewer, cast32_outputs[0]->Name()) ||
      !HasSingleConsumer(graph_viewer, gather_outputs[0]->Name()) ||
      !HasSingleConsumer(graph_viewer, concat_outputs[0]->Name()) ||
      !HasSingleConsumer(graph_viewer, cast64_outputs[0]->Name())) {
    return nullptr;
  }

  const auto input_type = GetElementType(reshape0->InputDefs()[0]);
  const auto weight_type = GetElementType(matmul->InputDefs()[1]);
  if ((input_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
       input_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) ||
      input_type != weight_type) {
    return nullptr;
  }

  std::vector<NodeIndex> node_indices = {
      shape->Index(), cast_to_i32->Index(), reshape0->Index(), matmul->Index(),
      gather->Index(), concat->Index(), cast_to_i64->Index(), final_reshape.Index()};
  std::sort(node_indices.begin(), node_indices.end());
  node_indices.erase(std::unique(node_indices.begin(), node_indices.end()), node_indices.end());

  auto sub_graph = IndexedSubGraph::Create();
  for (auto node_index : node_indices) {
    sub_graph->Nodes().push_back(node_index);
  }

  auto meta_def = IndexedSubGraph_MetaDef::Create();
  meta_def->name() = kMusaReshapeMatMulOpName;
  meta_def->domain() = kMSDomain;
  meta_def->since_version() = 1;
  meta_def->inputs().push_back(input_name);
  meta_def->inputs().push_back(weight_name);
  meta_def->outputs().push_back(final_output);
  AddIntAttribute(meta_def->attributes(), "transpose_b", 0);
  sub_graph->SetMetaDef(std::move(meta_def));
  return ComputeCapability::Create(std::move(sub_graph));
}

std::unique_ptr<ComputeCapability> TryCreateTranspose021ReshapeMatMulCapability(
    const GraphViewer& graph_viewer,
    const Node& final_reshape) {
  if (final_reshape.OpType() != "Reshape" || final_reshape.InputDefs().size() != 2 ||
      final_reshape.OutputDefs().size() != 1) {
    return nullptr;
  }

  std::string matmul_output;
  std::string final_shape_name;
  std::string final_output;
  if (!GetNodeInputName(final_reshape, 0, matmul_output) ||
      !GetNodeInputName(final_reshape, 1, final_shape_name) ||
      !GetNodeOutputName(final_reshape, 0, final_output)) {
    return nullptr;
  }

  const Node* matmul = graph_viewer.GetProducerNode(matmul_output);
  const Node* cast_to_i64 = graph_viewer.GetProducerNode(final_shape_name);
  if (matmul == nullptr || cast_to_i64 == nullptr || matmul->OpType() != "MatMul" ||
      !IsCastTo(*cast_to_i64, ONNX_NAMESPACE::TensorProto_DataType_INT64)) {
    return nullptr;
  }

  std::string reshape0_output;
  std::string weight_name;
  if (!GetNodeInputName(*matmul, 0, reshape0_output) || !GetNodeInputName(*matmul, 1, weight_name)) {
    return nullptr;
  }

  const Node* reshape0 = graph_viewer.GetProducerNode(reshape0_output);
  if (reshape0 == nullptr || reshape0->OpType() != "Reshape" || reshape0->InputDefs().size() != 2) {
    return nullptr;
  }

  std::string transpose_output;
  std::string reshape0_shape_name;
  if (!GetNodeInputName(*reshape0, 0, transpose_output) ||
      !GetNodeInputName(*reshape0, 1, reshape0_shape_name)) {
    return nullptr;
  }

  const Node* transpose = graph_viewer.GetProducerNode(transpose_output);
  if (transpose == nullptr || !IsPermute021(*transpose) || transpose->InputDefs().size() != 1) {
    return nullptr;
  }

  std::string input_name;
  if (!GetNodeInputName(*transpose, 0, input_name)) {
    return nullptr;
  }

  std::vector<int64_t> reshape0_shape;
  if (!ReadIntShapeInitializer(graph_viewer, reshape0_shape_name, reshape0_shape) ||
      reshape0_shape.size() != 2 || reshape0_shape[0] != -1 || reshape0_shape[1] <= 0) {
    return nullptr;
  }

  std::string concat_output;
  if (!GetNodeInputName(*cast_to_i64, 0, concat_output)) {
    return nullptr;
  }
  const Node* concat = graph_viewer.GetProducerNode(concat_output);
  int64_t concat_axis = -1;
  if (concat == nullptr || concat->OpType() != "Concat" ||
      !GetIntAttribute(*concat, "axis", concat_axis) || concat_axis != 0 ||
      concat->InputDefs().size() != 2) {
    return nullptr;
  }

  std::string gather_prefix_output;
  std::string concat_const_name;
  if (!GetNodeInputName(*concat, 0, gather_prefix_output) ||
      !GetNodeInputName(*concat, 1, concat_const_name)) {
    return nullptr;
  }

  std::vector<int64_t> concat_const;
  if (!ReadIntShapeInitializer(graph_viewer, concat_const_name, concat_const) ||
      concat_const.size() != 1 || concat_const[0] <= 0) {
    return nullptr;
  }

  const Node* gather_prefix = graph_viewer.GetProducerNode(gather_prefix_output);
  int64_t gather_prefix_axis = -1;
  if (gather_prefix == nullptr || gather_prefix->OpType() != "Gather" ||
      !GetIntAttribute(*gather_prefix, "axis", gather_prefix_axis) || gather_prefix_axis != 0 ||
      gather_prefix->InputDefs().size() != 2) {
    return nullptr;
  }

  std::string cast32_output;
  std::string gather_prefix_indices_name;
  if (!GetNodeInputName(*gather_prefix, 0, cast32_output) ||
      !GetNodeInputName(*gather_prefix, 1, gather_prefix_indices_name)) {
    return nullptr;
  }

  std::vector<int64_t> gather_prefix_indices;
  if (!ReadIntShapeInitializer(graph_viewer, gather_prefix_indices_name, gather_prefix_indices) ||
      gather_prefix_indices.size() != 2 || gather_prefix_indices[0] != 0 || gather_prefix_indices[1] != 1) {
    return nullptr;
  }

  const Node* cast_to_i32 = graph_viewer.GetProducerNode(cast32_output);
  if (cast_to_i32 == nullptr || !IsCastTo(*cast_to_i32, ONNX_NAMESPACE::TensorProto_DataType_INT32)) {
    return nullptr;
  }

  std::string reordered_shape_output;
  if (!GetNodeInputName(*cast_to_i32, 0, reordered_shape_output)) {
    return nullptr;
  }
  const Node* reorder_gather = graph_viewer.GetProducerNode(reordered_shape_output);
  int64_t reorder_axis = -1;
  if (reorder_gather == nullptr || reorder_gather->OpType() != "Gather" ||
      !GetIntAttribute(*reorder_gather, "axis", reorder_axis) || reorder_axis != 0 ||
      reorder_gather->InputDefs().size() != 2) {
    return nullptr;
  }

  std::string shape_output;
  std::string reorder_indices_name;
  if (!GetNodeInputName(*reorder_gather, 0, shape_output) ||
      !GetNodeInputName(*reorder_gather, 1, reorder_indices_name)) {
    return nullptr;
  }

  std::vector<int64_t> reorder_indices;
  if (!ReadIntShapeInitializer(graph_viewer, reorder_indices_name, reorder_indices) ||
      reorder_indices.size() != 3 || reorder_indices[0] != 0 ||
      reorder_indices[1] != 2 || reorder_indices[2] != 1) {
    return nullptr;
  }

  const Node* shape = graph_viewer.GetProducerNode(shape_output);
  if (shape == nullptr || shape->OpType() != "Shape" || shape->InputDefs().size() != 1) {
    return nullptr;
  }

  std::string shape_input;
  if (!GetNodeInputName(*shape, 0, shape_input) || shape_input != input_name) {
    return nullptr;
  }

  const auto* weight_tensor = graph_viewer.GetConstantInitializer(weight_name, true);
  if (weight_tensor == nullptr || weight_tensor->dims_size() != 2 ||
      weight_tensor->dims()[0] != reshape0_shape[1] || weight_tensor->dims()[1] != concat_const[0]) {
    return nullptr;
  }

  const auto input_type = GetElementType(transpose->InputDefs()[0]);
  const auto weight_type = GetElementType(matmul->InputDefs()[1]);
  if (input_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
      weight_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    return nullptr;
  }

  const auto transpose_outputs = transpose->OutputDefs();
  const auto reshape0_outputs = reshape0->OutputDefs();
  const auto matmul_outputs = matmul->OutputDefs();
  const auto shape_outputs = shape->OutputDefs();
  const auto reorder_outputs = reorder_gather->OutputDefs();
  const auto cast32_outputs = cast_to_i32->OutputDefs();
  const auto gather_prefix_outputs = gather_prefix->OutputDefs();
  const auto concat_outputs = concat->OutputDefs();
  const auto cast64_outputs = cast_to_i64->OutputDefs();
  if (transpose_outputs.empty() || reshape0_outputs.empty() || matmul_outputs.empty() ||
      shape_outputs.empty() || reorder_outputs.empty() || cast32_outputs.empty() ||
      gather_prefix_outputs.empty() || concat_outputs.empty() || cast64_outputs.empty() ||
      !HasSingleConsumer(graph_viewer, transpose_outputs[0]->Name()) ||
      !HasSingleConsumer(graph_viewer, reshape0_outputs[0]->Name()) ||
      !HasSingleConsumer(graph_viewer, matmul_outputs[0]->Name()) ||
      !HasSingleConsumer(graph_viewer, shape_outputs[0]->Name()) ||
      !HasSingleConsumer(graph_viewer, reorder_outputs[0]->Name()) ||
      !HasSingleConsumer(graph_viewer, cast32_outputs[0]->Name()) ||
      !HasSingleConsumer(graph_viewer, gather_prefix_outputs[0]->Name()) ||
      !HasSingleConsumer(graph_viewer, concat_outputs[0]->Name()) ||
      !HasSingleConsumer(graph_viewer, cast64_outputs[0]->Name())) {
    return nullptr;
  }

  std::vector<NodeIndex> node_indices = {
      shape->Index(), reorder_gather->Index(), cast_to_i32->Index(), gather_prefix->Index(),
      concat->Index(), cast_to_i64->Index(), transpose->Index(), reshape0->Index(),
      matmul->Index(), final_reshape.Index()};
  std::sort(node_indices.begin(), node_indices.end());
  node_indices.erase(std::unique(node_indices.begin(), node_indices.end()), node_indices.end());

  auto sub_graph = IndexedSubGraph::Create();
  for (auto node_index : node_indices) {
    sub_graph->Nodes().push_back(node_index);
  }

  auto meta_def = IndexedSubGraph_MetaDef::Create();
  meta_def->name() = kMusaReshapeMatMulOpName;
  meta_def->domain() = kMSDomain;
  meta_def->since_version() = 1;
  meta_def->inputs().push_back(input_name);
  meta_def->inputs().push_back(weight_name);
  meta_def->outputs().push_back(final_output);
  AddIntAttribute(meta_def->attributes(), "transpose_b", 0);
  AddIntAttribute(meta_def->attributes(), "transpose_input_021", 1);
  sub_graph->SetMetaDef(std::move(meta_def));
  return ComputeCapability::Create(std::move(sub_graph));
}

struct PlnCascadeBranch {
  const Node* mask_mul = nullptr;
  std::string mask_name;
  std::string data_name;
};

struct PlnCascadeStepMatch {
  const Node* select_add = nullptr;
  const Node* candidate_scale_mul = nullptr;
  const Node* candidate_add = nullptr;
  const Node* candidate_mask_mul = nullptr;
  const Node* passthrough_mask_mul = nullptr;
  std::string previous_value;
  std::string candidate_mask;
  std::string passthrough_mask;
  std::string add_values;
  std::string bias_values;
};

bool IsPlnSelectName(const std::string& name) {
  return name.find("/pln") != std::string::npos && name.find("Select") != std::string::npos;
}

void EnumerateMaskedBranches(const Node& mul,
                             std::vector<PlnCascadeBranch>& branches) {
  if (mul.OpType() != "Mul" || mul.InputDefs().size() != 2) {
    return;
  }
  std::string input0;
  std::string input1;
  if (!GetNodeInputName(mul, 0, input0) || !GetNodeInputName(mul, 1, input1)) {
    return;
  }
  branches.push_back(PlnCascadeBranch{&mul, input0, input1});
  branches.push_back(PlnCascadeBranch{&mul, input1, input0});
}

bool TryParsePlnAffineCandidate(const GraphViewer& graph_viewer,
                                const std::string& candidate_value,
                                const std::string& passthrough_value,
                                PlnCascadeStepMatch& match) {
  const Node* candidate_add = graph_viewer.GetProducerNode(candidate_value);
  if (candidate_add == nullptr || candidate_add->OpType() != "Add" ||
      candidate_add->InputDefs().size() != 2) {
    return false;
  }

  std::string add_input0;
  std::string add_input1;
  if (!GetNodeInputName(*candidate_add, 0, add_input0) ||
      !GetNodeInputName(*candidate_add, 1, add_input1)) {
    return false;
  }

  auto try_parse = [&](const std::string& mul_output,
                       const std::string& bias_name) -> bool {
    if (!IsFloatInitializer(graph_viewer, bias_name)) {
      return false;
    }

    const Node* scale_mul = graph_viewer.GetProducerNode(mul_output);
    if (scale_mul == nullptr || scale_mul->OpType() != "Mul" ||
        scale_mul->InputDefs().size() != 2) {
      return false;
    }

    std::string mul_input0;
    std::string mul_input1;
    if (!GetNodeInputName(*scale_mul, 0, mul_input0) ||
        !GetNodeInputName(*scale_mul, 1, mul_input1)) {
      return false;
    }

    std::string add_values;
    if (mul_input0 == passthrough_value && IsFloatInitializer(graph_viewer, mul_input1)) {
      add_values = mul_input1;
    } else if (mul_input1 == passthrough_value && IsFloatInitializer(graph_viewer, mul_input0)) {
      add_values = mul_input0;
    } else {
      return false;
    }

    match.candidate_scale_mul = scale_mul;
    match.candidate_add = candidate_add;
    match.add_values = add_values;
    match.bias_values = bias_name;
    match.previous_value = passthrough_value;
    return true;
  };

  return try_parse(add_input0, add_input1) || try_parse(add_input1, add_input0);
}

bool TryParsePlnCascadeStep(const GraphViewer& graph_viewer,
                            const Node& select_add,
                            PlnCascadeStepMatch& match) {
  if (select_add.OpType() != "Add" || select_add.InputDefs().size() != 2 ||
      select_add.OutputDefs().size() != 1 || !IsPlnSelectName(select_add.Name())) {
    return false;
  }

  if (GetElementType(select_add.OutputDefs()[0]) != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    return false;
  }

  std::string input0;
  std::string input1;
  if (!GetNodeInputName(select_add, 0, input0) || !GetNodeInputName(select_add, 1, input1)) {
    return false;
  }
  const Node* mul0 = graph_viewer.GetProducerNode(input0);
  const Node* mul1 = graph_viewer.GetProducerNode(input1);
  if (mul0 == nullptr || mul1 == nullptr || mul0->OpType() != "Mul" || mul1->OpType() != "Mul") {
    return false;
  }

  std::vector<PlnCascadeBranch> left_branches;
  std::vector<PlnCascadeBranch> right_branches;
  EnumerateMaskedBranches(*mul0, left_branches);
  EnumerateMaskedBranches(*mul1, right_branches);

  for (const auto& left : left_branches) {
    for (const auto& right : right_branches) {
      PlnCascadeStepMatch local;
      if (TryParsePlnAffineCandidate(graph_viewer, left.data_name, right.data_name, local)) {
        local.select_add = &select_add;
        local.candidate_mask_mul = left.mask_mul;
        local.passthrough_mask_mul = right.mask_mul;
        local.candidate_mask = left.mask_name;
        local.passthrough_mask = right.mask_name;
        match = std::move(local);
        return true;
      }

      local = PlnCascadeStepMatch{};
      if (TryParsePlnAffineCandidate(graph_viewer, right.data_name, left.data_name, local)) {
        local.select_add = &select_add;
        local.candidate_mask_mul = right.mask_mul;
        local.passthrough_mask_mul = left.mask_mul;
        local.candidate_mask = right.mask_name;
        local.passthrough_mask = left.mask_name;
        match = std::move(local);
        return true;
      }
    }
  }

  return false;
}

bool HasPlnCascadeContinuation(const GraphViewer& graph_viewer,
                               const std::string& value_name) {
  for (auto& node_index : graph_viewer.GetNodesInTopologicalOrder()) {
    const auto* node = graph_viewer.GetNode(node_index);
    if (node == nullptr) {
      continue;
    }

    PlnCascadeStepMatch step;
    if (TryParsePlnCascadeStep(graph_viewer, *node, step) &&
        step.previous_value == value_name) {
      return true;
    }
  }
  return false;
}

std::unique_ptr<ComputeCapability> TryCreatePlnCascadeBlockCapability(
    const GraphViewer& graph_viewer,
    const Node& final_add) {
  std::string final_output;
  if (!GetNodeOutputName(final_add, 0, final_output)) {
    return nullptr;
  }

  std::vector<PlnCascadeStepMatch> reverse_steps;
  const Node* current = &final_add;
  std::string base_value;
  while (current != nullptr && reverse_steps.size() < 16) {
    PlnCascadeStepMatch step;
    if (!TryParsePlnCascadeStep(graph_viewer, *current, step)) {
      break;
    }

    base_value = step.previous_value;
    reverse_steps.push_back(std::move(step));
    current = graph_viewer.GetProducerNode(base_value);
  }

  if (reverse_steps.size() < 2 || base_value.empty()) {
    return nullptr;
  }

  if (HasPlnCascadeContinuation(graph_viewer, final_output)) {
    return nullptr;
  }

  std::vector<PlnCascadeStepMatch> steps(reverse_steps.rbegin(), reverse_steps.rend());
  std::unordered_set<NodeIndex> node_index_set;
  for (const auto& step : steps) {
    node_index_set.insert(step.candidate_scale_mul->Index());
    node_index_set.insert(step.candidate_add->Index());
    node_index_set.insert(step.select_add->Index());
  }
  for (const auto& step : steps) {
    const auto candidate_mask_outputs = step.candidate_mask_mul->OutputDefs();
    if (candidate_mask_outputs.empty() ||
        !OutputHasExternalConsumer(graph_viewer, candidate_mask_outputs[0]->Name(),
                                   node_index_set, final_output)) {
      node_index_set.insert(step.candidate_mask_mul->Index());
    }
    const auto passthrough_mask_outputs = step.passthrough_mask_mul->OutputDefs();
    if (passthrough_mask_outputs.empty() ||
        !OutputHasExternalConsumer(graph_viewer, passthrough_mask_outputs[0]->Name(),
                                   node_index_set, final_output)) {
      node_index_set.insert(step.passthrough_mask_mul->Index());
    }
  }

  if (!ValidateNoExternalConsumers(graph_viewer, node_index_set, final_output)) {
    return nullptr;
  }

  std::vector<NodeIndex> node_indices(node_index_set.begin(), node_index_set.end());
  std::sort(node_indices.begin(), node_indices.end());

  auto sub_graph = IndexedSubGraph::Create();
  for (auto node_index : node_indices) {
    sub_graph->Nodes().push_back(node_index);
  }

  auto meta_def = IndexedSubGraph_MetaDef::Create();
  meta_def->name() = kMusaPlnCascadeBlockOpName;
  meta_def->domain() = kMSDomain;
  meta_def->since_version() = 1;
  meta_def->inputs().push_back(base_value);
  for (const auto& step : steps) {
    meta_def->inputs().push_back(step.candidate_mask);
    meta_def->inputs().push_back(step.passthrough_mask);
    meta_def->inputs().push_back(step.add_values);
    meta_def->inputs().push_back(step.bias_values);
  }
  meta_def->outputs().push_back(final_output);
  AddIntAttribute(meta_def->attributes(), "num_steps", static_cast<int64_t>(steps.size()));
  sub_graph->SetMetaDef(std::move(meta_def));
  return ComputeCapability::Create(std::move(sub_graph));
}

std::unique_ptr<ComputeCapability> TryCreateTokenMixerResidualCapability(
    const GraphViewer& graph_viewer,
    const Node& add_node,
    size_t permuted_input_index,
    size_t residual_input_index) {
  std::string permuted_value;
  std::string residual_value;
  std::string add_output;
  if (!GetNodeInputName(add_node, permuted_input_index, permuted_value) ||
      !GetNodeInputName(add_node, residual_input_index, residual_value) ||
      !GetNodeOutputName(add_node, 0, add_output)) {
    return nullptr;
  }

  const Node* reshape2 = graph_viewer.GetProducerNode(permuted_value);
  const Node* reshape0 = graph_viewer.GetProducerNode(residual_value);
  if (reshape2 == nullptr || reshape0 == nullptr ||
      reshape2->OpType() != "Reshape" || reshape0->OpType() != "Reshape") {
    return nullptr;
  }

  std::string reshape2_data;
  std::string reshape2_shape_name;
  std::string reshape0_data;
  std::string reshape0_shape_name;
  std::string reshape0_output;
  if (!GetNodeInputName(*reshape2, 0, reshape2_data) ||
      !GetNodeInputName(*reshape2, 1, reshape2_shape_name) ||
      !GetNodeInputName(*reshape0, 0, reshape0_data) ||
      !GetNodeInputName(*reshape0, 1, reshape0_shape_name) ||
      !GetNodeOutputName(*reshape0, 0, reshape0_output)) {
    return nullptr;
  }

  const Node* transpose = graph_viewer.GetProducerNode(reshape2_data);
  if (transpose == nullptr || !IsPermute0213(*transpose)) {
    return nullptr;
  }

  std::string transpose_data;
  std::string transpose_output;
  if (!GetNodeInputName(*transpose, 0, transpose_data) ||
      !GetNodeOutputName(*transpose, 0, transpose_output)) {
    return nullptr;
  }

  const Node* reshape1 = graph_viewer.GetProducerNode(transpose_data);
  if (reshape1 == nullptr || reshape1->OpType() != "Reshape") {
    return nullptr;
  }

  std::string reshape1_data;
  std::string reshape1_shape_name;
  std::string reshape1_output;
  if (!GetNodeInputName(*reshape1, 0, reshape1_data) ||
      !GetNodeInputName(*reshape1, 1, reshape1_shape_name) ||
      !GetNodeOutputName(*reshape1, 0, reshape1_output)) {
    return nullptr;
  }

  if (reshape1_data != reshape0_output || reshape2_data != transpose_output ||
      transpose_data != reshape1_output) {
    return nullptr;
  }

  if (CountConsumers(graph_viewer, reshape0_output) != 2 ||
      !HasSingleConsumer(graph_viewer, reshape1_output) ||
      !HasSingleConsumer(graph_viewer, transpose_output) ||
      !HasSingleConsumer(graph_viewer, permuted_value)) {
    return nullptr;
  }

  std::vector<int64_t> shape0;
  std::vector<int64_t> shape1;
  std::vector<int64_t> shape2;
  if (!ReadIntShapeInitializer(graph_viewer, reshape0_shape_name, shape0) ||
      !ReadIntShapeInitializer(graph_viewer, reshape1_shape_name, shape1) ||
      !ReadIntShapeInitializer(graph_viewer, reshape2_shape_name, shape2)) {
    return nullptr;
  }

  if (shape0.size() != 3 || shape1.size() != 4 || shape2.size() != 3) {
    return nullptr;
  }

  const int64_t num_t = shape1[1];
  const int64_t num_h = shape1[2];
  const int64_t d_k = shape1[3];
  if (num_t <= 0 || num_h <= 0 || d_k <= 0 || num_t != num_h) {
    return nullptr;
  }

  if (shape0[1] != num_t || shape0[2] != num_h * d_k ||
      shape2[1] != num_h || shape2[2] != num_t * d_k) {
    return nullptr;
  }

  const auto reshape0_inputs = reshape0->InputDefs();
  if (reshape0_inputs.empty() || !IsSupportedTokenMixerType(reshape0_inputs[0])) {
    return nullptr;
  }

  auto sub_graph = IndexedSubGraph::Create();
  sub_graph->Nodes().push_back(reshape0->Index());
  sub_graph->Nodes().push_back(reshape1->Index());
  sub_graph->Nodes().push_back(transpose->Index());
  sub_graph->Nodes().push_back(reshape2->Index());
  sub_graph->Nodes().push_back(add_node.Index());

  auto meta_def = IndexedSubGraph_MetaDef::Create();
  meta_def->name() = kMusaTokenMixerResidualOpName;
  meta_def->domain() = kMSDomain;
  meta_def->since_version() = 1;
  meta_def->inputs().push_back(reshape0_data);
  meta_def->outputs().push_back(add_output);
  AddIntAttribute(meta_def->attributes(), "num_T", num_t);
  AddIntAttribute(meta_def->attributes(), "num_H", num_h);
  AddIntAttribute(meta_def->attributes(), "d_k", d_k);
  sub_graph->SetMetaDef(std::move(meta_def));
  return ComputeCapability::Create(std::move(sub_graph));
}

std::unique_ptr<ComputeCapability> TryCreateTokenMixerResidualCapability(
    const GraphViewer& graph_viewer,
    const Node& add_node) {
  if (add_node.OpType() != "Add" || add_node.InputDefs().size() != 2 ||
      add_node.OutputDefs().size() != 1) {
    return nullptr;
  }

  if (auto capability = TryCreateTokenMixerResidualCapability(graph_viewer, add_node, 0, 1)) {
    return capability;
  }
  return TryCreateTokenMixerResidualCapability(graph_viewer, add_node, 1, 0);
}

}  // namespace

// ============================================================================
// GetCapability - Intelligent Device Placement
// ============================================================================

std::vector<std::unique_ptr<ComputeCapability>>
MusaExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                     const IKernelLookup& kernel_lookup,
                                     const GraphOptimizerRegistry& /* graph_optimizer_registry */,
                                     IResourceAccountant* /* resource_accountant */) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;
  std::vector<NodeIndex> candidates;

  // Prefer provider-side fusion for the prunedGraph token mixer residual pattern.
  for (auto& node_index : graph_viewer.GetNodesInTopologicalOrder()) {
    const auto* p_node = graph_viewer.GetNode(node_index);
    if (p_node == nullptr || !p_node->GetExecutionProviderType().empty()) {
      continue;
    }

    if (auto fused_capability = TryCreatePlnCascadeBlockCapability(graph_viewer, *p_node)) {
      LOGS(*GetLogger(), INFO) << "MUSA EP fused PLN cascade block at node: " << p_node->Name();
      result.push_back(std::move(fused_capability));
      continue;
    }

    if (auto fused_capability = TryCreateLayerNormLastDimCapability(graph_viewer, *p_node)) {
      LOGS(*GetLogger(), INFO) << "MUSA EP fused layer norm at node: " << p_node->Name();
      result.push_back(std::move(fused_capability));
      continue;
    }

    if (auto fused_capability = TryCreateFeatureNormCapability(graph_viewer, *p_node)) {
      LOGS(*GetLogger(), INFO) << "MUSA EP fused feature norm at node: " << p_node->Name();
      result.push_back(std::move(fused_capability));
      continue;
    }

    if (auto fused_capability = TryCreateGeluCapability(graph_viewer, *p_node)) {
      LOGS(*GetLogger(), INFO) << "MUSA EP fused gelu at node: " << p_node->Name();
      result.push_back(std::move(fused_capability));
      continue;
    }

    if (auto fused_capability = TryCreateTranspose021ReshapeMatMulCapability(graph_viewer, *p_node)) {
      LOGS(*GetLogger(), INFO) << "MUSA EP fused transpose021 reshape matmul at node: " << p_node->Name();
      result.push_back(std::move(fused_capability));
      continue;
    }

    if (auto fused_capability = TryCreateReshapeMatMulCapability(graph_viewer, *p_node)) {
      LOGS(*GetLogger(), INFO) << "MUSA EP fused reshape matmul at node: " << p_node->Name();
      result.push_back(std::move(fused_capability));
      continue;
    }

    if (auto fused_capability = TryCreateTokenMixerResidualCapability(graph_viewer, *p_node)) {
      LOGS(*GetLogger(), INFO) << "MUSA EP fused token mixer residual at node: " << p_node->Name();
      result.push_back(std::move(fused_capability));
    }
  }

  // 1. Collect all nodes that have MUSA kernel
  for (auto& node_index : graph_viewer.GetNodesInTopologicalOrder()) {
    const auto* p_node = graph_viewer.GetNode(node_index);
    if (p_node == nullptr)
      continue;

    const auto& node = *p_node;
    if (!node.GetExecutionProviderType().empty()) {
      if (node.GetExecutionProviderType() == kMusaExecutionProvider) {
        candidates.push_back(node.Index());
      }
      continue;
    }

    // Check if there is a corresponding MUSA kernel
    const KernelCreateInfo* musa_kernel_def = kernel_lookup.LookUpKernel(node);
    if (musa_kernel_def == nullptr) {
      LOGS(*GetLogger(), INFO) << "MUSA kernel not found in registries for Op type: "
                               << node.OpType() << " node name: " << node.Name();
      continue;
    }

    candidates.push_back(node.Index());
  }

  // Claim all nodes that have MUSA kernels. Shape-related subgraphs can still be
  // cheap on CPU, but this EP is being validated with CPU fallback disabled.
  for (auto& node_index : candidates) {
    auto sub_graph = IndexedSubGraph::Create();
    sub_graph->Nodes().push_back(node_index);
    result.push_back(ComputeCapability::Create(std::move(sub_graph)));
  }

  return result;
}

}  // namespace onnxruntime
