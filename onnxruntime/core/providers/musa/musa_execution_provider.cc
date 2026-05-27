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

// Exp operations (op 6-12, 13)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Exp);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, float, Exp);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Exp);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, Exp);

// Sqrt operations (op 6-12, 13)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Sqrt);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, float, Sqrt);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 6, 12, double, Sqrt);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, MLFloat16, Sqrt);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, float, Sqrt);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, double, Sqrt);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, BFloat16, Sqrt);

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

// ConstantOfShape operations (op 9)
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, ConstantOfShape);

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

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, int8_t, BitwiseAnd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, uint8_t, BitwiseAnd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, int16_t, BitwiseAnd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, uint16_t, BitwiseAnd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, int32_t, BitwiseAnd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, uint32_t, BitwiseAnd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, int64_t, BitwiseAnd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 18, uint64_t, BitwiseAnd);

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
#if defined(MUDNN_VERSION) && (MUDNN_VERSION >= 3100)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 11, 17, MLFloat16, Pad);
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
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 5, 12, int32_t, Reshape);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 5, 12, int64_t, Reshape);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 5, 12, MLFloat16, Reshape);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 5, 12, float, Reshape);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 13, int32_t, Reshape);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 13, int64_t, Reshape);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, Reshape);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 13, float, Reshape);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 14, 18, int32_t, Reshape);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 14, 18, int64_t, Reshape);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 14, 18, MLFloat16, Reshape);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 14, 18, float, Reshape);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, int32_t, Reshape);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, int64_t, Reshape);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, MLFloat16, Reshape);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, float, Reshape);

// Shape operations (multi-version support)
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 1, 12, Shape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 13, 14, Shape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 15, 18, Shape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 19, 20, Shape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 21, 22, Shape);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 23, Shape);

// Where operations (op 9-15, 16)
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 15, float, Where);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 15, MLFloat16, Where);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 15, int32_t, Where);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 9, 15, int64_t, Where);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 16, float, Where);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 16, MLFloat16, Where);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 16, int32_t, Where);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMusaExecutionProvider, kOnnxDomain, 16, int64_t, Where);

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

      // Register ConstantOfShape operator
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 9, ConstantOfShape)>,

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

      // TEMPORARILY DISABLED - ReduceMax/Min/Sum/Mean/Prod fallback to CPU due to new SDK issues
      // Register ReduceMax operators (version 1-17)
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
      //     kMusaExecutionProvider, kOnnxDomain, 1, 17, MLFloat16, ReduceMax)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
      //     kMusaExecutionProvider, kOnnxDomain, 1, 17, float, ReduceMax)>,

      // // Register ReduceMax operators (version 18)
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
      //     kMusaExecutionProvider, kOnnxDomain, 18, MLFloat16, ReduceMax)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
      //     kMusaExecutionProvider, kOnnxDomain, 18, float, ReduceMax)>,

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
#if defined(MUDNN_VERSION) && (MUDNN_VERSION >= 3100)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 11, 17, MLFloat16, Pad)>,
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

      // Register Reshape operators (version 5-12)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 5, 12, int32_t, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 5, 12, int64_t, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 5, 12, MLFloat16, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 5, 12, float, Reshape)>,

      // Register Reshape operators (version 13)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, 13, int32_t, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, 13, int64_t, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 13, 13, float, Reshape)>,

      // Register Reshape operators (version 14-18)
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, 18, int32_t, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, 18, int64_t, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, 18, MLFloat16, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 14, 18, float, Reshape)>,

      // Register Reshape operators (version 19)
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 19, int32_t, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 19, int64_t, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 19, MLFloat16, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kMusaExecutionProvider, kOnnxDomain, 19, float, Reshape)>,

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
    OrtDevice::DeviceId device_id, musaStream_t stream) {
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

  // Enable TF32 (TensorFloat-32) for FP32 acceleration on Matrix Unit / Tensor Core
  // Following torch_musa pattern: allow_tf32_ = true by default
  // This provides huge performance gain for MatMul and Conv while preserving FP32-level accuracy
  // Reference: CUDA EP also enables TF32 by default (use_tf32{true})
  status = mudnn_handle_->SetAllowTF32(true);
  if (status != ::musa::dnn::Status::SUCCESS) {
    ORT_THROW("Failed to enable TF32 for MUSA DNN handle, status: " +
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
      context = std::make_shared<PerThreadContext>(info_.device_id, stream_);
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

  // 2. Core: Identify CPU-preferred nodes (Shape-related computation subgraphs)
  auto cpu_nodes = GetCpuPreferredNodes(graph_viewer, kernel_lookup, candidates, *GetLogger());

  // 3. Only claim non CPU-preferred nodes
  for (auto& node_index : candidates) {
    if (cpu_nodes.count(node_index) > 0) {
      // Do not claim this node, let it fallback to CPU EP
      const auto* node = graph_viewer.GetNode(node_index);
      LOGS(*GetLogger(), INFO) << "MUSA EP: Let node [" << node->OpType()
                               << "] " << node->Name() << " fallback to CPU for optimization";
      continue;
    }

    auto sub_graph = IndexedSubGraph::Create();
    sub_graph->Nodes().push_back(node_index);
    result.push_back(ComputeCapability::Create(std::move(sub_graph)));
  }

  return result;
}

}  // namespace onnxruntime
