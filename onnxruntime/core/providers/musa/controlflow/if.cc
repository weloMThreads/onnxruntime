// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/controlflow/if.h"

#include "core/providers/musa/musa_fwd.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace musa {

// Create custom type constraints without double and bfloat16
// which are not supported by MusaEP
std::vector<MLDataType> MusaFixedSizeTensorTypes() {
  return {
      DataTypeImpl::GetTensorType<float>(),
      DataTypeImpl::GetTensorType<MLFloat16>(),
      DataTypeImpl::GetTensorType<int8_t>(),
      DataTypeImpl::GetTensorType<uint8_t>(),
      DataTypeImpl::GetTensorType<int16_t>(),
      DataTypeImpl::GetTensorType<uint16_t>(),
      DataTypeImpl::GetTensorType<int32_t>(),
      DataTypeImpl::GetTensorType<uint32_t>(),
      DataTypeImpl::GetTensorType<int64_t>(),
      DataTypeImpl::GetTensorType<uint64_t>(),
      DataTypeImpl::GetTensorType<bool>(),
  };
}

std::vector<MLDataType> MusaTensorAndSequenceTensorTypes() {
  // For simplicity, just use the standard AllTensorAndSequenceTensorTypes but filter out unsupported ones
  auto all_types = DataTypeImpl::AllTensorAndSequenceTensorTypes();
  std::vector<MLDataType> supported_types;
  
  for (const auto& type : all_types) {
    // Skip double and bfloat16 types which are not supported by MusaEP
    if (type != DataTypeImpl::GetTensorType<double>() &&
        type != DataTypeImpl::GetTensorType<BFloat16>()) {
      supported_types.push_back(type);
    }
  }
  return supported_types;
}

ONNX_OPERATOR_VERSIONED_KERNEL_EX(If,
                                  kOnnxDomain,
                                  1, 10,
                                  kMusaExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .InputMemoryType(OrtMemTypeCPUInput, 0)  // 'cond' needs to be on CPU
                                      .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
                                      .TypeConstraint("V", MusaFixedSizeTensorTypes()),
                                  If);

// output shape rules requiring the output shapes of the 'THEN' and 'ELSE'
// branches to be the same were relaxed in opset-11
ONNX_OPERATOR_VERSIONED_KERNEL_EX(If,
                                  kOnnxDomain,
                                  11, 12,
                                  kMusaExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .InputMemoryType(OrtMemTypeCPUInput, 0)  // 'cond' needs to be on CPU
                                      .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
                                      .TypeConstraint("V", MusaFixedSizeTensorTypes()),
                                  If);

// opset-13 supports sequence type for If's subgraph outputs
ONNX_OPERATOR_VERSIONED_KERNEL_EX(If,
                                  kOnnxDomain,
                                  13, 18,
                                  kMusaExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .InputMemoryType(OrtMemTypeCPUInput, 0)  // 'cond' needs to be on CPU
                                      .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
                                      .TypeConstraint("V", MusaTensorAndSequenceTensorTypes()),
                                  If);

// opset-19+ uses MusaFixedSizeTensorTypes since we don't support newer types like float8
ONNX_OPERATOR_KERNEL_EX(If,
                        kOnnxDomain,
                        19,
                        kMusaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .InputMemoryType(OrtMemTypeCPUInput, 0)  // 'cond' needs to be on CPU
                            .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
                            .TypeConstraint("V", MusaFixedSizeTensorTypes()),
                        If);

Status If::Compute(OpKernelContext* ctx) const {
  // call the base CPU version.
  // we have this MUSA implementation so the inputs/outputs stay on GPU where possible.
  // the logic to run the subgraph must be on CPU either way.
  // technically we don't need this override of Compute, but it will be optimized out and it's easier to debug
  // that this implementation is being called with it.
  auto status = onnxruntime::If::Compute(ctx);
  return status;
}

}  // namespace musa
}  // namespace onnxruntime