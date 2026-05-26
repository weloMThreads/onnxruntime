// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/math/mod.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/musa_execution_provider.h"
#include <musa_runtime.h>
#include <mudnn.h>

using onnxruntime::common::Status;
namespace onnxruntime {
namespace musa {

template <typename T>
Status SimpleMusaModOp(const MusaPreparation& prepare, bool is_fmod, const std::string& op_name) {
    ::musa::dnn::Binary op;

    // Set the appropriate mode based on fmod flag
    ::musa::dnn::Binary::Mode mode = is_fmod ? 
        ::musa::dnn::Binary::Mode::TRUNCATEMOD :  // fmod=1: use TRUNCATEMOD (like C fmod)
        ::musa::dnn::Binary::Mode::FLOORMOD;      // fmod=0: use FLOORMOD (like Python %)

    auto status = op.SetMode(mode);
    if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set mudnn Binary mode");
    }

    // Get non-const references to tensors for MUSA API
    auto& output_tensor = const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]);

    status = op.Run(prepare.GetHandle(), output_tensor, 
                   prepare.inputTensors[0], prepare.inputTensors[1]);
    if (status != ::musa::dnn::Status::SUCCESS) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to run mudnn Binary " + op_name);
    }
    
    return Status::OK();
}

template <typename T>
Status Mod<T>::ComputeInternal(OpKernelContext* context) const {
    const auto* ep = static_cast<const MusaExecutionProvider*>(
        Info().GetExecutionProvider());
    MusaPreparation prepare(ep);
    ORT_RETURN_IF_ERROR(Prepare<T>(context, prepare));
    
    return SimpleMusaModOp<T>(prepare, fmod_, "Mod");
}

// Register supported types based on the调研文档中提到的类型支持
// 根据MUSA接口调研，支持float, int32, int64, uint32, uint64等类型
// 注意：不支持double和bfloat16（根据用户要求）

#define REGISTER_MUSA_MOD_TYPED_KERNEL(ver, T) \
    ONNX_OPERATOR_TYPED_KERNEL_EX( \
        Mod, kOnnxDomain, ver, T, kMusaExecutionProvider, \
        (*KernelDefBuilder::Create()) \
            .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
        Mod<T>);

// v10-v12版本注册
#define REGISTER_MUSA_MOD_VERSIONED_TYPED_KERNEL(startver, endver, T) \
    ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX( \
        Mod, kOnnxDomain, startver, endver, T, kMusaExecutionProvider, \
        (*KernelDefBuilder::Create()) \
            .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
        Mod<T>);

// Register v10-v12 versions - only for supported types (MUSA-compatible)
REGISTER_MUSA_MOD_VERSIONED_TYPED_KERNEL(10, 12, int32_t)
REGISTER_MUSA_MOD_VERSIONED_TYPED_KERNEL(10, 12, int64_t)
// REGISTER_MUSA_MOD_VERSIONED_TYPED_KERNEL(10, 12, uint32_t)  // Disabled: muDNN Binary doesn't support unsigned types
// REGISTER_MUSA_MOD_VERSIONED_TYPED_KERNEL(10, 12, uint64_t)  // Disabled: muDNN Binary doesn't support unsigned types
REGISTER_MUSA_MOD_VERSIONED_TYPED_KERNEL(10, 12, float)
REGISTER_MUSA_MOD_VERSIONED_TYPED_KERNEL(10, 12, MLFloat16)

// Register v13+ version - only for supported types (MUSA-compatible)
REGISTER_MUSA_MOD_TYPED_KERNEL(13, int32_t)
REGISTER_MUSA_MOD_TYPED_KERNEL(13, int64_t)
// REGISTER_MUSA_MOD_TYPED_KERNEL(13, uint32_t)  // Disabled: muDNN Binary doesn't support unsigned types
// REGISTER_MUSA_MOD_TYPED_KERNEL(13, uint64_t)  // Disabled: muDNN Binary doesn't support unsigned types
REGISTER_MUSA_MOD_TYPED_KERNEL(13, float)
REGISTER_MUSA_MOD_TYPED_KERNEL(13, MLFloat16)


}  // namespace musa
}  // namespace onnxruntime