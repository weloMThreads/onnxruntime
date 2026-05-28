// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_c_api.h"

#ifndef ORT_MUSA_PROVIDER_OPTIONS_DEFINED
#define ORT_MUSA_PROVIDER_OPTIONS_DEFINED

struct OrtMUSAProviderOptions {
#ifdef __cplusplus
  OrtMUSAProviderOptions() : device_id{0}, prefer_nhwc{0}, enable_musa_graph{0}, use_tf32{0} {}
#endif
  int device_id; // MUSA device id
  int prefer_nhwc; // Prefer NHWC layout (0 = NCHW, 1 = NHWC)
  int enable_musa_graph; // Enable MUSAGraph skeleton path
  int use_tf32; // Enable TF32/TensorCore math for FP32 MatMul/Conv (0 = strict FP32, 1 = faster TF32)
};

#endif  // ORT_MUSA_PROVIDER_OPTIONS_DEFINED
