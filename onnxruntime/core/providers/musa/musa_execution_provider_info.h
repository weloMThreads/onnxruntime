// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "core/common/make_string.h"
#include "core/framework/provider_options.h"
#include "core/framework/provider_options_utils.h"
#include "core/session/onnxruntime_c_api.h"

// OrtMUSAProviderOptions definition (public C API)
// This is defined in include/onnxruntime/core/providers/musa/musa_provider_options.h
// For internal builds, we define it here to avoid include path issues
#ifndef ORT_MUSA_PROVIDER_OPTIONS_DEFINED
#define ORT_MUSA_PROVIDER_OPTIONS_DEFINED
struct OrtMUSAProviderOptions {
#ifdef __cplusplus
  OrtMUSAProviderOptions() : device_id{0}, prefer_nhwc{false}, enable_musa_graph{0}, use_tf32{0}, use_bf16{0} {}
#endif
  int device_id;
  bool prefer_nhwc;
  int enable_musa_graph;
  int use_tf32;
  int use_bf16;
};
#endif

namespace onnxruntime {

// Provider option names
namespace musa {
namespace provider_option_names {
constexpr const char* kDeviceId = "device_id";
constexpr const char* kPreferNHWC = "prefer_nhwc";
constexpr const char* kEnableMUSAGraph = "enable_musa_graph";
constexpr const char* kUseTF32 = "use_tf32";
constexpr const char* kUseBF16 = "use_bf16";
}  // namespace provider_option_names
}  // namespace musa

struct MusaExecutionProviderInfo {
  int device_id{0};
  bool prefer_nhwc{false};
  bool enable_musa_graph{false};
  bool use_tf32{false};
  bool use_bf16{false};

  static inline MusaExecutionProviderInfo FromProviderOptions(const ProviderOptions& options) {
    MusaExecutionProviderInfo info{};

    ORT_THROW_IF_ERROR(
        ProviderOptionsParser{}
            .AddAssignmentToReference(musa::provider_option_names::kDeviceId, info.device_id)
            .AddValueParser(
                musa::provider_option_names::kPreferNHWC,
                [&info](const std::string& value_str) {
                  ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, info.prefer_nhwc));
                  return Status::OK();
                })
            .AddAssignmentToReference(musa::provider_option_names::kEnableMUSAGraph, info.enable_musa_graph)
            .AddAssignmentToReference(musa::provider_option_names::kUseTF32, info.use_tf32)
            .AddAssignmentToReference(musa::provider_option_names::kUseBF16, info.use_bf16)
            .Parse(options));

    return info;
  }

  static inline ProviderOptions ToProviderOptions(const MusaExecutionProviderInfo& info) {
    const ProviderOptions options{
        {musa::provider_option_names::kDeviceId, MakeStringWithClassicLocale(info.device_id)},
        {musa::provider_option_names::kPreferNHWC, MakeStringWithClassicLocale(info.prefer_nhwc)},
        {musa::provider_option_names::kEnableMUSAGraph, MakeStringWithClassicLocale(info.enable_musa_graph)},
        {musa::provider_option_names::kUseTF32, MakeStringWithClassicLocale(info.use_tf32)},
        {musa::provider_option_names::kUseBF16, MakeStringWithClassicLocale(info.use_bf16)},
    };
    return options;
  }
};

}  // namespace onnxruntime
