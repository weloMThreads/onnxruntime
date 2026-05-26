// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/ortdevice.h"
#include "core/framework/provider_options.h"
#include "core/providers/musa/musa_provider_options.h"
#include "core/providers/musa/musa_execution_provider_info.h"
#include "onnxruntime_c_api.h"
#include <memory>

namespace onnxruntime {
class IAllocator;
class IDataTransfer;
struct IExecutionProviderFactory;
enum class ArenaExtendStrategy : int32_t;


struct ProviderInfo_MUSA {
  virtual int musaGetDeviceCount() = 0;

  virtual void musaMemcpy_HostToDevice(void *dst, const void *src,
                                       size_t count) = 0;
  virtual void musaMemcpy_DeviceToHost(void *dst, const void *src,
                                       size_t count) = 0;
  virtual void MusaExecutionProviderInfo__FromProviderOptions(
      const onnxruntime::ProviderOptions &options,
      onnxruntime::MusaExecutionProviderInfo &info) = 0;
  virtual std::shared_ptr<onnxruntime::IExecutionProviderFactory>
  CreateExecutionProviderFactory(
      const onnxruntime::MusaExecutionProviderInfo &info) = 0;
  virtual std::shared_ptr<onnxruntime::IAllocator>
  CreateMusaAllocator(int16_t device_id, size_t gpu_mem_limit,
                      onnxruntime::ArenaExtendStrategy arena_extend_strategy,
                      OrtArenaCfg *default_memory_arena_cfg) = 0;
};

} // namespace onnxruntime
