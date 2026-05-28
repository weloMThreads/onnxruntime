// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/musa_provider_factory.h"
#include "core/providers/musa/musa_execution_provider.h"
#include "core/providers/musa/musa_call.h"

namespace onnxruntime {
namespace musa_ep {

// Forward declarations - implemented in musa_execution_provider.cc
void InitializeRegistry();
void DeleteRegistry();

struct MusaProviderFactory : IExecutionProviderFactory {
  explicit MusaProviderFactory(const MusaExecutionProviderInfo &info)
      : info_(info) {}
  ~MusaProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

private:
  MusaExecutionProviderInfo info_;
};

std::unique_ptr<IExecutionProvider> MusaProviderFactory::CreateProvider() {
  return std::make_unique<MusaExecutionProvider>(info_);
}

struct ProviderInfo_MUSA_Impl : ProviderInfo_MUSA {
  int musaGetDeviceCount() override {
    int num_devices = 0;
    MUSA_CALL_THROW(::musaGetDeviceCount(&num_devices));
    return num_devices;
  }

  void musaMemcpy_HostToDevice(void *dst, const void *src,
                               size_t count) override {
    // CANN_CALL_THROW(aclrtMemcpy(dst, count, src, count,
    // ACL_MEMCPY_HOST_TO_DEVICE)); CANN_CALL_THROW(aclrtSynchronizeStream(0));
  }

  void musaMemcpy_DeviceToHost(void *dst, const void *src,
                               size_t count) override {
    // CANN_CALL_THROW(aclrtMemcpy(dst, count, src, count,
    // ACL_MEMCPY_DEVICE_TO_HOST));
  }

  void MusaExecutionProviderInfo__FromProviderOptions(
      const ProviderOptions &options,
      MusaExecutionProviderInfo &info) override {
    info = MusaExecutionProviderInfo::FromProviderOptions(options);
  }

  std::shared_ptr<IAllocator>
  CreateMusaAllocator(int16_t device_id, size_t gpu_mem_limit,
                      ArenaExtendStrategy arena_extend_strategy,
                      OrtArenaCfg *default_memory_arena_cfg) override {
    return nullptr; // For now, return nullptr
  }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(
      const MusaExecutionProviderInfo &info) override {
    return std::make_shared<MusaProviderFactory>(info);
  }
} g_info;

struct Musa_Provider : Provider {
  void *GetInfo() override { return &g_info; }

  std::shared_ptr<IExecutionProviderFactory>
  CreateExecutionProviderFactory(const void *void_params) override {
    auto params = reinterpret_cast<const OrtMUSAProviderOptions *>(void_params);

    MusaExecutionProviderInfo info{};
    info.device_id = static_cast<OrtDevice::DeviceId>(params->device_id);
    info.prefer_nhwc = params->prefer_nhwc;
    info.enable_musa_graph = params->enable_musa_graph != 0;
    info.use_tf32 = params->use_tf32 != 0;
    info.use_bf16 = params->use_bf16 != 0;
    // info.npu_mem_limit = params->npu_mem_limit;
    // info.arena_extend_strategy = params->arena_extend_strategy;
    // info.enable_cann_graph = params->enable_cann_graph != 0;
    // info.dump_graphs = params->dump_graphs != 0;
    // info.dump_om_model = params->dump_om_model != 0;
    // info.precision_mode = params->precision_mode;
    // info.op_select_impl_mode = params->op_select_impl_mode;
    // info.optypelist_for_implmode = params->optypelist_for_implmode;
    // info.default_memory_arena_cfg = params->default_memory_arena_cfg;

    return std::make_shared<musa_ep::MusaProviderFactory>(info);
  }

  void UpdateProviderOptions(void *provider_options,
                             const ProviderOptions &options) override {
    auto internal_options = onnxruntime::MusaExecutionProviderInfo::FromProviderOptions(options);
    auto& musa_options = *reinterpret_cast<OrtMUSAProviderOptions*>(provider_options);

    musa_options.device_id = internal_options.device_id;
    musa_options.prefer_nhwc = internal_options.prefer_nhwc;
    musa_options.enable_musa_graph = internal_options.enable_musa_graph;
    musa_options.use_tf32 = internal_options.use_tf32 ? 1 : 0;
    musa_options.use_bf16 = internal_options.use_bf16 ? 1 : 0;
  }

  ProviderOptions GetProviderOptions(const void *provider_options) override {
    auto& musa_options = *reinterpret_cast<const OrtMUSAProviderOptions*>(provider_options);

    MusaExecutionProviderInfo info{};
    info.device_id = static_cast<OrtDevice::DeviceId>(musa_options.device_id);
    info.prefer_nhwc = musa_options.prefer_nhwc != 0;
    info.enable_musa_graph = musa_options.enable_musa_graph != 0;
    info.use_tf32 = musa_options.use_tf32 != 0;
    info.use_bf16 = musa_options.use_bf16 != 0;

    return onnxruntime::MusaExecutionProviderInfo::ToProviderOptions(info);
  }

  void Initialize() override { musa_ep::InitializeRegistry(); }

  void Shutdown() override { musa_ep::DeleteRegistry(); }
} g_provider;

} // namespace musa_ep

} // namespace onnxruntime

extern "C" {

ORT_API(onnxruntime::Provider *, GetProvider) {
  return &onnxruntime::musa_ep::g_provider;
}
}
