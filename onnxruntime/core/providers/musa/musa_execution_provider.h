// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/framework/execution_provider.h"
#include "core/providers/musa/musa_provider_factory.h"
#include "core/providers/musa/musa_graph.h"
#include "core/providers/musa/musa_execution_provider_info.h"

#include <mutex>
#include <set>
#include <memory>
#include <unordered_map>
#include <musa_runtime.h>
#include <mudnn.h>

namespace onnxruntime {

class MusaExecutionProvider : public IExecutionProvider {
 public:
  explicit MusaExecutionProvider(const MusaExecutionProviderInfo &info);
  ~MusaExecutionProvider() override;

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

  std::unique_ptr<IDataTransfer> GetDataTransfer() const override;

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
      const onnxruntime::GraphViewer& graph_viewer,
      const IKernelLookup& kernel_lookup,
      const GraphOptimizerRegistry& graph_optimizer_registry,
      IResourceAccountant* resource_accountant) const override;

  std::vector<AllocatorPtr> CreatePreferredAllocators() override;

  const void *GetExecutionHandle() const noexcept override { return nullptr; }

  DataLayout GetPreferredLayout() const override { 
    return info_.prefer_nhwc ? DataLayout::NHWC : DataLayout::NCHW; 
  }

  bool IsNHWCPreferred() const { return info_.prefer_nhwc; }

  bool UseTF32() const { return info_.use_tf32; }

  bool UseBF16() const { return info_.use_bf16; }

  int GetDeviceId() const override { return info_.device_id; }

  FusionStyle GetFusionStyle() const override {
    return FusionStyle::FilteredGraphViewer;
  }

  bool IsGraphCaptureEnabled() const override;
  bool IsGraphCaptured(int graph_annotation_id) const override;
  Status ReplayGraph(int graph_annotation_id) override;

  void RegisterStreamHandlers(IStreamCommandHandleRegistry& stream_handle_registry,
                             AllocatorMap& allocator_map) const override;

  // ============================================================================
  // PerThreadContext - Thread-local context for MUSA operations
  // ============================================================================
  class PerThreadContext final {
   public:
    PerThreadContext(OrtDevice::DeviceId device_id, musaStream_t stream, bool use_tf32, bool use_bf16);
    ~PerThreadContext();

    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(PerThreadContext);

    ::musa::dnn::Handle& MudnnHandle() const {
      return *mudnn_handle_;
    }

    bool IsGraphCaptureAllowed(MusaGraphAnnotation_t musa_graph_annotation_id) const;
    bool IsGraphCaptureAllowedOnRun(MusaGraphAnnotation_t musa_graph_annotation_id) const;
    MusaGraphAnnotation_t GetMusaGraphAnnotationId(const RunOptions& run_options) const;
    void CaptureBegin(MusaGraphAnnotation_t musa_graph_annotation_id);
    void CaptureEnd(MusaGraphAnnotation_t musa_graph_annotation_id);
    bool IsGraphCaptured(MusaGraphAnnotation_t musa_graph_annotation_id) const;
    Status ReplayGraph(MusaGraphAnnotation_t musa_graph_annotation_id);
    void IncrementRegularRunCountBeforeGraphCapture(MusaGraphAnnotation_t musa_graph_annotation_id);

   private:
    std::unique_ptr<::musa::dnn::Handle> mudnn_handle_;

    // Graph support
    MUSAGraphManager musa_graph_;
    std::unordered_map<MusaGraphAnnotation_t, int> graph_id_to_run_count_;
    static constexpr int min_num_runs_before_musa_graph_capture_ = 1;
  };

  // Lifecycle hooks
  Status OnRunStart(const onnxruntime::RunOptions& run_options) override;
  Status OnRunEnd(bool sync_stream, const onnxruntime::RunOptions& run_options) override;

  // PerThreadContext access
  PerThreadContext& GetPerThreadContext() const;

  // Get the EP-level stream
  musaStream_t GetStream() const { return stream_; }

 private:
  MusaExecutionProviderInfo info_;

  // Stream management
  musaStream_t stream_ = nullptr;
  bool use_ep_level_unified_stream_ = true;
  bool enable_musa_graph_{false};

  // ============================================================================
  // PerThreadContext Cache Infrastructure
  // ============================================================================

  // thread_local cache type definition
  using PerThreadContextMap = std::unordered_map<const MusaExecutionProvider*, std::weak_ptr<PerThreadContext>>;

  struct ContextCacheHolder {
    ContextCacheHolder() {
      // Use weak pointer to prevent resource leaks during DLL unload
      RunOnUnload([&, weak_p_ = std::weak_ptr<PerThreadContextMap>(p)] {
        if (auto lock = weak_p_.lock())
          p.reset();
      });
    }
    std::shared_ptr<PerThreadContextMap> p = std::make_shared<PerThreadContextMap>();
  };

  static const std::shared_ptr<PerThreadContextMap>& PerThreadContextCache() {
    thread_local const ContextCacheHolder per_thread_context_cache;
    return per_thread_context_cache.p;
  }

  struct PerThreadContextState {
    // Currently active contexts
    std::set<std::shared_ptr<PerThreadContext>, std::owner_less<std::shared_ptr<PerThreadContext>>> active_contexts;
    // Reusable contexts pool
    std::vector<std::shared_ptr<PerThreadContext>> retired_context_pool;
    // Thread local caches that need to be updated on EP destruction
    std::set<std::weak_ptr<PerThreadContextMap>, std::owner_less<std::weak_ptr<PerThreadContextMap>>>
        caches_to_update_on_destruction;
    // Mutex protecting PerThreadContextState
    std::mutex mutex;
  };

  // PerThreadContext state management
  mutable PerThreadContextState context_state_;

  void ReleasePerThreadContext() const;
};

} // namespace onnxruntime
