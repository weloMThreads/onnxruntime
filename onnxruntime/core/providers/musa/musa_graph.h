// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>

#include "core/common/common.h"
#include "core/providers/musa/musa_inc.h"

namespace onnxruntime {

using MusaGraphAnnotation_t = int;
using MusaGraphExec_t = musaGraphExec_t;
using MusaGraphSet_t = std::unordered_map<MusaGraphAnnotation_t, MusaGraphExec_t>;

constexpr MusaGraphAnnotation_t kMusaGraphAnnotationSkip = -1;
constexpr MusaGraphAnnotation_t kMusaGraphAnnotationDefault = 0;

struct MusaGraphSet {
  MusaGraphSet() {}
  ~MusaGraphSet();

  void Clear();
  bool Contains(MusaGraphAnnotation_t musa_graph_annotation_id) const;
  void Put(MusaGraphAnnotation_t musa_graph_annotation_id, MusaGraphExec_t graph_exec);
  MusaGraphExec_t Get(MusaGraphAnnotation_t musa_graph_annotation_id) const;

 private:
  MusaGraphSet_t musa_graphs_;
};

struct MUSAGraphManager {
  MUSAGraphManager() {}
  ~MUSAGraphManager();

  void SetStream(musaStream_t stream);
  void CaptureBegin(MusaGraphAnnotation_t musa_graph_annotation_id);
  void CaptureEnd(MusaGraphAnnotation_t musa_graph_annotation_id);
  Status Replay(MusaGraphAnnotation_t musa_graph_annotation_id);

  void Reset();

  bool IsGraphCaptureAllowedOnRun(MusaGraphAnnotation_t musa_graph_annotation_id) const;
  bool IsGraphCaptured(MusaGraphAnnotation_t musa_graph_annotation_id) const;

 private:
  MusaGraphSet musa_graph_set_;
  MusaGraphAnnotation_t musa_graph_annotation_id_ = kMusaGraphAnnotationDefault;
  musaStream_t stream_ = nullptr;  // Does not own the stream
};

using MUSAGraph = MUSAGraphManager;

}  // namespace onnxruntime
