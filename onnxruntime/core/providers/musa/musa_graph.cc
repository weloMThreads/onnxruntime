// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/musa_graph.h"
#include "core/providers/musa/musa_call.h"

namespace onnxruntime {

MusaGraphSet::~MusaGraphSet() {
  Clear();
}

void MusaGraphSet::Clear() {
  for (auto& it : musa_graphs_) {
    MUSA_CALL_THROW(musaGraphExecDestroy(it.second));
  }
  musa_graphs_.clear();
}

bool MusaGraphSet::Contains(MusaGraphAnnotation_t musa_graph_annotation_id) const {
  return musa_graphs_.count(musa_graph_annotation_id) > 0;
}

void MusaGraphSet::Put(MusaGraphAnnotation_t musa_graph_annotation_id, MusaGraphExec_t graph_exec) {
  musa_graphs_[musa_graph_annotation_id] = graph_exec;
}

MusaGraphExec_t MusaGraphSet::Get(MusaGraphAnnotation_t musa_graph_annotation_id) const {
  return musa_graphs_.at(musa_graph_annotation_id);
}

MUSAGraphManager::~MUSAGraphManager() {
  Reset();
}

void MUSAGraphManager::SetStream(musaStream_t stream) {
  stream_ = stream;
}

void MUSAGraphManager::CaptureBegin(MusaGraphAnnotation_t musa_graph_annotation_id) {
  ORT_ENFORCE(IsGraphCaptureAllowedOnRun(musa_graph_annotation_id));
  ORT_ENFORCE(!musa_graph_set_.Contains(musa_graph_annotation_id),
              "Trying to capture a graph with annotation id ", musa_graph_annotation_id,
              " that already used. Please use a different annotation id.");

  MUSA_CALL_THROW(musaStreamSynchronize(stream_));
  MUSA_CALL_THROW(musaStreamBeginCapture(stream_, musaStreamCaptureModeGlobal));
}

void MUSAGraphManager::CaptureEnd(MusaGraphAnnotation_t musa_graph_annotation_id) {
  musaGraph_t graph = nullptr;
  MUSA_CALL_THROW(musaStreamEndCapture(stream_, &graph));
  if (graph == nullptr) {
    ORT_THROW("MUSAGraphManager::CaptureEnd: graph is NULL");
  }

  musaGraphExec_t graph_exec = nullptr;
  MUSA_CALL_THROW(musaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  MUSA_CALL_THROW(musaGraphDestroy(graph));
  musa_graph_set_.Put(musa_graph_annotation_id, graph_exec);
}

Status MUSAGraphManager::Replay(MusaGraphAnnotation_t musa_graph_annotation_id) {
  LOGS_DEFAULT(INFO) << "Replaying MUSA graph on stream " << stream_ << " with musa_graph_annotation_id "
                     << musa_graph_annotation_id;

  MUSA_RETURN_IF_ERROR(musaGraphLaunch(musa_graph_set_.Get(musa_graph_annotation_id), stream_));
  return Status::OK();
}

void MUSAGraphManager::Reset() {
  musa_graph_set_.Clear();
}

bool MUSAGraphManager::IsGraphCaptureAllowedOnRun(MusaGraphAnnotation_t musa_graph_annotation_id) const {
  return musa_graph_annotation_id != kMusaGraphAnnotationSkip;
}

bool MUSAGraphManager::IsGraphCaptured(MusaGraphAnnotation_t musa_graph_annotation_id) const {
  return musa_graph_set_.Contains(musa_graph_annotation_id);
}

}  // namespace onnxruntime
