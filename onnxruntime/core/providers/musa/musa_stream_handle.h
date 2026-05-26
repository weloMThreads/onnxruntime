// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <vector>

#include "core/framework/stream_handles.h"
#include "core/providers/musa/musa_inc.h"
#include "core/providers/musa/musa_call.h"

namespace onnxruntime {

// Forward declaration
void WaitMusaNotificationOnDevice(Stream* stream, synchronize::Notification& notification);

struct MusaStream : Stream {
  MusaStream(musaStream_t stream, const OrtDevice& device, bool own_flag);

  ~MusaStream() override;

  MusaStream(const MusaStream&) = delete;
  MusaStream& operator=(const MusaStream&) = delete;
  MusaStream(MusaStream&&) = delete;
  MusaStream& operator=(MusaStream&&) = delete;

  std::unique_ptr<synchronize::Notification> CreateNotification(size_t num_consumers) override;

  void Flush() override;


private:
  bool own_stream_{true};
};

void RegisterMusaStreamHandles(IStreamCommandHandleRegistry& stream_handle_registry,
                               OrtDevice::DeviceType device_type,
                               musaStream_t external_stream,
                               bool use_existing_stream);

}  // namespace onnxruntime
