// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/musa_stream_handle.h"
#include "core/common/spin_pause.h"
#include <cassert>

namespace onnxruntime {

struct MusaNotification : public synchronize::Notification {
  explicit MusaNotification(Stream& s) : Notification(s), event_(nullptr) {
    MUSA_CALL_THROW(musaEventCreate(&event_));
  }

  ~MusaNotification() override {
    if (event_ != nullptr) {
      musaEventDestroy(event_);  // Don't throw in destructor
    }
  }

  MusaNotification(const MusaNotification&) = delete;
  MusaNotification& operator=(const MusaNotification&) = delete;
  MusaNotification(MusaNotification&&) = delete;
  MusaNotification& operator=(MusaNotification&&) = delete;

  void Activate() override {
    MUSA_CALL_THROW(musaEventRecord(event_, static_cast<musaStream_t>(GetStream().GetHandle())));
  }

  void wait_on_device(Stream& device_stream) const {
    ORT_ENFORCE(device_stream.GetDevice().Type() == OrtDevice::GPU);
    MUSA_CALL_THROW(musaStreamWaitEvent(static_cast<musaStream_t>(device_stream.GetHandle()), event_));
  }

  void wait_on_host() const {
    MUSA_CALL_THROW(musaEventSynchronize(event_));
  }

private:
  musaEvent_t event_;
};

MusaStream::MusaStream(musaStream_t stream,
                       const OrtDevice& device,
                       bool own_flag) : Stream(stream, device),
                                        own_stream_(own_flag) {}

MusaStream::~MusaStream() {
  ORT_IGNORE_RETURN_VALUE(CleanUpOnRunEnd());
  if (own_stream_) {
    auto* handle = GetHandle();
    if (handle != nullptr) {
      musaStreamDestroy(static_cast<musaStream_t>(handle));
    }
  }
}

std::unique_ptr<synchronize::Notification> MusaStream::CreateNotification(size_t /*num_consumers*/) {
  return std::make_unique<MusaNotification>(*this);
}

void MusaStream::Flush() {
  if (own_stream_) {
    MUSA_CALL_THROW(musaStreamSynchronize(static_cast<musaStream_t>(GetHandle())));
  }
}

// Stream command handles
void WaitMusaNotificationOnDevice(Stream* stream, synchronize::Notification& notification) {
  assert(stream != nullptr);
  static_cast<MusaNotification*>(&notification)->wait_on_device(*stream);
}

void WaitMusaNotificationOnHost(Stream* /*stream*/, synchronize::Notification& notification) {
  static_cast<MusaNotification*>(&notification)->wait_on_host();
}

void RegisterMusaStreamHandles(IStreamCommandHandleRegistry& stream_handle_registry,
                               const OrtDevice::DeviceType device_type,
                               musaStream_t external_stream,
                               bool use_existing_stream) {
  // Register device-to-device wait function
  stream_handle_registry.RegisterWaitFn(device_type, device_type, WaitMusaNotificationOnDevice);
  // Register device-to-CPU wait function
  stream_handle_registry.RegisterWaitFn(device_type, OrtDevice::CPU, WaitMusaNotificationOnHost);

  // Register Stream creation function
  if (!use_existing_stream) {
    // Create new stream for each request
    stream_handle_registry.RegisterCreateStreamFn(device_type, [](const OrtDevice& device) {
      MUSA_CALL_THROW(musaSetDevice(device.Id()));
      musaStream_t stream = nullptr;
      MUSA_CALL_THROW(musaStreamCreateWithFlags(&stream, musaStreamNonBlocking));
      return std::make_unique<MusaStream>(stream, device, true);  // own_flag = true
    });
  } else {
    // Reuse EP's unified stream
    stream_handle_registry.RegisterCreateStreamFn(device_type, [external_stream](const OrtDevice& device) {
      return std::make_unique<MusaStream>(external_stream, device, false);  // own_flag = false
    });
  }
  stream_handle_registry.RegisterSetDeviceFn(device_type, [](OrtDevice::DeviceId id) {
    MUSA_CALL_THROW(musaSetDevice(id));
  });
}

}  // namespace onnxruntime
