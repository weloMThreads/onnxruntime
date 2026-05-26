// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers.h"
#include "core/framework/allocator.h"
#include <mutex>

namespace onnxruntime {

class MusaAllocator : public IAllocator {
 public:
  MusaAllocator(OrtDevice::DeviceId device_id, const char* name)
      : IAllocator(
            OrtMemoryInfo(name, OrtAllocatorType::OrtDeviceAllocator,
                          OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NONE, device_id),
                          OrtMemTypeDefault)) {}
  void* Alloc(size_t size) override;
  void Free(void* p) override;

 private:
  void CheckDevice(bool throw_when_fail) const;
  void SetDevice(bool throw_when_fail) const;
};

class MusaPinnedAllocator : public IAllocator {
 public:
  MusaPinnedAllocator(OrtDevice::DeviceId device_id, const char* name)
      : IAllocator(
            OrtMemoryInfo(name, OrtAllocatorType::OrtDeviceAllocator,
                          OrtDevice(OrtDevice::CPU, OrtDevice::MemType::HOST_ACCESSIBLE, OrtDevice::VendorIds::NONE, device_id),
                          OrtMemTypeCPUOutput)) {}

  void* Alloc(size_t size) override;
  void Free(void* p) override;
};

}  // namespace onnxruntime
