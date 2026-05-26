// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/musa_allocator.h"
#include "core/providers/musa/musa_call.h"
#include <musa_runtime.h>

namespace onnxruntime {

void MusaAllocator::CheckDevice(bool throw_when_fail) const {
#ifndef NDEBUG
  int current_device;
  auto musa_err = musaGetDevice(&current_device);
  if (musa_err == musaSuccess) {
    ORT_ENFORCE(current_device == Info().device.Id());
  } else if (throw_when_fail) {
    MUSA_CALL_THROW(musa_err);
  }
#else
  ORT_UNUSED_PARAMETER(throw_when_fail);
#endif
}

void MusaAllocator::SetDevice(bool throw_when_fail) const {
  int current_device;
  auto musa_err = musaGetDevice(&current_device);
  if (musa_err == musaSuccess) {
    int allocator_device_id = Info().device.Id();
    if (current_device != allocator_device_id) {
      musa_err = musaSetDevice(allocator_device_id);
    }
  }

  if (musa_err != musaSuccess && throw_when_fail) {
    MUSA_CALL_THROW(musa_err);
  }
}

void* MusaAllocator::Alloc(size_t size) {
  SetDevice(true);
  CheckDevice(true);
  void* p = nullptr;
  if (size > 0) {
    MUSA_CALL_THROW(musaMalloc(&p, size));
  }
  return p;
}

void MusaAllocator::Free(void* p) {
  SetDevice(false);
  CheckDevice(false);
  if (p != nullptr) {
    MUSA_CALL_THROW(musaFree(p));
  }
}

void* MusaPinnedAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    musaError_t status = musaMallocHost(&p, size);
    if (status != musaSuccess) {
      ORT_THROW("MUSA host memory allocation failed, status: " + std::to_string(static_cast<int>(status)));
    }
  }
  return p;
}

void MusaPinnedAllocator::Free(void* p) {
  if (p != nullptr) {
    musaError_t status = musaFreeHost(p);
    if (status != musaSuccess) {
      ORT_THROW("MUSA host memory free failed, status: " + std::to_string(static_cast<int>(status)));
    }
  }
}

}  // namespace onnxruntime
