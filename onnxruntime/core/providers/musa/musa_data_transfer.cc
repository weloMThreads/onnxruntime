// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/musa_data_transfer.h"
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/musa/musa_stream_handle.h"

namespace onnxruntime {

MusaDataTransfer::MusaDataTransfer() = default;

MusaDataTransfer::~MusaDataTransfer() = default;

bool MusaDataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return src_device.Type() == OrtDevice::GPU || dst_device.Type() == OrtDevice::GPU;
}

common::Status MusaDataTransfer::CopyTensor(const Tensor& src, Tensor& dst) const {
  size_t bytes = src.SizeInBytes();
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();

  auto& src_device = src.Location().device;
  auto& dst_device = dst.Location().device;

  // Handle different copy scenarios
  if (dst_device.Type() == OrtDevice::GPU) {
    if (src_device.Type() == OrtDevice::GPU) {
      // GPU to GPU copy
      if (dst_data != src_data) {
        return MusaMemcpyHelper(dst_data, src_data, bytes, musaMemcpyDeviceToDevice);
      }
    } else {
      // CPU to GPU copy (Host to Device)
      return MusaMemcpyHelper(dst_data, src_data, bytes, musaMemcpyHostToDevice);
    }
  } else if (src_device.Type() == OrtDevice::GPU) {
    // GPU to CPU copy (Device to Host)
    return MusaMemcpyHelper(dst_data, src_data, bytes, musaMemcpyDeviceToHost);
  } else {
    // CPU to CPU copy
    if (dst_data != src_data) {
      memcpy(dst_data, src_data, bytes);
    }
  }

  return Status::OK();
}

common::Status MusaDataTransfer::CopyTensorAsync(const Tensor& src, Tensor& dst, Stream& stream) const {
  size_t bytes = src.SizeInBytes();
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();

  auto& src_device = src.Location().device;
  auto& dst_device = dst.Location().device;

  if (dst_device.Type() == OrtDevice::GPU) {
    if (src_device.Type() == OrtDevice::CPU) {
      // CPU to GPU async copy
      return MusaMemcpyAsyncHelper(dst_data, src_data, bytes,
                                  musaMemcpyHostToDevice,
                                  static_cast<musaStream_t>(stream.GetHandle()));
    } else if (src_device.Type() == OrtDevice::GPU) {
      // GPU to GPU async copy
      if (dst_data != src_data) {
        return MusaMemcpyAsyncHelper(dst_data, src_data, bytes,
                                    musaMemcpyDeviceToDevice,
                                    static_cast<musaStream_t>(stream.GetHandle()));
      }
    }
  } else if (src_device.Type() == OrtDevice::GPU) {
    if (dst_device.Type() == OrtDevice::CPU) {
      // GPU to CPU async copy
      return MusaMemcpyAsyncHelper(dst_data, src_data, bytes,
                                  musaMemcpyDeviceToHost,
                                  static_cast<musaStream_t>(stream.GetHandle()));
    }
  } else {
    // CPU to CPU copy: Handle pinned memory case
    // TODO: Add proper pinned memory detection when OrtDevice::MemType::MUSA_PINNED is available
    // For now, assume it's regular CPU memory and handle synchronously

    if (dst_data != src_data) {
      memcpy(dst_data, src_data, bytes);
    }
  }

  return Status::OK();
}

bool MusaDataTransfer::IsDevicePointer(const void* ptr) const {
  if (ptr == nullptr) {
    return false;
  }

  musaPointerAttributes attrs{};
  musaError_t status = musaPointerGetAttributes(&attrs, ptr);

  if (status == musaSuccess) {
    return (attrs.type == musaMemoryTypeDevice);
  }

  return false;
}

common::Status MusaDataTransfer::MusaMemcpyHelper(void* dst, const void* src, size_t bytes, musaMemcpyKind kind) const {
  if (bytes == 0) {
    return Status::OK();
  }

  musaError_t copy_status = musaMemcpy(dst, src, bytes, kind);
  if (copy_status != musaSuccess) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                          "MUSA memory copy failed, status: " +
                          std::to_string(static_cast<int>(copy_status)));
  }

  // Synchronize to ensure copy completion for sync version
  musaError_t sync_status = musaDeviceSynchronize();
  if (sync_status != musaSuccess) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                          "MUSA device synchronization failed, status: " +
                          std::to_string(static_cast<int>(sync_status)));
  }

  return Status::OK();
}

common::Status MusaDataTransfer::MusaMemcpyAsyncHelper(void* dst, const void* src, size_t bytes,
                                                      musaMemcpyKind kind, musaStream_t stream) const {
  if (bytes == 0) {
    return Status::OK();
  }

  musaError_t status = musaMemcpyAsync(dst, src, bytes, kind, stream);
  if (status != musaSuccess) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                          "MUSA async memory copy failed, status: " +
                          std::to_string(static_cast<int>(status)));
  }

  return Status::OK();
}

}  // namespace onnxruntime
