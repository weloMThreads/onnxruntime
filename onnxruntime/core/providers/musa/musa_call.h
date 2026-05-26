// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/common/exceptions.h"
#include "musa_inc.h"
#include <string>

namespace onnxruntime {

// MUSA error checking macro similar to CUDA_CALL
#define MUSA_CALL(x) \
  do { \
    musaError_t status = (x); \
    if (status != musaSuccess) { \
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, \
                           "MUSA call failed: " + std::string(musaGetErrorString(status)) + \
                           " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
  } while(0)

// MUSA error checking macro that throws exception
#define MUSA_CALL_THROW(x) \
  do { \
    musaError_t status = (x); \
    if (status != musaSuccess) { \
      const char* error_str = musaGetErrorString(status); \
      ORT_THROW("MUSA call failed: " + std::string(error_str ? error_str : "Unknown error") + \
               " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
  } while(0)

// MUSA error checking macro that returns on error
#define MUSA_RETURN_IF_ERROR(x) \
  do { \
    musaError_t status = (x); \
    if (status != musaSuccess) { \
      const char* error_str = musaGetErrorString(status); \
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, \
                           "MUSA call failed: " + std::string(error_str ? error_str : "Unknown error") + \
                           " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
  } while(0)

}  // namespace onnxruntime
