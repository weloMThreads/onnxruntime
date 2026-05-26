// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/musa_provider_options.h"
#include "onnxruntime_cxx_api.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

TEST(MusaSessionOptionsTest, AppendExecutionProviderMUSA) {
  // This test is designed to trigger a compilation error initially
  // to demonstrate the missing AppendExecutionProvider_MUSA implementation

  Ort::SessionOptions session_options;
  OrtMUSAProviderOptions options;
  options.device_id = 0;

  // This line should trigger the compilation error:
  // undefined reference to
  // `Ort::detail::SessionOptionsImpl<OrtSessionOptions>::AppendExecutionProvider_MUSA(OrtMUSAProviderOptions
  // const&)'
  session_options.AppendExecutionProvider_MUSA(options);

  // If we get here, the compilation succeeded
  ASSERT_TRUE(true);
}

TEST(MusaSessionOptionsTest, MusaProviderOptionsInitialization) {
  // Test that OrtMUSAProviderOptions can be initialized properly
  OrtMUSAProviderOptions options;
  options.device_id = 0;

  ASSERT_EQ(options.device_id, 0);

  // Test setting different device ID
  options.device_id = 1;
  ASSERT_EQ(options.device_id, 1);
}

} // namespace test
} // namespace onnxruntime
