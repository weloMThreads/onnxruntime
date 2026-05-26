// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

// ============================================================================
// Test cases: shape tensor contains zero dimension
// These tests verify the scenario where the target shape has a zero dimension
// (e.g., batch_size=0), which is a valid ONNX scenario.
// CPU EP allows this and returns an empty tensor.
// MusaEP should have the same behavior.
// ============================================================================

// Test case 1: 3D tensor with zero first dimension
TEST(MusaExpandOpTest, Expand_ZeroDim_float) {
  OpTester test("Expand", 13);
  // Input: a normal tensor with shape {1, 7, 256}
  std::vector<float> input_data(1 * 7 * 256, 1.0f);
  test.AddInput<float>("data", {1, 7, 256}, input_data);
  // Shape tensor: {0, 7, 256} - first dimension is 0
  test.AddInput<int64_t>("shape", {3}, {0, 7, 256});
  // Expected output: empty tensor with shape {0, 7, 256}
  test.AddOutput<float>("expanded", {0, 7, 256}, {});
  test.Run();
}

// Test case 2: Simpler 2D tensor with zero first dimension
TEST(MusaExpandOpTest, Expand_ZeroDim_2D_float) {
  OpTester test("Expand", 13);
  test.AddInput<float>("data", {1, 3}, {1.0f, 2.0f, 3.0f});
  test.AddInput<int64_t>("shape", {2}, {0, 3});
  test.AddOutput<float>("expanded", {0, 3}, {});
  test.Run();
}

// Test case 3: Shape tensor contains zero in middle dimension
TEST(MusaExpandOpTest, Expand_ZeroDim_Middle_float) {
  OpTester test("Expand", 13);
  test.AddInput<float>("data", {2, 1, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddInput<int64_t>("shape", {3}, {2, 0, 3});
  test.AddOutput<float>("expanded", {2, 0, 3}, {});
  test.Run();
}

// Test case 4: with int32 type
TEST(MusaExpandOpTest, Expand_ZeroDim_int32) {
  OpTester test("Expand", 13);
  test.AddInput<int32_t>("data", {1, 3}, {1, 2, 3});
  test.AddInput<int64_t>("shape", {2}, {0, 3});
  test.AddOutput<int32_t>("expanded", {0, 3}, {});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
