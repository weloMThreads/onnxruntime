// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Test fixture for Conv operator
class MusaConvTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Test will skip if MUSA provider is not available
  }
};

// Basic Conv test without bias (2D convolution) - Simple 1x1 convolution
TEST_F(MusaConvTest, ConvBasicNoBias) {
  OpTester test("Conv", 11);

  test.AddAttribute("kernel_shape", std::vector<int64_t>{1, 1});
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});

  // Input: 1x1x3x3 (simple test data)
  test.AddInput<float>("X", {1, 1, 3, 3},
    {1.0f, 2.0f, 3.0f,
     4.0f, 5.0f, 6.0f,
     7.0f, 8.0f, 9.0f});

  // Weight: 1x1x1x1 (simple scale by 2)
  test.AddInput<float>("W", {1, 1, 1, 1},
    {2.0f});

  // Expected output: 1x1x3x3 (each element multiplied by 2)
  test.AddOutput<float>("Y", {1, 1, 3, 3},
    {2.0f, 4.0f, 6.0f,
     8.0f, 10.0f, 12.0f,
     14.0f, 16.0f, 18.0f});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  auto musa_provider = DefaultMusaExecutionProvider();
  if (musa_provider) {
    execution_providers.push_back(std::move(musa_provider));
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
             &execution_providers);
  } else {
    GTEST_SKIP() << "MUSA execution provider not available";
  }
}

// 3x3 Conv test without bias - Based on CPU Conv2D tests
TEST_F(MusaConvTest, Conv3x3NoBias) {
  OpTester test("Conv", 11);

  test.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});

  // Input: 1x1x3x3
  test.AddInput<float>("X", {1, 1, 3, 3},
    {1.0f, 2.0f, 3.0f,
     4.0f, 5.0f, 6.0f,
     7.0f, 8.0f, 9.0f});

  // Weight: 1x1x3x3 (simple identity-like kernel)
  test.AddInput<float>("W", {1, 1, 3, 3},
    {0.0f, 0.0f, 0.0f,
     0.0f, 1.0f, 0.0f,
     0.0f, 0.0f, 0.0f});

  // Expected output: 1x1x1x1 (only center element)
  test.AddOutput<float>("Y", {1, 1, 1, 1},
    {5.0f});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  auto musa_provider = DefaultMusaExecutionProvider();
  if (musa_provider) {
    execution_providers.push_back(std::move(musa_provider));
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
             &execution_providers);
  } else {
    GTEST_SKIP() << "MUSA execution provider not available";
  }
}

// Conv test with bias
TEST_F(MusaConvTest, ConvWithBias) {
  OpTester test("Conv", 11);

  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});

  // Input: 1x1x3x3
  test.AddInput<float>("X", {1, 1, 3, 3},
    {1.0f, 2.0f, 3.0f,
     4.0f, 5.0f, 6.0f,
     7.0f, 8.0f, 9.0f});

  // Weight: 1x1x2x2
  test.AddInput<float>("W", {1, 1, 2, 2},
    {1.0f, 1.0f,
     1.0f, 1.0f});

  // Bias: 1
  test.AddInput<float>("B", {1}, {2.0f});

  // Expected output: 1x1x2x2
  // Conv result: [[12, 16], [24, 28]] + bias 2 = [[14, 18], [26, 30]]
  test.AddOutput<float>("Y", {1, 1, 2, 2},
    {14.0f, 18.0f,
     26.0f, 30.0f});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  auto musa_provider = DefaultMusaExecutionProvider();
  if (musa_provider) {
    execution_providers.push_back(std::move(musa_provider));
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
             &execution_providers);
  } else {
    GTEST_SKIP() << "MUSA execution provider not available";
  }
}

// Conv test with stride
TEST_F(MusaConvTest, ConvWithStride) {
  OpTester test("Conv", 11);

  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});
  test.AddAttribute("strides", std::vector<int64_t>{2, 2});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});

  // Input: 1x1x4x4
  test.AddInput<float>("X", {1, 1, 4, 4},
    {1.0f, 2.0f, 3.0f, 4.0f,
     5.0f, 6.0f, 7.0f, 8.0f,
     9.0f, 10.0f, 11.0f, 12.0f,
     13.0f, 14.0f, 15.0f, 16.0f});

  // Weight: 1x1x2x2
  test.AddInput<float>("W", {1, 1, 2, 2},
    {1.0f, 1.0f,
     1.0f, 1.0f});

  // Expected output: 1x1x2x2 (stride=2 reduces output size)
  // Top-left: 1+2+5+6 = 14, Top-right: 3+4+7+8 = 22
  // Bottom-left: 9+10+13+14 = 46, Bottom-right: 11+12+15+16 = 54
  test.AddOutput<float>("Y", {1, 1, 2, 2},
    {14.0f, 22.0f,
     46.0f, 54.0f});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  auto musa_provider = DefaultMusaExecutionProvider();
  if (musa_provider) {
    execution_providers.push_back(std::move(musa_provider));
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
             &execution_providers);
  } else {
    GTEST_SKIP() << "MUSA execution provider not available";
  }
}

// Conv test with dilation
TEST_F(MusaConvTest, ConvWithDilation) {
  OpTester test("Conv", 11);

  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("dilations", std::vector<int64_t>{2, 2});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});

  // Input: 1x1x4x4
  test.AddInput<float>("X", {1, 1, 4, 4},
    {1.0f, 2.0f, 3.0f, 4.0f,
     5.0f, 6.0f, 7.0f, 8.0f,
     9.0f, 10.0f, 11.0f, 12.0f,
     13.0f, 14.0f, 15.0f, 16.0f});

  // Weight: 1x1x2x2
  test.AddInput<float>("W", {1, 1, 2, 2},
    {1.0f, 1.0f,
     1.0f, 1.0f});

  // Expected output: 1x1x2x2
  // With dilation=2, kernel covers positions: (0,0), (0,2), (2,0), (2,2)
  // Top-left: 1+3+9+11 = 24, Top-right: 2+4+10+12 = 28
  // Bottom-left: 5+7+13+15 = 40, Bottom-right: 6+8+14+16 = 44
  test.AddOutput<float>("Y", {1, 1, 2, 2},
    {24.0f, 28.0f,
     40.0f, 44.0f});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  auto musa_provider = DefaultMusaExecutionProvider();
  if (musa_provider) {
    execution_providers.push_back(std::move(musa_provider));
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
             &execution_providers);
  } else {
    GTEST_SKIP() << "MUSA execution provider not available";
  }
}

// Conv test with group convolution (using CPU reference test data)
TEST_F(MusaConvTest, ConvWithGroup) {
  OpTester test("Conv", 11);

  test.AddAttribute("kernel_shape", std::vector<int64_t>{1, 1});
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("group", static_cast<int64_t>(2));

  // Input: 1x2x3x3 (2 input channels) - using CPU test data
  test.AddInput<float>("X", {1, 2, 3, 3},
    {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
     9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f});

  // Weight: 2x1x1x1 (2 output channels, 1 input channel per group)
  test.AddInput<float>("W", {2, 1, 1, 1},
    {1.0f, 2.0f});

  // Expected output: 1x2x3x3 - using CPU reference expected values
  // Group 1: input channel 0 * weight 1.0
  // Group 2: input channel 1 * weight 2.0
  test.AddOutput<float>("Y", {1, 2, 3, 3},
    {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
     18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f, 32.0f, 34.0f});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  auto musa_provider = DefaultMusaExecutionProvider();
  if (musa_provider) {
    execution_providers.push_back(std::move(musa_provider));
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
             &execution_providers);
  } else {
    GTEST_SKIP() << "MUSA execution provider not available";
  }
}

// Simple 1x1 convolution test
TEST_F(MusaConvTest, ConvSimple1x1) {
  OpTester test("Conv", 11);

  test.AddAttribute("kernel_shape", std::vector<int64_t>{1, 1});
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});

  // Input: 1x1x2x2
  test.AddInput<float>("X", {1, 1, 2, 2},
    {1.0f, 2.0f,
     3.0f, 4.0f});

  // Weight: 1x1x1x1 (1x1 convolution is just element-wise multiplication)
  test.AddInput<float>("W", {1, 1, 1, 1},
    {2.0f});

  // Expected output: 1x1x2x2 (each element multiplied by 2)
  test.AddOutput<float>("Y", {1, 1, 2, 2},
    {2.0f, 4.0f,
     6.0f, 8.0f});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  auto musa_provider = DefaultMusaExecutionProvider();
  if (musa_provider) {
    execution_providers.push_back(std::move(musa_provider));
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
             &execution_providers);
  } else {
    GTEST_SKIP() << "MUSA execution provider not available";
  }
}

// 1D Conv tests based on CPU tests - to improve coverage for 1D convolution cases
// Test with kernel_shape=[3], similar to the user's failing case with kernel_shape=[5]
TEST_F(MusaConvTest, Conv1D_KernelSize3_WithBias) {
  OpTester test("Conv", 11);

  test.AddAttribute("kernel_shape", std::vector<int64_t>{3});
  test.AddAttribute("strides", std::vector<int64_t>{1});
  test.AddAttribute("pads", std::vector<int64_t>{1, 0});  // asymmetric padding like CPU test
  test.AddAttribute("dilations", std::vector<int64_t>{1});
  test.AddAttribute("group", static_cast<int64_t>(1));

  // Input: 1x1x3 (simple 1D data)
  test.AddInput<float>("X", {1, 1, 3}, {1.0f, 2.0f, 3.0f});

  // Weight: 1x1x3 (1D kernel of size 3)
  test.AddInput<float>("W", {1, 1, 3}, {1.0f, 1.0f, 1.0f});

  // Bias: 1
  test.AddInput<float>("B", {1}, {0.0f});

  // Expected output: 1x1x2 (based on CPU Conv1D_asymmetric_padding test)
  // With padding [1,0]: pad 1 zero at start, 0 at end
  // Output[0] = 0*1 + 1*1 + 2*1 = 3
  // Output[1] = 1*1 + 2*1 + 3*1 = 6
  test.AddOutput<float>("Y", {1, 1, 2}, {3.0f, 6.0f});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  auto musa_provider = DefaultMusaExecutionProvider();
  if (musa_provider) {
    execution_providers.push_back(std::move(musa_provider));
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
             &execution_providers);
  } else {
    GTEST_SKIP() << "MUSA execution provider not available";
  }
}

// Test 1D Conv with kernel_shape=[5] and dilations, similar to the failing case
TEST_F(MusaConvTest, Conv1D_KernelSize5_Dilations) {
  OpTester test("Conv", 11);

  test.AddAttribute("kernel_shape", std::vector<int64_t>{5});
  test.AddAttribute("strides", std::vector<int64_t>{1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0});  // no padding like the failing case
  test.AddAttribute("dilations", std::vector<int64_t>{1});
  test.AddAttribute("group", static_cast<int64_t>(1));

  // Input: 1x1x7 (need at least kernel_size input for no padding case)
  test.AddInput<float>("X", {1, 1, 7},
    {-0.21559301f, 0.46916878f, 0.44267005f, -0.45174667f,
     -0.05216420f, 0.29067183f, 0.25101000f});

  // Weight: 1x1x5 (kernel size 5, like the failing case)
  test.AddInput<float>("W", {1, 1, 5},
    {0.24472862f, -0.12559029f, 0.44889551f, -0.31006178f, 0.13522828f});

  // No bias input (2-input Conv)

  // Expected output: 1x1x3 (7-5+1=3 output elements)
  // Computed from input and weight convolution
  test.AddOutput<float>("Y", {1, 1, 3},
    {0.220043063f, -0.0880369991f, 0.08543986f});

  // MUSA has slightly higher precision tolerance
  test.SetOutputTolerance(1.5e-4f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  auto musa_provider = DefaultMusaExecutionProvider();
  if (musa_provider) {
    execution_providers.push_back(std::move(musa_provider));
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
             &execution_providers);
  } else {
    GTEST_SKIP() << "MUSA execution provider not available";
  }
}

// Test basic 1D Conv without bias
TEST_F(MusaConvTest, Conv1D_Basic_NoBias) {
  OpTester test("Conv", 11);

  test.AddAttribute("kernel_shape", std::vector<int64_t>{1});
  test.AddAttribute("strides", std::vector<int64_t>{1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0});
  test.AddAttribute("dilations", std::vector<int64_t>{1});
  test.AddAttribute("group", static_cast<int64_t>(1));

  // Input: 1x1x7 (from CPU Conv1D_1 test)
  test.AddInput<float>("X", {1, 1, 7},
    {-0.21559301f, 0.46916878f, 0.44267005f, -0.45174667f,
     -0.05216420f, 0.29067183f, 0.25101000f});

  // Weight: 1x1x1 (simple 1D kernel)
  test.AddInput<float>("W", {1, 1, 1}, {0.24472862f});

  // Expected output: 1x1x7 (element-wise multiplication for kernel size 1)
  test.AddOutput<float>("Y", {1, 1, 7},
    {-0.05276191f, 0.1148465f, 0.108334035f,  -0.1105553f,
     -0.01276127f, 0.0711666f, 0.06142933f});

  // MUSA has slightly higher precision tolerance
  test.SetOutputTolerance(1.2e-4f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  auto musa_provider = DefaultMusaExecutionProvider();
  if (musa_provider) {
    execution_providers.push_back(std::move(musa_provider));
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
             &execution_providers);
  } else {
    GTEST_SKIP() << "MUSA execution provider not available";
  }
}

} // namespace test
} // namespace onnxruntime
