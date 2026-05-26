// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <vector>

#include "core/graph/constants.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

namespace {

float QuickGeluReference(float x, float alpha) {
  const float scaled = alpha * x;
  const float sigmoid = scaled >= 0.0f ? 1.0f / (1.0f + std::exp(-scaled))
                                       : 1.0f - 1.0f / (1.0f + std::exp(scaled));
  return x * sigmoid;
}

void RunQuickGeluFloatTest(float alpha) {
  std::vector<float> input_values{-3.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 4.0f};
  std::vector<float> expected_values;
  expected_values.reserve(input_values.size());
  for (float input_value : input_values) {
    expected_values.push_back(QuickGeluReference(input_value, alpha));
  }

  OpTester op_tester("QuickGelu", 1, kMSDomain);
  op_tester.AddAttribute("alpha", alpha);
  op_tester.AddInput<float>("X", {static_cast<int64_t>(input_values.size())}, input_values);
  op_tester.AddOutput<float>("Y", {static_cast<int64_t>(expected_values.size())}, expected_values);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  execution_providers.push_back(std::move(musa_provider));
  op_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

void RunQuickGeluMlFloat16Test(float alpha) {
  std::vector<float> input_values{-3.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 4.0f};
  std::vector<float> expected_float_values;
  expected_float_values.reserve(input_values.size());
  for (float input_value : input_values) {
    expected_float_values.push_back(QuickGeluReference(input_value, alpha));
  }

  std::vector<MLFloat16> input_fp16(input_values.size());
  std::vector<MLFloat16> expected_fp16(expected_float_values.size());
  ConvertFloatToMLFloat16(input_values.data(), input_fp16.data(), static_cast<int>(input_values.size()));
  ConvertFloatToMLFloat16(expected_float_values.data(), expected_fp16.data(), static_cast<int>(expected_float_values.size()));

  OpTester op_tester("QuickGelu", 1, kMSDomain);
  op_tester.AddAttribute("alpha", alpha);
  op_tester.SetOutputTolerance(0.005f);
  op_tester.AddInput<MLFloat16>("X", {static_cast<int64_t>(input_fp16.size())}, input_fp16);
  op_tester.AddOutput<MLFloat16>("Y", {static_cast<int64_t>(expected_fp16.size())}, expected_fp16);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  execution_providers.push_back(std::move(musa_provider));
  op_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

}  // namespace

TEST(MusaQuickGeluTest, QuickGeluFloat) {
  RunQuickGeluFloatTest(1.702f);
}

TEST(MusaQuickGeluTest, QuickGeluMLFloat16) {
  RunQuickGeluMlFloat16Test(1.0f);
}

}  // namespace test
}  // namespace onnxruntime
