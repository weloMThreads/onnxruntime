// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_session_options_config_keys.h"
#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

namespace {

SessionOptions MakeNoFallbackSessionOptions() {
  SessionOptions so;
  so.session_log_severity_level = 0;
  so.session_log_verbosity_level = 2;
  ORT_THROW_IF_ERROR(so.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  return so;
}

}  // namespace

TEST(MusaReduceTest, ReduceMeanBceHiddenFp16NoCpuFallback) {
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  constexpr int64_t batch = 8;
  constexpr int64_t seq = 1;
  constexpr int64_t hidden = 768;
  std::vector<float> input(static_cast<size_t>(batch * seq * hidden));
  std::vector<float> expected(static_cast<size_t>(batch * seq));

  for (int64_t b = 0; b < batch; ++b) {
    float sum = 0.0f;
    for (int64_t h = 0; h < hidden; ++h) {
      const float value = static_cast<float>((b * 17 + h * 13) % 251 - 125) / 64.0f;
      input[static_cast<size_t>(b * hidden + h)] = value;
      sum += value;
    }
    expected[static_cast<size_t>(b)] = sum / static_cast<float>(hidden);
  }

  OpTester test("ReduceMean", 13);
  test.AddAttribute("axes", std::vector<int64_t>{-1});
  test.AddAttribute("keepdims", static_cast<int64_t>(1));
  test.AddInput<MLFloat16>("data", {batch, seq, hidden}, ToFloat16(input));
  test.AddOutput<MLFloat16>("reduced", {batch, seq, 1}, ToFloat16(expected));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(musa_provider));
  test.Run(MakeNoFallbackSessionOptions(),
           OpTester::ExpectResult::kExpectSuccess,
           "",
           {},
           nullptr,
           &execution_providers);
}

TEST(MusaReduceTest, ReduceL2KeepDimsNoCpuFallback) {
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  OpTester test("ReduceL2", 13);
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", static_cast<int64_t>(1));
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,
                        5.0f, 6.0f,
                        7.0f, 8.0f,
                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 2, 1},
                        {15.71623325f, 20.07485962f});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(musa_provider));
  test.Run(MakeNoFallbackSessionOptions(),
           OpTester::ExpectResult::kExpectSuccess,
           "",
           {},
           nullptr,
           &execution_providers);
}

TEST(MusaReduceTest, ReduceL2AxesInputNoCpuFallback) {
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  OpTester test("ReduceL2", 18);
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddAttribute("noop_with_empty_axes", static_cast<int64_t>(0));
  test.AddInput<float>("data", {2, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,
                        5.0f, 6.0f,
                        7.0f, 8.0f});
  test.AddInput<int64_t>("axes", {2}, {0, 2});
  test.AddOutput<float>("reduced", {2},
                        {8.12403870f, 11.74734020f});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(musa_provider));
  test.Run(MakeNoFallbackSessionOptions(),
           OpTester::ExpectResult::kExpectSuccess,
           "",
           {},
           nullptr,
           &execution_providers);
}

TEST(MusaPoolTest, GlobalAveragePool3DNoCpuFallback) {
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  OpTester test("GlobalAveragePool", 1);
  test.AddInput<float>("X", {1, 2, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        5.0f, 6.0f, 7.0f, 8.0f});
  test.AddOutput<float>("Y", {1, 2, 1}, {2.5f, 6.5f});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(musa_provider));
  test.Run(MakeNoFallbackSessionOptions(),
           OpTester::ExpectResult::kExpectSuccess,
           "",
           {},
           nullptr,
           &execution_providers);
}

TEST(MusaPoolTest, MaxPool2DNoCpuFallback) {
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  OpTester test("MaxPool", 12);
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});
  test.AddAttribute("strides", std::vector<int64_t>{2, 2});
  test.AddInput<float>("X", {1, 1, 4, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        5.0f, 6.0f, 7.0f, 8.0f,
                        9.0f, 10.0f, 11.0f, 12.0f,
                        13.0f, 14.0f, 15.0f, 16.0f});
  test.AddOutput<float>("Y", {1, 1, 2, 2}, {6.0f, 8.0f, 14.0f, 16.0f});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(musa_provider));
  test.Run(MakeNoFallbackSessionOptions(),
           OpTester::ExpectResult::kExpectSuccess,
           "",
           {},
           nullptr,
           &execution_providers);
}

TEST(MusaPoolTest, GlobalMaxPoolNoCpuFallback) {
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  OpTester test("GlobalMaxPool", 1);
  test.AddInput<float>("X", {1, 2, 4}, {1.0f, 3.0f, 2.0f, 4.0f, 8.0f, 5.0f, 7.0f, 6.0f});
  test.AddOutput<float>("Y", {1, 2, 1}, {4.0f, 8.0f});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(musa_provider));
  test.Run(MakeNoFallbackSessionOptions(),
           OpTester::ExpectResult::kExpectSuccess,
           "",
           {},
           nullptr,
           &execution_providers);
}

TEST(MusaTensorTest, FlattenNoCpuFallback) {
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  OpTester test("Flatten", 13);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddInput<float>("input", {2, 3, 4}, 
                       {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
                        13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f});
  test.AddOutput<float>("output", {2, 12},
                        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
                         13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(musa_provider));
  test.Run(MakeNoFallbackSessionOptions(),
           OpTester::ExpectResult::kExpectSuccess,
           "",
           {},
           nullptr,
           &execution_providers);
}

}  // namespace test
}  // namespace onnxruntime
