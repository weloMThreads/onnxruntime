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
  ORT_THROW_IF_ERROR(so.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  return so;
}

bool MakeMusaExecutionProviders(std::vector<std::unique_ptr<IExecutionProvider>>& execution_providers) {
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    return false;
  }
  execution_providers.push_back(std::move(musa_provider));
  return true;
}

}  // namespace

TEST(MusaBceBridgeOpsTest, GemmOpset12Fp16NoCpuFallback) {
  OpTester test("Gemm", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute("transA", int64_t{0});
  test.AddAttribute("transB", int64_t{0});
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<MLFloat16>("A", {2, 3},
                           {MLFloat16(1.0f), MLFloat16(2.0f), MLFloat16(3.0f),
                            MLFloat16(4.0f), MLFloat16(5.0f), MLFloat16(6.0f)});
  test.AddInput<MLFloat16>("B", {3, 2},
                           {MLFloat16(1.0f), MLFloat16(2.0f), MLFloat16(3.0f),
                            MLFloat16(4.0f), MLFloat16(5.0f), MLFloat16(6.0f)});
  test.AddInput<MLFloat16>("C", {2}, {MLFloat16(1.0f), MLFloat16(1.0f)});
  test.AddOutput<MLFloat16>("Y", {2, 2},
                            {MLFloat16(23.0f), MLFloat16(29.0f),
                             MLFloat16(50.0f), MLFloat16(65.0f)});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  if (!MakeMusaExecutionProviders(execution_providers)) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }
  test.Run(MakeNoFallbackSessionOptions(),
           OpTester::ExpectResult::kExpectSuccess,
           "",
           {},
           nullptr,
           &execution_providers);
}

TEST(MusaBceBridgeOpsTest, CumSumOpset12Int32Axis1NoCpuFallback) {
  OpTester test("CumSum", 12, onnxruntime::kOnnxDomain);
  test.AddInput<int32_t>("x", {2, 4}, {1, 0, 1, 1, 0, 1, 1, 0});
  test.AddInput<int32_t>("axis", {}, {1});
  test.AddOutput<int32_t>("y", {2, 4}, {1, 1, 2, 3, 0, 1, 2, 2});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  if (!MakeMusaExecutionProviders(execution_providers)) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }
  test.Run(MakeNoFallbackSessionOptions(),
           OpTester::ExpectResult::kExpectSuccess,
           "",
           {},
           nullptr,
           &execution_providers);
}

TEST(MusaBceBridgeOpsTest, CumSumOpset11Int32Axis1ExclusiveReverseNoCpuFallback) {
  OpTester test("CumSum", 11, onnxruntime::kOnnxDomain);
  test.AddAttribute("exclusive", int64_t{1});
  test.AddAttribute("reverse", int64_t{1});
  test.AddInput<int32_t>("x", {2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
  test.AddInput<int32_t>("axis", {}, {1});
  test.AddOutput<int32_t>("y", {2, 4}, {9, 7, 4, 0, 21, 15, 8, 0});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  if (!MakeMusaExecutionProviders(execution_providers)) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }
  test.Run(MakeNoFallbackSessionOptions(),
           OpTester::ExpectResult::kExpectSuccess,
           "",
           {},
           nullptr,
           &execution_providers);
}

TEST(MusaBceBridgeOpsTest, CumSumOpset14Fp16NoCpuFallback) {
  OpTester test("CumSum", 14, onnxruntime::kOnnxDomain);
  test.AddInput<MLFloat16>("x", {1, 4},
                           {MLFloat16(1.0f), MLFloat16(2.0f),
                            MLFloat16(3.0f), MLFloat16(4.0f)});
  test.AddInput<int32_t>("axis", {}, {1});
  test.AddOutput<MLFloat16>("y", {1, 4},
                            {MLFloat16(1.0f), MLFloat16(3.0f),
                             MLFloat16(6.0f), MLFloat16(10.0f)});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  if (!MakeMusaExecutionProviders(execution_providers)) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }
  test.Run(MakeNoFallbackSessionOptions(),
           OpTester::ExpectResult::kExpectSuccess,
           "",
           {},
           nullptr,
           &execution_providers);
}

TEST(MusaBceBridgeOpsTest, NotBoolNoCpuFallback) {
  OpTester test("Not", 1, onnxruntime::kOnnxDomain);
  test.AddInput<bool>("X", {2, 4}, {false, true, false, true, true, true, false, false});
  test.AddOutput<bool>("Y", {2, 4}, {true, false, true, false, false, false, true, true});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  if (!MakeMusaExecutionProviders(execution_providers)) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }
  test.Run(MakeNoFallbackSessionOptions(),
           OpTester::ExpectResult::kExpectSuccess,
           "",
           {},
           nullptr,
           &execution_providers);
}

}  // namespace test
}  // namespace onnxruntime
