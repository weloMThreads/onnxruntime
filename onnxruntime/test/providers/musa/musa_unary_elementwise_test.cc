// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_session_options_config_keys.h"
#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

#include <limits>

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

void RunSqrtNoFallback(OpTester& test) {
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(musa_provider));
  test.Run(MakeNoFallbackSessionOptions(),
           OpTester::ExpectResult::kExpectSuccess,
           "",
           {},
           nullptr,
           &execution_providers);
}

}  // namespace

TEST(MusaUnaryElementwiseSqrtTest, SqrtDoubleNoCpuFallback) {
  OpTester test("Sqrt", 13);
  test.AddInput<double>("X", {2, 3},
                        {0.0,
                         1.0,
                         4.0,
                         9.0,
                         1.0e300,
                         std::numeric_limits<double>::infinity()});
  test.AddOutput<double>("Y", {2, 3},
                         {0.0,
                          1.0,
                          2.0,
                          3.0,
                          1.0e150,
                          std::numeric_limits<double>::infinity()});

  RunSqrtNoFallback(test);
}

TEST(MusaUnaryElementwiseSqrtTest, SqrtFloatSpecialValuesNoCpuFallback) {
  OpTester test("Sqrt", 13);
  const float nan = std::numeric_limits<float>::quiet_NaN();
  test.AddInput<float>("X", {2, 4},
                       {0.0f,
                        1.0f,
                        4.0f,
                        1.0e20f,
                        std::numeric_limits<float>::infinity(),
                        -1.0f,
                        nan,
                        65504.0f});
  test.AddOutput<float>("Y", {2, 4},
                        {0.0f,
                         1.0f,
                         2.0f,
                         1.0e10f,
                         std::numeric_limits<float>::infinity(),
                         nan,
                         nan,
                         255.93749f});

  RunSqrtNoFallback(test);
}

TEST(MusaUnaryElementwiseSqrtTest, SqrtFp16NoCpuFallback) {
  OpTester test("Sqrt", 13);
  test.AddInput<MLFloat16>("X", {2, 3},
                           ToFloat16({0.0f,
                                      1.0f,
                                      4.0f,
                                      9.0f,
                                      256.0f,
                                      1024.0f}));
  test.AddOutput<MLFloat16>("Y", {2, 3},
                            ToFloat16({0.0f,
                                       1.0f,
                                       2.0f,
                                       3.0f,
                                       16.0f,
                                       32.0f}));

  RunSqrtNoFallback(test);
}

TEST(MusaUnaryElementwiseSqrtTest, SqrtBFloat16NoCpuFallback) {
  OpTester test("Sqrt", 13);
  test.AddInput<BFloat16>("X", {2, 3},
                          MakeBFloat16({0.0f,
                                        1.0f,
                                        4.0f,
                                        9.0f,
                                        256.0f,
                                        1024.0f}));
  test.AddOutput<BFloat16>("Y", {2, 3},
                           MakeBFloat16({0.0f,
                                         1.0f,
                                         2.0f,
                                         3.0f,
                                         16.0f,
                                         32.0f}));

  RunSqrtNoFallback(test);
}

TEST(MusaUnaryElementwiseSqrtTest, SqrtMultiShapeNoCpuFallback) {
  OpTester test("Sqrt", 13);
  test.AddInput<float>("X", {2, 1, 3},
                       {1.0f, 4.0f, 9.0f,
                        16.0f, 25.0f, 36.0f});
  test.AddOutput<float>("Y", {2, 1, 3},
                        {1.0f, 2.0f, 3.0f,
                         4.0f, 5.0f, 6.0f});

  RunSqrtNoFallback(test);
}

TEST(MusaUnaryElementwiseSqrtTest, SqrtZeroElementNoCpuFallback) {
  OpTester test("Sqrt", 13);
  test.AddInput<float>("X", {2, 0, 3}, {});
  test.AddOutput<float>("Y", {2, 0, 3}, {});

  RunSqrtNoFallback(test);
}

}  // namespace test
}  // namespace onnxruntime
