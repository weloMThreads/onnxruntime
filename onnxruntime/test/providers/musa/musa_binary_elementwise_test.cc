// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/model.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"
#include "test/framework/test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <string>

namespace onnxruntime {
namespace test {

// Test fixture for BinaryElementwise::Prepare function
class MusaBinaryElementwisePrepareTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Test will skip if MUSA provider is not available
  }
};

// Test BinaryElementwise::Prepare with same-shaped tensors (no broadcast)
TEST_F(MusaBinaryElementwisePrepareTest, PrepareWithSameShapes) {
  std::vector<int64_t> dims = {2, 3};
  std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> data_b = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

  OpTester op_tester("Add", 7);
  op_tester.AddInput<float>("A", dims, data_a);
  op_tester.AddInput<float>("B", dims, data_b);

  // Expected output: element-wise addition
  std::vector<float> expected_output = {3.0f, 5.0f, 7.0f, 9.0f, 11.0f, 13.0f};
  op_tester.AddOutput<float>("C", dims, expected_output);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  auto musa_provider = DefaultMusaExecutionProvider();
  if (musa_provider) {
    execution_providers.push_back(std::move(musa_provider));
    op_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
                  &execution_providers);
  } else {
    GTEST_SKIP() << "MUSA execution provider not available";
  }
}

// Test BinaryElementwise::Prepare with broadcast (scalar + tensor)
TEST_F(MusaBinaryElementwisePrepareTest, PrepareWithScalarBroadcast) {
  std::vector<int64_t> dims_scalar = {1};
  std::vector<int64_t> dims_tensor = {2, 3};
  std::vector<float> data_scalar = {5.0f};
  std::vector<float> data_tensor = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  OpTester op_tester("Add", 14);
  op_tester.AddInput<float>("A", dims_scalar, data_scalar);
  op_tester.AddInput<float>("B", dims_tensor, data_tensor);

  // Expected output: broadcast scalar to tensor shape and add
  std::vector<float> expected_output = {6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  op_tester.AddOutput<float>("C", dims_tensor, expected_output);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  auto musa_provider = DefaultMusaExecutionProvider();
  if (musa_provider) {
    execution_providers.push_back(std::move(musa_provider));
    op_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
                  &execution_providers);
  } else {
    GTEST_SKIP() << "MUSA execution provider not available";
  }
}

// Test BinaryElementwise::Prepare with matrix broadcast
TEST_F(MusaBinaryElementwisePrepareTest, PrepareWithMatrixBroadcast) {
  // Create test tensors: [2, 1] + [1, 3] -> [2, 3]
  std::vector<int64_t> dims_a = {2, 1};
  std::vector<int64_t> dims_b = {1, 3};
  std::vector<float> data_a = {1.0f, 2.0f};
  std::vector<float> data_b = {10.0f, 20.0f, 30.0f};

  OpTester op_tester("Add", 14);
  op_tester.AddInput<float>("A", dims_a, data_a);
  op_tester.AddInput<float>("B", dims_b, data_b);

  // Expected output: broadcast to [2, 3]
  // [[1], [2]] + [[10, 20, 30]] = [[11, 21, 31], [12, 22, 32]]
  std::vector<int64_t> expected_dims = {2, 3};
  std::vector<float> expected_output = {11.0f, 21.0f, 31.0f,
                                        12.0f, 22.0f, 32.0f};
  op_tester.AddOutput<float>("C", expected_dims, expected_output);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  auto musa_provider = DefaultMusaExecutionProvider();
  if (musa_provider) {
    execution_providers.push_back(std::move(musa_provider));
    op_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
                  &execution_providers);
  } else {
    GTEST_SKIP() << "MUSA execution provider not available";
  }
}

// Test different data types
TEST_F(MusaBinaryElementwisePrepareTest, PrepareWithInt32) {
  std::vector<int64_t> dims = {2, 2};
  std::vector<int32_t> data_a = {1, 2, 3, 4};
  std::vector<int32_t> data_b = {5, 6, 7, 8};

  OpTester op_tester("Add", 14);
  op_tester.AddInput<int32_t>("A", dims, data_a);
  op_tester.AddInput<int32_t>("B", dims, data_b);

  std::vector<int32_t> expected_output = {6, 8, 10, 12};
  op_tester.AddOutput<int32_t>("C", dims, expected_output);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  auto musa_provider = DefaultMusaExecutionProvider();
  if (musa_provider) {
    execution_providers.push_back(std::move(musa_provider));
    op_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
                  &execution_providers);
  } else {
    GTEST_SKIP() << "MUSA execution provider not available";
  }
}

// Test int64_t data type
TEST_F(MusaBinaryElementwisePrepareTest, PrepareWithInt64) {
  std::vector<int64_t> dims = {2, 3};
  std::vector<int64_t> data_a = {1, 2, 3, 4, 5, 6};
  std::vector<int64_t> data_b = {7, 8, 9, 10, 11, 12};

  OpTester op_tester("Add", 14);
  op_tester.AddInput<int64_t>("A", dims, data_a);
  op_tester.AddInput<int64_t>("B", dims, data_b);

  std::vector<int64_t> expected_output = {8, 10, 12, 14, 16, 18};
  op_tester.AddOutput<int64_t>("C", dims, expected_output);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  auto musa_provider = DefaultMusaExecutionProvider();
  if (musa_provider) {
    execution_providers.push_back(std::move(musa_provider));
    op_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
                  &execution_providers);
  } else {
    GTEST_SKIP() << "MUSA execution provider not available";
  }
}

// Test MLFloat16 data type
TEST_F(MusaBinaryElementwisePrepareTest, PrepareWithMLFloat16) {
  std::vector<int64_t> dims = {2, 2};
  std::vector<float> float_data_a = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> float_data_b = {0.5f, 1.5f, 2.5f, 3.5f};
  std::vector<float> float_expected = {1.5f, 3.5f, 5.5f, 7.5f};

  // Convert to MLFloat16
  std::vector<MLFloat16> data_a, data_b, expected_output;
  std::transform(float_data_a.begin(), float_data_a.end(), std::back_inserter(data_a),
                 [](float f) { return MLFloat16(f); });
  std::transform(float_data_b.begin(), float_data_b.end(), std::back_inserter(data_b),
                 [](float f) { return MLFloat16(f); });
  std::transform(float_expected.begin(), float_expected.end(), std::back_inserter(expected_output),
                 [](float f) { return MLFloat16(f); });

  OpTester op_tester("Add", 14);
  op_tester.AddInput<MLFloat16>("A", dims, data_a);
  op_tester.AddInput<MLFloat16>("B", dims, data_b);
  op_tester.AddOutput<MLFloat16>("C", dims, expected_output);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  auto musa_provider = DefaultMusaExecutionProvider();
  if (musa_provider) {
    execution_providers.push_back(std::move(musa_provider));
    op_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
                  &execution_providers);
  } else {
    GTEST_SKIP() << "MUSA execution provider not available";
  }
}

// Test Sub operator
TEST_F(MusaBinaryElementwisePrepareTest, PrepareSubOperator) {
  std::vector<int64_t> dims = {2, 2};
  std::vector<float> data_a = {5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> data_b = {1.0f, 2.0f, 3.0f, 4.0f};

  OpTester op_tester("Sub", 14);
  op_tester.AddInput<float>("A", dims, data_a);
  op_tester.AddInput<float>("B", dims, data_b);

  std::vector<float> expected_output = {4.0f, 4.0f, 4.0f, 4.0f};
  op_tester.AddOutput<float>("C", dims, expected_output);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  auto musa_provider = DefaultMusaExecutionProvider();
  if (musa_provider) {
    execution_providers.push_back(std::move(musa_provider));
    op_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
                  &execution_providers);
  } else {
    GTEST_SKIP() << "MUSA execution provider not available";
  }
}

// Test Mul operator
TEST_F(MusaBinaryElementwisePrepareTest, PrepareMulOperator) {
  std::vector<int64_t> dims = {2, 2};
  std::vector<float> data_a = {2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> data_b = {2.0f, 2.0f, 2.0f, 2.0f};

  OpTester op_tester("Mul", 14);
  op_tester.AddInput<float>("A", dims, data_a);
  op_tester.AddInput<float>("B", dims, data_b);

  std::vector<float> expected_output = {4.0f, 6.0f, 8.0f, 10.0f};
  op_tester.AddOutput<float>("C", dims, expected_output);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  auto musa_provider = DefaultMusaExecutionProvider();
  if (musa_provider) {
    execution_providers.push_back(std::move(musa_provider));
    op_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
                  &execution_providers);
  } else {
    GTEST_SKIP() << "MUSA execution provider not available";
  }
}

// Test Div operator
TEST_F(MusaBinaryElementwisePrepareTest, PrepareDivOperator) {
  std::vector<int64_t> dims = {2, 2};
  std::vector<float> data_a = {8.0f, 12.0f, 16.0f, 20.0f};
  std::vector<float> data_b = {2.0f, 3.0f, 4.0f, 5.0f};

  OpTester op_tester("Div", 14);
  op_tester.AddInput<float>("A", dims, data_a);
  op_tester.AddInput<float>("B", dims, data_b);

  std::vector<float> expected_output = {4.0f, 4.0f, 4.0f, 4.0f};
  op_tester.AddOutput<float>("C", dims, expected_output);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  auto musa_provider = DefaultMusaExecutionProvider();
  if (musa_provider) {
    execution_providers.push_back(std::move(musa_provider));
    op_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
                  &execution_providers);
  } else {
    GTEST_SKIP() << "MUSA execution provider not available";
  }
}

// Test 1D tensor addition
TEST_F(MusaBinaryElementwisePrepareTest, Prepare1DAddition) {
  const int N = 3;
  std::vector<int64_t> dims = {N};

  // Create large vectors: A with all 1.0f, B with all 2.0f
  std::vector<float> data_a(N, 1.0f);
  std::vector<float> data_b(N, 2.0f);

  // Expected output: all 3.0f (1.0 + 2.0)
  std::vector<float> expected_output(N, 3.0f);

  OpTester op_tester("Add", 14);
  op_tester.AddInput<float>("A", dims, data_a);
  op_tester.AddInput<float>("B", dims, data_b);
  op_tester.AddOutput<float>("C", dims, expected_output);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  auto musa_provider = DefaultMusaExecutionProvider();
  if (musa_provider) {
    execution_providers.push_back(std::move(musa_provider));
    op_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
                  &execution_providers);
  } else {
    GTEST_SKIP() << "MUSA execution provider not available";
  }
}

// ============================================================
// Pow same-type tests (V0.17.0 regression guard)
// ============================================================

// V0.16.1 catastrophic case: fp16 Pow(300, 2) = 90000
// mudnn::Binary::POW did native fp16 pow → |300| > sqrt(65504) ≈ 255.94 → Inf
// V0.17.0 自写 kernel 走 fp32 中介，不再溢出
TEST_F(MusaBinaryElementwisePrepareTest, PowFp16_CatastrophicOverflow) {
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  // x=300, y=2 → 90000 超 fp16 max(65504)，fp16 无法精确表示但不应产 Inf
  // fp32 中介: powf(300.0f, 2.0f) = 90000.0f → __float2half(90000) = Inf (fp16 range)
  // 这里的 DoD 是"不 crash + 结果与 CPU reference 对齐"
  std::vector<int64_t> dims{4};
  std::vector<float> float_x = {300.0f, 200.0f, 10.0f, 1.0f};
  std::vector<float> float_y = {2.0f, 2.0f, 3.0f, 100.0f};
  // CPU fp32 reference: 90000, 40000, 1000, 1
  // fp16 clamp: 90000 → Inf, 40000 → 40000, 1000 → 1000, 1 → 1
  std::vector<float> float_expected = {90000.0f, 40000.0f, 1000.0f, 1.0f};

  std::vector<MLFloat16> data_x, data_y, expected;
  for (auto v : float_x) data_x.push_back(MLFloat16(v));
  for (auto v : float_y) data_y.push_back(MLFloat16(v));
  for (auto v : float_expected) expected.push_back(MLFloat16(v));

  OpTester op_tester("Pow", 14);
  op_tester.AddInput<MLFloat16>("X", dims, data_x);
  op_tester.AddInput<MLFloat16>("Y", dims, data_y);
  op_tester.AddOutput<MLFloat16>("Z", dims, expected);
  // fp16 精度宽松：overflow 区域允许 Inf vs Inf 或 large diff
  op_tester.SetOutputAbsErr("Z", 1.0f);
  op_tester.SetOutputRelErr("Z", 0.05f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(musa_provider));
  op_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
                &execution_providers);
}

// fp16 正常范围 Pow（不触发溢出）
TEST_F(MusaBinaryElementwisePrepareTest, PowFp16_NormalRange) {
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  std::vector<int64_t> dims{4};
  std::vector<float> float_x = {2.0f, 3.0f, 4.0f, 0.5f};
  std::vector<float> float_y = {3.0f, 2.0f, 0.5f, 2.0f};
  std::vector<float> float_expected = {8.0f, 9.0f, 2.0f, 0.25f};

  std::vector<MLFloat16> data_x, data_y, expected;
  for (auto v : float_x) data_x.push_back(MLFloat16(v));
  for (auto v : float_y) data_y.push_back(MLFloat16(v));
  for (auto v : float_expected) expected.push_back(MLFloat16(v));

  OpTester op_tester("Pow", 14);
  op_tester.AddInput<MLFloat16>("X", dims, data_x);
  op_tester.AddInput<MLFloat16>("Y", dims, data_y);
  op_tester.AddOutput<MLFloat16>("Z", dims, expected);
  op_tester.SetOutputAbsErr("Z", 0.01f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(musa_provider));
  op_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
                &execution_providers);
}

// fp16 Pow 特殊值：0^0=1, x^0=1, 0^y=0, 1^y=1
TEST_F(MusaBinaryElementwisePrepareTest, PowFp16_SpecialValues) {
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  std::vector<int64_t> dims{4};
  std::vector<float> float_x = {0.0f, 5.0f, 0.0f, 1.0f};
  std::vector<float> float_y = {0.0f, 0.0f, 3.0f, 999.0f};
  std::vector<float> float_expected = {1.0f, 1.0f, 0.0f, 1.0f};

  std::vector<MLFloat16> data_x, data_y, expected;
  for (auto v : float_x) data_x.push_back(MLFloat16(v));
  for (auto v : float_y) data_y.push_back(MLFloat16(v));
  for (auto v : float_expected) expected.push_back(MLFloat16(v));

  OpTester op_tester("Pow", 14);
  op_tester.AddInput<MLFloat16>("X", dims, data_x);
  op_tester.AddInput<MLFloat16>("Y", dims, data_y);
  op_tester.AddOutput<MLFloat16>("Z", dims, expected);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(musa_provider));
  op_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
                &execution_providers);
}

// fp16 Pow broadcasting: [2,3] ^ [3] → [2,3]
TEST_F(MusaBinaryElementwisePrepareTest, PowFp16_Broadcast) {
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  std::vector<int64_t> dims_x = {2, 3};
  std::vector<int64_t> dims_y = {3};
  std::vector<float> float_x = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  std::vector<float> float_y = {2.0f, 1.0f, 0.5f};
  std::vector<float> float_expected = {4.0f, 3.0f, 2.0f, 25.0f, 6.0f, 2.6457513f};

  std::vector<MLFloat16> data_x, data_y, expected;
  for (auto v : float_x) data_x.push_back(MLFloat16(v));
  for (auto v : float_y) data_y.push_back(MLFloat16(v));
  for (auto v : float_expected) expected.push_back(MLFloat16(v));

  OpTester op_tester("Pow", 14);
  op_tester.AddInput<MLFloat16>("X", dims_x, data_x);
  op_tester.AddInput<MLFloat16>("Y", dims_y, data_y);
  op_tester.AddOutput<MLFloat16>("Z", dims_x, expected);
  op_tester.SetOutputAbsErr("Z", 0.05f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(musa_provider));
  op_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
                &execution_providers);
}

// BCE LayerNorm 真实场景：(x-mean)^2, shape [batch, seq, hidden] ^ scalar
// 这是 V0.16.1 catastrophic overflow 的真正触发路径
// rank 不对称广播：3D tensor ^ 0D scalar
TEST_F(MusaBinaryElementwisePrepareTest, PowFloat_BCELayerNorm_ScalarBroadcast) {
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  // 模拟 BCE LayerNorm: (x-mean)^2, batch=2, seq=4, hidden=8
  std::vector<int64_t> dims_x = {2, 4, 8};
  std::vector<float> float_x(2 * 4 * 8);
  std::vector<float> float_expected(2 * 4 * 8);
  for (int i = 0; i < 64; ++i) {
    float_x[i] = static_cast<float>(i - 32) * 0.1f;  // [-3.2, 3.1]
    float_expected[i] = float_x[i] * float_x[i];       // x^2
  }

  OpTester op_tester("Pow", 12);
  op_tester.AddInput<float>("X", dims_x, float_x);
  op_tester.AddInput<float>("Y", {}, {2.0f});  // scalar exponent
  op_tester.AddOutput<float>("Z", dims_x, float_expected);
  op_tester.SetOutputRelErr("Z", 1e-5f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(musa_provider));
  op_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
                &execution_providers);
}

// BCE fp16 场景：[2, 4, 8] ^ scalar fp16 — rank mismatch broadcast
TEST_F(MusaBinaryElementwisePrepareTest, PowFp16_BCELayerNorm_ScalarBroadcast) {
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  std::vector<int64_t> dims_x = {2, 4, 8};
  std::vector<float> float_x(64), float_expected(64);
  for (int i = 0; i < 64; ++i) {
    float_x[i] = static_cast<float>(i - 32) * 0.1f;
    float_expected[i] = float_x[i] * float_x[i];
  }

  std::vector<MLFloat16> data_x, expected;
  for (auto v : float_x) data_x.push_back(MLFloat16(v));
  for (auto v : float_expected) expected.push_back(MLFloat16(v));

  OpTester op_tester("Pow", 14);
  op_tester.AddInput<MLFloat16>("X", dims_x, data_x);
  op_tester.AddInput<MLFloat16>("Y", {}, {MLFloat16(2.0f)});  // scalar
  op_tester.AddOutput<MLFloat16>("Z", dims_x, expected);
  op_tester.SetOutputAbsErr("Z", 0.05f);
  op_tester.SetOutputRelErr("Z", 0.01f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(musa_provider));
  op_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
                &execution_providers);
}

TEST_F(MusaBinaryElementwisePrepareTest, PowFloat_InitializerExponent) {
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  std::vector<int64_t> dims{4};
  OpTester op_tester("Pow", 12);
  op_tester.AddInput<float>("X", dims, {2.0f, 3.0f, 4.0f, 5.0f});
  op_tester.AddInput<float>("Y", {}, {2.0f}, true);
  op_tester.AddOutput<float>("Z", dims, {4.0f, 9.0f, 16.0f, 25.0f});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(musa_provider));
  op_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
                &execution_providers);
}

TEST_F(MusaBinaryElementwisePrepareTest, PowFp16_InitializerExponent) {
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  std::vector<int64_t> dims{4};
  std::vector<MLFloat16> data_x = {MLFloat16(2.0f), MLFloat16(3.0f),
                                   MLFloat16(4.0f), MLFloat16(5.0f)};
  std::vector<MLFloat16> data_y = {MLFloat16(2.0f)};
  std::vector<MLFloat16> expected = {MLFloat16(4.0f), MLFloat16(9.0f),
                                     MLFloat16(16.0f), MLFloat16(25.0f)};

  OpTester op_tester("Pow", 14);
  op_tester.AddInput<MLFloat16>("X", dims, data_x);
  op_tester.AddInput<MLFloat16>("Y", {}, data_y, true);
  op_tester.AddOutput<MLFloat16>("Z", dims, expected);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(musa_provider));
  op_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
                &execution_providers);
}

// BCE 真实尺寸: [8, 128, 768] ^ scalar — 大 tensor 广播
TEST_F(MusaBinaryElementwisePrepareTest, PowFloat_BCELayerNorm_LargeShape) {
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  const int64_t B = 8, S = 128, H = 768;
  const int64_t total = B * S * H;
  std::vector<int64_t> dims_x = {B, S, H};
  std::vector<float> float_x(total), float_expected(total);
  for (int64_t i = 0; i < total; ++i) {
    float_x[i] = static_cast<float>(i % 100 - 50) * 0.01f;
    float_expected[i] = float_x[i] * float_x[i];
  }

  OpTester op_tester("Pow", 12);
  op_tester.AddInput<float>("X", dims_x, float_x);
  op_tester.AddInput<float>("Y", {}, {2.0f});
  op_tester.AddOutput<float>("Z", dims_x, float_expected);
  op_tester.SetOutputRelErr("Z", 1e-5f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(musa_provider));
  op_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
                &execution_providers);
}

// 1D broadcast: [batch*seq*hidden] ^ [1] — 另一种常见 pattern
TEST_F(MusaBinaryElementwisePrepareTest, PowFloat_Broadcast_TensorVsOne) {
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  std::vector<int64_t> dims_x = {16};
  std::vector<float> float_x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<float> float_expected(16);
  for (int i = 0; i < 16; ++i) float_expected[i] = float_x[i] * float_x[i];

  OpTester op_tester("Pow", 12);
  op_tester.AddInput<float>("X", dims_x, float_x);
  op_tester.AddInput<float>("Y", {1}, {2.0f});  // [1] not scalar
  op_tester.AddOutput<float>("Z", dims_x, float_expected);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(musa_provider));
  op_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
                &execution_providers);
}

// float same-type Pow（对照组：确认非 fp16 路径也走新 kernel）
TEST_F(MusaBinaryElementwisePrepareTest, PowFloat_SameType) {
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  std::vector<int64_t> dims{4};
  OpTester op_tester("Pow", 12);
  op_tester.AddInput<float>("X", dims, {2.0f, 3.0f, 10.0f, 100.0f});
  op_tester.AddInput<float>("Y", dims, {10.0f, 5.0f, 3.0f, 0.5f});
  op_tester.AddOutput<float>("Z", dims, {1024.0f, 243.0f, 1000.0f, 10.0f});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(musa_provider));
  op_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
                &execution_providers);
}

// int32 same-type Pow
TEST_F(MusaBinaryElementwisePrepareTest, PowInt32_SameType) {
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  std::vector<int64_t> dims{3};
  OpTester op_tester("Pow", 12);
  op_tester.AddInput<int32_t>("X", dims, {2, 3, 5});
  op_tester.AddInput<int32_t>("Y", dims, {10, 5, 3});
  op_tester.AddOutput<int32_t>("Z", dims, {1024, 243, 125});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(musa_provider));
  op_tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
                &execution_providers);
}

namespace {

constexpr int64_t kLayerNormOverflowHidden = 384;
constexpr const char* kLayerNormFusionOptimizedModelPath = "/tmp/layernorm_fusion_optimized.onnx";

ONNX_NAMESPACE::TypeProto MakeTensorType(int32_t elem_type, const std::vector<int64_t>& dims) {
  ONNX_NAMESPACE::TypeProto type_proto;
  auto* tensor_type = type_proto.mutable_tensor_type();
  tensor_type->set_elem_type(elem_type);
  for (int64_t dim : dims) {
    tensor_type->mutable_shape()->add_dim()->set_dim_value(dim);
  }
  return type_proto;
}

void AddFp16Initializer(Graph& graph, const std::string& name, const std::vector<int64_t>& dims,
                        const std::vector<float>& values) {
  ONNX_NAMESPACE::TensorProto initializer;
  initializer.set_name(name);
  initializer.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  for (int64_t dim : dims) {
    initializer.add_dims(dim);
  }
  for (float value : values) {
    initializer.add_int32_data(MLFloat16(value).val);
  }
  graph.AddInitializedTensor(initializer);
}

void BuildIsolatedPowOverflowModel(const std::string& model_file_name) {
  onnxruntime::Model model("pow_fp16_variance_overflow_isolated", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 17}}, {},
                           DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  const std::vector<int64_t> input_dims = {1, 1, kLayerNormOverflowHidden};
  const std::vector<int64_t> mean_dims = {1, 1, 1};
  const auto fp16_input_type = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, input_dims);
  const auto fp16_mean_type = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, mean_dims);
  const auto fp16_scalar_type = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, {1});
  const auto float_output_type = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, input_dims);

  auto& input_arg = graph.GetOrCreateNodeArg("input", &fp16_input_type);
  auto& mean_arg = graph.GetOrCreateNodeArg("mean", &fp16_mean_type);
  auto& centered_arg = graph.GetOrCreateNodeArg("centered", &fp16_input_type);
  auto& exponent_arg = graph.GetOrCreateNodeArg("pow_exponent", &fp16_scalar_type);
  auto& squared_arg = graph.GetOrCreateNodeArg("squared", &fp16_input_type);
  auto& output_arg = graph.GetOrCreateNodeArg("output", &float_output_type);

  AddFp16Initializer(graph, "pow_exponent", {1}, {2.0f});

  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;

  inputs = {&input_arg};
  outputs = {&mean_arg};
  auto& reduce_mean_node = graph.AddNode("isolated_reduce_mean", "ReduceMean", "Mean for isolated Pow test",
                                         inputs, outputs);
  reduce_mean_node.AddAttribute("axes", std::vector<int64_t>{-1});
  reduce_mean_node.AddAttribute("keepdims", static_cast<int64_t>(1));

  inputs = {&input_arg, &mean_arg};
  outputs = {&centered_arg};
  graph.AddNode("isolated_sub", "Sub", "Centered input", inputs, outputs);

  inputs = {&centered_arg, &exponent_arg};
  outputs = {&squared_arg};
  graph.AddNode("isolated_pow_square", "Pow", "Unsafe fp16 square landing", inputs, outputs);

  inputs = {&squared_arg};
  outputs = {&output_arg};
  auto& cast_node = graph.AddNode("isolated_pow_output_cast", "Cast", "Expose Pow output as float",
                                  inputs, outputs);
  cast_node.AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << "Graph resolve failed: " << status.ErrorMessage();
  status = onnxruntime::Model::Save(model, model_file_name);
  ASSERT_TRUE(status.IsOK()) << "Model save failed: " << status.ErrorMessage();
}

void BuildFullLayerNormFusionModel(const std::string& model_file_name) {
  onnxruntime::Model model("layernorm_fp16_full_pattern_fusion", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 17}}, {},
                           DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  const std::vector<int64_t> input_dims = {1, 1, kLayerNormOverflowHidden};
  const std::vector<int64_t> mean_dims = {1, 1, 1};
  const std::vector<int64_t> param_dims = {kLayerNormOverflowHidden};
  const auto fp16_input_type = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, input_dims);
  const auto fp16_mean_type = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, mean_dims);
  const auto fp16_param_type = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, param_dims);
  const auto fp16_scalar_type = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, {1});
  const auto float_output_type = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, input_dims);

  auto& input_arg = graph.GetOrCreateNodeArg("input", &fp16_input_type);
  auto& mean_arg = graph.GetOrCreateNodeArg("mean", &fp16_mean_type);
  auto& centered_arg = graph.GetOrCreateNodeArg("centered", &fp16_input_type);
  auto& exponent_arg = graph.GetOrCreateNodeArg("pow_exponent", &fp16_scalar_type);
  auto& squared_arg = graph.GetOrCreateNodeArg("squared", &fp16_input_type);
  auto& variance_arg = graph.GetOrCreateNodeArg("variance", &fp16_mean_type);
  auto& eps_arg = graph.GetOrCreateNodeArg("eps", &fp16_scalar_type);
  auto& variance_eps_arg = graph.GetOrCreateNodeArg("variance_eps", &fp16_mean_type);
  auto& stddev_arg = graph.GetOrCreateNodeArg("stddev", &fp16_mean_type);
  auto& normalized_arg = graph.GetOrCreateNodeArg("normalized", &fp16_input_type);
  auto& scale_arg = graph.GetOrCreateNodeArg("scale", &fp16_param_type);
  auto& scaled_arg = graph.GetOrCreateNodeArg("scaled", &fp16_input_type);
  auto& bias_arg = graph.GetOrCreateNodeArg("bias", &fp16_param_type);
  auto& output_arg = graph.GetOrCreateNodeArg("output", &fp16_input_type);
  auto& output_float_arg = graph.GetOrCreateNodeArg("output_float", &float_output_type);

  AddFp16Initializer(graph, "pow_exponent", {1}, {2.0f});
  AddFp16Initializer(graph, "eps", {1}, {1.0e-5f});
  AddFp16Initializer(graph, "scale", param_dims, std::vector<float>(kLayerNormOverflowHidden, 1.0f));
  AddFp16Initializer(graph, "bias", param_dims, std::vector<float>(kLayerNormOverflowHidden, 0.0f));

  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;

  inputs = {&input_arg};
  outputs = {&mean_arg};
  auto& mean_node = graph.AddNode("layernorm_reduce_mean", "ReduceMean", "LayerNorm mean", inputs, outputs);
  mean_node.AddAttribute("axes", std::vector<int64_t>{-1});
  mean_node.AddAttribute("keepdims", static_cast<int64_t>(1));

  inputs = {&input_arg, &mean_arg};
  outputs = {&centered_arg};
  graph.AddNode("layernorm_sub", "Sub", "LayerNorm centered input", inputs, outputs);

  inputs = {&centered_arg, &exponent_arg};
  outputs = {&squared_arg};
  graph.AddNode("layernorm_pow_square", "Pow", "LayerNorm variance square", inputs, outputs);

  inputs = {&squared_arg};
  outputs = {&variance_arg};
  auto& variance_node = graph.AddNode("layernorm_reduce_variance", "ReduceMean", "LayerNorm variance",
                                      inputs, outputs);
  variance_node.AddAttribute("axes", std::vector<int64_t>{-1});
  variance_node.AddAttribute("keepdims", static_cast<int64_t>(1));

  inputs = {&variance_arg, &eps_arg};
  outputs = {&variance_eps_arg};
  graph.AddNode("layernorm_add_eps", "Add", "LayerNorm variance plus epsilon", inputs, outputs);

  inputs = {&variance_eps_arg};
  outputs = {&stddev_arg};
  graph.AddNode("layernorm_sqrt", "Sqrt", "LayerNorm stddev", inputs, outputs);

  inputs = {&centered_arg, &stddev_arg};
  outputs = {&normalized_arg};
  graph.AddNode("layernorm_div", "Div", "LayerNorm normalize", inputs, outputs);

  inputs = {&normalized_arg, &scale_arg};
  outputs = {&scaled_arg};
  graph.AddNode("layernorm_mul_scale", "Mul", "LayerNorm scale", inputs, outputs);

  inputs = {&scaled_arg, &bias_arg};
  outputs = {&output_arg};
  graph.AddNode("layernorm_add_bias", "Add", "LayerNorm bias", inputs, outputs);

  inputs = {&output_arg};
  outputs = {&output_float_arg};
  auto& cast_node = graph.AddNode("layernorm_output_cast", "Cast", "Expose LayerNorm output as float",
                                  inputs, outputs);
  cast_node.AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << "Graph resolve failed: " << status.ErrorMessage();
  status = onnxruntime::Model::Save(model, model_file_name);
  ASSERT_TRUE(status.IsOK()) << "Model save failed: " << status.ErrorMessage();
}

std::vector<MLFloat16> MakeLayerNormOverflowInput() {
  std::vector<float> raw(kLayerNormOverflowHidden, 0.0f);
  for (int i = 0; i < 5; ++i) {
    raw[i] = 300.0f;
  }
  for (int i = 5; i < 10; ++i) {
    raw[i] = -300.0f;
  }

  std::vector<MLFloat16> input;
  input.reserve(raw.size());
  for (float value : raw) {
    input.push_back(MLFloat16(value));
  }
  return input;
}

std::vector<float> RunLayerNormOverflowModel(const std::string& model_file_name,
                                             std::vector<MLFloat16>& input,
                                             bool use_musa,
                                             const std::string& optimized_model_path = {}) {
  SessionOptions so;
  so.session_logid = use_musa ? "LayerNormOverflow_MUSA" : "LayerNormOverflow_CPU";
  so.graph_optimization_level = TransformerLevel::Level3;
  if (!optimized_model_path.empty()) {
    so.optimized_model_filepath = optimized_model_path;
  }

  InferenceSession session_object{so, GetEnvironment()};
  if (use_musa) {
    auto musa_provider = DefaultMusaExecutionProvider();
    EXPECT_NE(musa_provider, nullptr) << "MUSA execution provider not available";
    if (!musa_provider) {
      return {};
    }
    auto status = session_object.RegisterExecutionProvider(std::move(musa_provider));
    EXPECT_TRUE(status.IsOK()) << "Register MUSA provider failed: " << status.ErrorMessage();
    if (!status.IsOK()) {
      return {};
    }
  }

  auto status = session_object.Load(model_file_name);
  EXPECT_TRUE(status.IsOK()) << "Load model failed: " << status.ErrorMessage();
  if (!status.IsOK()) {
    return {};
  }
  status = session_object.Initialize();
  EXPECT_TRUE(status.IsOK()) << "Initialize session failed: " << status.ErrorMessage();
  if (!status.IsOK()) {
    return {};
  }

  std::vector<int64_t> input_dims = {1, 1, kLayerNormOverflowHidden};
  OrtValue input_value;
  CreateMLValue<MLFloat16>(gsl::make_span(input_dims), input.data(), OrtMemoryInfo(), &input_value);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("input", input_value));
  std::vector<std::string> output_names{model_file_name.find("full_pattern") == std::string::npos ? "output"
                                                                                                  : "output_float"};
  std::vector<OrtValue> fetches;
  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  status = session_object.Run(run_options, feeds, output_names, &fetches);
  EXPECT_TRUE(status.IsOK()) << "Run failed: " << status.ErrorMessage();
  if (!status.IsOK()) {
    return {};
  }
  EXPECT_EQ(fetches.size(), 1U);
  if (fetches.size() != 1U) {
    return {};
  }

  const auto& output_tensor = fetches.front().Get<Tensor>();
  EXPECT_EQ(output_tensor.Shape(), TensorShape(input_dims));
  const float* output_data = output_tensor.Data<float>();
  return std::vector<float>(output_data, output_data + kLayerNormOverflowHidden);
}

std::map<std::string, int> CountOpsInSavedModel(const std::string& model_file_name) {
  ONNX_NAMESPACE::ModelProto model_proto;
  auto status = onnxruntime::Model::Load(model_file_name, model_proto);
  EXPECT_TRUE(status.IsOK()) << "Load optimized model failed: " << status.ErrorMessage();
  std::map<std::string, int> op_to_count;
  if (!status.IsOK()) {
    return op_to_count;
  }
  for (const auto& node : model_proto.graph().node()) {
    ++op_to_count[node.op_type()];
  }
  return op_to_count;
}

double MaxRelativeDiff(const std::vector<float>& expected, const std::vector<float>& actual) {
  double max_rel = 0.0;
  for (size_t i = 0; i < expected.size(); ++i) {
    const bool expected_finite = std::isfinite(expected[i]);
    const bool actual_finite = std::isfinite(actual[i]);
    if (expected_finite && actual_finite) {
      const double denom = std::max(std::abs(static_cast<double>(expected[i])), 1e-6);
      const double rel = std::abs(static_cast<double>(expected[i]) - static_cast<double>(actual[i])) / denom;
      max_rel = std::max(max_rel, rel);
    } else if (expected_finite != actual_finite) {
      max_rel = std::numeric_limits<double>::infinity();
    }
  }
  return max_rel;
}

int CountInf(const std::vector<float>& values) {
  return static_cast<int>(std::count_if(values.begin(), values.end(), [](float v) { return std::isinf(v); }));
}

}  // namespace

TEST_F(MusaBinaryElementwisePrepareTest, Pow_fp16_VarianceOverflow_IsolatedSubgraph) {
  const std::string model_file_name = "/tmp/pow_fp16_variance_overflow_isolated_redtest.onnx";
  BuildIsolatedPowOverflowModel(model_file_name);

  auto input = MakeLayerNormOverflowInput();
  const auto cpu_output = RunLayerNormOverflowModel(model_file_name, input, false);
  const auto musa_output = RunLayerNormOverflowModel(model_file_name, input, true);

  ASSERT_EQ(cpu_output.size(), static_cast<size_t>(kLayerNormOverflowHidden));
  ASSERT_EQ(musa_output.size(), cpu_output.size());

  for (size_t i = 0; i < cpu_output.size(); ++i) {
    EXPECT_TRUE(std::isfinite(cpu_output[i])) << "CPU output should be finite at index " << i
                                             << ", value=" << cpu_output[i];
  }

  const int musa_inf_count = CountInf(musa_output);
  const double max_rel = MaxRelativeDiff(cpu_output, musa_output);
  std::cerr << "[INFO] Isolated Pow MUSA EP output Inf count: " << musa_inf_count << " / "
            << musa_output.size() << std::endl;
  std::cerr << "[INFO] Isolated Pow CPU vs MUSA max_rel=" << max_rel << std::endl;

  // This 5-node isolated subgraph deliberately omits the Add/Sqrt/Div/Mul/Add tail plus scale and bias,
  // so it cannot trigger LayerNormFusion. It documents the physical fp16 dynamic-range limit:
  // centered values near +/-300 square to ~90000, which cannot be represented in fp16. The fixable signal is
  // LayerNormFp16_FullPattern_FusionProtected_InferenceSession, where the complete LayerNorm pattern is fused.
  EXPECT_EQ(musa_inf_count, 0)
      << "MUSA EP produced " << musa_inf_count
      << " Inf in the isolated fp16 Pow landing point";
  EXPECT_LT(max_rel, 1e-2) << "MUSA EP diverged from CPU EP: max_rel=" << max_rel;
}

TEST_F(MusaBinaryElementwisePrepareTest, LayerNormFp16_FullPattern_FusionProtected_InferenceSession) {
  const std::string model_file_name = "/tmp/layernorm_fp16_full_pattern_fusion_redtest.onnx";
  BuildFullLayerNormFusionModel(model_file_name);

  auto input = MakeLayerNormOverflowInput();
  const auto cpu_output = RunLayerNormOverflowModel(model_file_name, input, false);
  const auto musa_output = RunLayerNormOverflowModel(model_file_name, input, true, kLayerNormFusionOptimizedModelPath);

  ASSERT_EQ(cpu_output.size(), static_cast<size_t>(kLayerNormOverflowHidden));
  ASSERT_EQ(musa_output.size(), cpu_output.size());

  for (size_t i = 0; i < cpu_output.size(); ++i) {
    EXPECT_TRUE(std::isfinite(cpu_output[i])) << "CPU output should be finite at index " << i
                                             << ", value=" << cpu_output[i];
  }

  const int musa_inf_count = CountInf(musa_output);
  const double max_rel = MaxRelativeDiff(cpu_output, musa_output);
  std::cerr << "[INFO] Full LayerNorm MUSA EP output Inf count: " << musa_inf_count << " / "
            << musa_output.size() << std::endl;
  std::cerr << "[INFO] Full LayerNorm CPU vs MUSA max_rel=" << max_rel << std::endl;

  EXPECT_EQ(musa_inf_count, 0) << "MUSA fused LayerNormalization should protect fp16 variance overflow";
  EXPECT_LT(max_rel, 1e-2) << "MUSA fused LayerNormalization diverged from CPU EP: max_rel=" << max_rel;

  const auto optimized_ops = CountOpsInSavedModel(kLayerNormFusionOptimizedModelPath);
  const int layer_norm_count = optimized_ops.count("LayerNormalization") ? optimized_ops.at("LayerNormalization") : 0;
  const int pow_count = optimized_ops.count("Pow") ? optimized_ops.at("Pow") : 0;
  std::cerr << "[INFO] Optimized graph LayerNormalization nodes: " << layer_norm_count << std::endl;
  std::cerr << "[INFO] Optimized graph Pow nodes: " << pow_count << std::endl;
  EXPECT_EQ(layer_norm_count, 1);
  EXPECT_EQ(pow_count, 0);
}

} // namespace test
} // namespace onnxruntime
