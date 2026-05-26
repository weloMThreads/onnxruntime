// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Test fixture for Gemm operator
class MusaGemmTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Test will skip if MUSA provider is not available
  }
};

// Basic Gemm test without bias
TEST_F(MusaGemmTest, GemmBasicNoBias) {
  OpTester test("Gemm", 14);

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 0.0f);

  // Matrix A: 2x3
  test.AddInput<float>("A", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  // Matrix B: 3x2
  test.AddInput<float>("B", {3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  // Expected output: 2x2 matrix
  // A*B = [[1*1+2*3+3*5, 1*2+2*4+3*6], [4*1+5*3+6*5, 4*2+5*4+6*6]]
  //     = [[22, 28], [49, 64]]
  test.AddOutput<float>("Y", {2, 2}, {22.0f, 28.0f, 49.0f, 64.0f});

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

// Gemm test with bias
TEST_F(MusaGemmTest, GemmWithBias) {
  OpTester test("Gemm", 14);

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  // Matrix A: 2x2
  test.AddInput<float>("A", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  // Matrix B: 2x2
  test.AddInput<float>("B", {2, 2}, {1.0f, 1.0f, 1.0f, 1.0f});
  // Bias C: 2x2
  test.AddInput<float>("C", {2}, {1.0f, 1.0f});

  // Expected output: alpha*A*B + beta*C
  // A*B = [[3, 3], [7, 7]]
  // Result = [[4, 4], [8, 8]]
  test.AddOutput<float>("Y", {2, 2}, {4.0f, 4.0f, 8.0f, 8.0f});

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

// Gemm test with transpose A
TEST_F(MusaGemmTest, GemmTransposeA) {
  OpTester test("Gemm", 14);

  test.AddAttribute("transA", (int64_t)1);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 0.0f);

  // Matrix A: 2x3 (will be transposed to 3x2)
  test.AddInput<float>("A", {2, 3}, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f});
  // Matrix B: 2x2
  test.AddInput<float>("B", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

  test.AddOutput<float>("Y", {3, 2}, {16.0f, 22.0f, 13.0f, 20.0f, 20.0f, 28.0f});

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

// Gemm test with transpose B
TEST_F(MusaGemmTest, GemmTransposeB) {
  OpTester test("Gemm", 14);

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)1);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 0.0f);

  // Matrix A: 2x2
  test.AddInput<float>("A", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  // Matrix B: 2x2 (will be transposed)
  test.AddInput<float>("B", {2, 2}, {1.0f, 3.0f, 2.0f, 4.0f});

  // After transpose, B becomes [[1, 2], [3, 4]]
  // A*B = [[7, 10], [15, 22]]
  test.AddOutput<float>("Y", {2, 2}, {7.0f, 10.0f, 15.0f, 22.0f});

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

// Gemm test with alpha and beta scaling
TEST_F(MusaGemmTest, GemmAlphaBeta) {
  OpTester test("Gemm", 14);

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 2.0f);
  test.AddAttribute("beta", 0.5f);

  // Matrix A: 2x2
  test.AddInput<float>("A", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  // Matrix B: 2x2
  test.AddInput<float>("B", {2, 2}, {1.0f, 1.0f, 1.0f, 1.0f});
  // Bias C: 2x2
  test.AddInput<float>("C", {2}, {2.0f, 2.0f});

  // Expected output: alpha*A*B + beta*C
  // A*B = [[3, 3], [7, 7]]
  // alpha*A*B = [[6, 6], [14, 14]]
  // beta*C = [[1, 1], [1, 1]]
  // Result = [[7, 7], [15, 15]]
  test.AddOutput<float>("Y", {2, 2}, {7.0f, 7.0f, 15.0f, 15.0f});

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

// Small matrix test
TEST_F(MusaGemmTest, GemmSmallMatrix) {
  OpTester test("Gemm", 14);

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 0.0f);

  // Matrix A: 1x1
  test.AddInput<float>("A", {1, 1}, {5.0f});
  // Matrix B: 1x1
  test.AddInput<float>("B", {1, 1}, {3.0f});

  // Expected output: 1x1 matrix
  test.AddOutput<float>("Y", {1, 1}, {15.0f});

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

// Gemm test with transpose B and bias
TEST_F(MusaGemmTest, GemmTransposeBWithBias) {
  OpTester test("Gemm", 14);

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)1);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  // Matrix A: 2x3
  test.AddInput<float>("A", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  // Matrix B: 2x3 (will be transposed to 3x2)
  test.AddInput<float>("B", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  // Bias C: 2x2
  test.AddInput<float>("C", {2}, {1.0f, 1.0f});

  // After transpose, B becomes [[1, 4], [2, 5], [3, 6]]
  // A*B = [[1*1+2*2+3*3, 1*4+2*5+3*6], [4*1+5*2+6*3, 4*4+5*5+6*6]]
  //     = [[14, 32], [32, 77]]
  // Expected output: alpha*A*B + beta*C = [[14+1, 32+1], [32+1, 77+1]] = [[15, 33], [33, 78]]
  test.AddOutput<float>("Y", {2, 2}, {15.0f, 33.0f, 33.0f, 78.0f});

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
