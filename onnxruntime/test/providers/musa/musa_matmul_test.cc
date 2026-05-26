// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Test fixture for MatMul operator 
class MusaMatMulTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Test will skip if MUSA provider is not available
  }
};

// Test cases based on debug output analysis

// SUCCESS CASES (3 tests expected to pass)

TEST_F(MusaMatMulTest, TestLeftOneDRightTwoD) {
  // [2] x [2, 3] → [3] - EXPECTED SUCCESS (based on debug output)
  OpTester test("MatMul", 13);
  
  test.AddInput<float>("A", {2}, {0.f, 1.f});
  test.AddInput<float>("B", {2, 3}, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f});
  test.AddOutput<float>("Y", {3}, {3.f, 4.f, 5.f});
  
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

TEST_F(MusaMatMulTest, TestScalarOutput) {
  // [3] x [3] → [] - EXPECTED SUCCESS (based on debug output)
  OpTester test("MatMul", 13);
  
  test.AddInput<float>("A", {3}, {0.f, 1.f, 2.f});
  test.AddInput<float>("B", {3}, {0.f, 1.f, 2.f});
  test.AddOutput<float>("Y", {}, {5.f});
  
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

TEST_F(MusaMatMulTest, TestTwoD) {
  // [3, 4] x [4, 3] → [3, 3] - EXPECTED SUCCESS (based on debug output)
  OpTester test("MatMul", 13);
  
  test.AddInput<float>("A", {3, 4}, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f});
  test.AddInput<float>("B", {4, 3}, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f});
  test.AddOutput<float>("Y", {3, 3}, {42.f, 48.f, 54.f, 114.f, 136.f, 158.f, 186.f, 224.f, 262.f});
  
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

// FAILURE CASES - muDNN MatMul fails on complex broadcast scenarios

TEST_F(MusaMatMulTest, TestPaddingBroadcastABigger) {
  // [3, 1, 1, 2] x [2, 2, 2] → [3, 2, 1, 2] - EXPECTED FAILURE due to muDNN issue
  OpTester test("MatMul", 13);
  
  test.AddInput<float>("A", {3, 1, 1, 2}, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f});
  test.AddInput<float>("B", {2, 2, 2}, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f});
  test.AddOutput<float>("Y", {3, 2, 1, 2}, {2.f, 3.f, 6.f, 7.f, 6.f, 11.f, 26.f, 31.f, 10.f, 19.f, 46.f, 55.f});
  
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  auto musa_provider = DefaultMusaExecutionProvider();
  if (musa_provider) {
    execution_providers.push_back(std::move(musa_provider));
    // This test will fail due to muDNN MatMul broadcast issue
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
             &execution_providers);
  } else {
    GTEST_SKIP() << "MUSA execution provider not available";
  }
}

TEST_F(MusaMatMulTest, TestPaddingBroadcastBBigger) {
  // [2, 3, 2] x [3, 2, 2, 1] → [3, 2, 3, 1] - EXPECTED FAILURE due to muDNN issue
  OpTester test("MatMul", 13);
  
  test.AddInput<float>("A", {2, 3, 2}, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f});
  test.AddInput<float>("B", {3, 2, 2, 1}, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f});
  test.AddOutput<float>("Y", {3, 2, 3, 1}, {1.f, 3.f, 5.f, 33.f, 43.f, 53.f, 5.f, 23.f, 41.f, 85.f, 111.f, 137.f, 9.f, 43.f, 77.f, 137.f, 179.f, 221.f});
  
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  auto musa_provider = DefaultMusaExecutionProvider();
  if (musa_provider) {
    execution_providers.push_back(std::move(musa_provider));
    // This test will fail due to muDNN MatMul broadcast issue
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
             &execution_providers);
  } else {
    GTEST_SKIP() << "MUSA execution provider not available";
  }
}

TEST_F(MusaMatMulTest, TestLeftOneD) {
  // [2] x [3, 2, 1] → [3, 1] - EXPECTED FAILURE due to muDNN issue
  OpTester test("MatMul", 13);
  
  test.AddInput<float>("A", {2}, {0.f, 1.f});
  test.AddInput<float>("B", {3, 2, 1}, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f});
  test.AddOutput<float>("Y", {3, 1}, {1.f, 3.f, 5.f});
  
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  auto musa_provider = DefaultMusaExecutionProvider();
  if (musa_provider) {
    execution_providers.push_back(std::move(musa_provider));
    // This test will fail due to muDNN MatMul broadcast issue  
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
             &execution_providers);
  } else {
    GTEST_SKIP() << "MUSA execution provider not available";
  }
}

TEST_F(MusaMatMulTest, TestTwoDSpecial) {
  // [2, 2, 3] x [3, 4] → [2, 2, 4] - EXPECTED FAILURE due to muDNN issue
  OpTester test("MatMul", 13);
  
  test.AddInput<float>("A", {2, 2, 3}, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f});
  test.AddInput<float>("B", {3, 4}, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f});
  test.AddOutput<float>("Y", {2, 2, 4}, {20.f, 23.f, 26.f, 29.f, 56.f, 68.f, 80.f, 92.f, 92.f, 113.f, 134.f, 155.f, 128.f, 158.f, 188.f, 218.f});
  
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  auto musa_provider = DefaultMusaExecutionProvider();
  if (musa_provider) {
    execution_providers.push_back(std::move(musa_provider));
    // This test will fail due to muDNN MatMul broadcast issue
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
             &execution_providers);
  } else {
    GTEST_SKIP() << "MUSA execution provider not available";
  }
}

// FLOAT16 TEST CASES - Based on MathOpTest.MatMulFloat16

TEST_F(MusaMatMulTest, TestFloat16Simple) {
  // Simple Float16 test: [2, 4] x [4, 3] → [2, 3]
  OpTester test("MatMul", 14);
  
  std::vector<float> A{1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f};
  std::vector<float> B(12, 1.0f);
  std::vector<float> Y{10.0f, 10.0f, 10.0f, -10.0f, -10.0f, -10.0f};

  std::vector<MLFloat16> f_A(8);
  std::vector<MLFloat16> f_B(12);
  std::vector<MLFloat16> f_Y(6);
  ConvertFloatToMLFloat16(A.data(), f_A.data(), 8);
  ConvertFloatToMLFloat16(B.data(), f_B.data(), 12);
  ConvertFloatToMLFloat16(Y.data(), f_Y.data(), 6);

  test.AddInput<MLFloat16>("A", {2, 4}, f_A);
  test.AddInput<MLFloat16>("B", {4, 3}, f_B);
  test.AddOutput<MLFloat16>("Y", {2, 3}, f_Y);
  
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

TEST_F(MusaMatMulTest, TestFloat16PaddingBroadcastABigger) {
  // [3, 1, 1, 2] x [2, 2, 2] → [3, 2, 1, 2] - Float16 version
  OpTester test("MatMul", 14);
  
  std::vector<float> A_f{0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
  std::vector<float> B_f{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
  std::vector<float> Y_f{2.f, 3.f, 6.f, 7.f, 6.f, 11.f, 26.f, 31.f, 10.f, 19.f, 46.f, 55.f};

  std::vector<MLFloat16> A_fp16(6), B_fp16(8), Y_fp16(12);
  ConvertFloatToMLFloat16(A_f.data(), A_fp16.data(), 6);
  ConvertFloatToMLFloat16(B_f.data(), B_fp16.data(), 8);
  ConvertFloatToMLFloat16(Y_f.data(), Y_fp16.data(), 12);
  
  test.AddInput<MLFloat16>("A", {3, 1, 1, 2}, A_fp16);
  test.AddInput<MLFloat16>("B", {2, 2, 2}, B_fp16);
  test.AddOutput<MLFloat16>("Y", {3, 2, 1, 2}, Y_fp16);
  
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

TEST_F(MusaMatMulTest, TestFloat16TwoDSpecial) {
  // [2, 2, 3] x [3, 4] → [2, 2, 4] - Float16 version
  OpTester test("MatMul", 14);
  
  std::vector<float> A_f{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f};
  std::vector<float> B_f{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f};
  std::vector<float> Y_f{20.f, 23.f, 26.f, 29.f, 56.f, 68.f, 80.f, 92.f, 92.f, 113.f, 134.f, 155.f, 128.f, 158.f, 188.f, 218.f};

  std::vector<MLFloat16> A_fp16(12), B_fp16(12), Y_fp16(16);
  ConvertFloatToMLFloat16(A_f.data(), A_fp16.data(), 12);
  ConvertFloatToMLFloat16(B_f.data(), B_fp16.data(), 12);
  ConvertFloatToMLFloat16(Y_f.data(), Y_fp16.data(), 16);
  
  test.AddInput<MLFloat16>("A", {2, 2, 3}, A_fp16);
  test.AddInput<MLFloat16>("B", {3, 4}, B_fp16);
  test.AddOutput<MLFloat16>("Y", {2, 2, 4}, Y_fp16);
  
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

TEST_F(MusaMatMulTest, TestFloat16LeftOneDRightTwoD) {
  // [2] x [2, 3] → [3] - Float16 version
  OpTester test("MatMul", 14);
  
  std::vector<float> A_f{0.f, 1.f};
  std::vector<float> B_f{0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
  std::vector<float> Y_f{3.f, 4.f, 5.f};

  std::vector<MLFloat16> A_fp16(2), B_fp16(6), Y_fp16(3);
  ConvertFloatToMLFloat16(A_f.data(), A_fp16.data(), 2);
  ConvertFloatToMLFloat16(B_f.data(), B_fp16.data(), 6);
  ConvertFloatToMLFloat16(Y_f.data(), Y_fp16.data(), 3);
  
  test.AddInput<MLFloat16>("A", {2}, A_fp16);
  test.AddInput<MLFloat16>("B", {2, 3}, B_fp16);
  test.AddOutput<MLFloat16>("Y", {3}, Y_fp16);
  
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

TEST_F(MusaMatMulTest, TestFloat16ScalarOutput) {
  // [3] x [3] → [] - Float16 version
  OpTester test("MatMul", 14);
  
  std::vector<float> A_f{0.f, 1.f, 2.f};
  std::vector<float> B_f{0.f, 1.f, 2.f};
  std::vector<float> Y_f{5.f};

  std::vector<MLFloat16> A_fp16(3), B_fp16(3), Y_fp16(1);
  ConvertFloatToMLFloat16(A_f.data(), A_fp16.data(), 3);
  ConvertFloatToMLFloat16(B_f.data(), B_fp16.data(), 3);
  ConvertFloatToMLFloat16(Y_f.data(), Y_fp16.data(), 1);
  
  test.AddInput<MLFloat16>("A", {3}, A_fp16);
  test.AddInput<MLFloat16>("B", {3}, B_fp16);
  test.AddOutput<MLFloat16>("Y", {}, Y_fp16);
  
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

TEST_F(MusaMatMulTest, TestFloat16TwoD) {
  // [3, 4] x [4, 3] → [3, 3] - Float16 version  
  OpTester test("MatMul", 14);
  
  std::vector<float> A_f{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f};
  std::vector<float> B_f{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f};
  std::vector<float> Y_f{42.f, 48.f, 54.f, 114.f, 136.f, 158.f, 186.f, 224.f, 262.f};

  std::vector<MLFloat16> A_fp16(12), B_fp16(12), Y_fp16(9);
  ConvertFloatToMLFloat16(A_f.data(), A_fp16.data(), 12);
  ConvertFloatToMLFloat16(B_f.data(), B_fp16.data(), 12);
  ConvertFloatToMLFloat16(Y_f.data(), Y_fp16.data(), 9);
  
  test.AddInput<MLFloat16>("A", {3, 4}, A_fp16);
  test.AddInput<MLFloat16>("B", {4, 3}, B_fp16);
  test.AddOutput<MLFloat16>("Y", {3, 3}, Y_fp16);
  
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

// ============================================================================
// 4D @ 4D Attention Pattern Tests (batch dims matching, no broadcasting)
// These tests verify 4D->3D reshape optimization for MuDNN BatchMatMul
// ============================================================================

TEST_F(MusaMatMulTest, Test4DAttentionPattern) {
  // Larger 4D test: {2,3,2,2} @ {2,3,2,2} -> {2,3,2,2}
  // batch dims: (2,3) == (2,3) -> can use 4D->3D reshape optimization
  OpTester test("MatMul", 13);

  // Use simple sequential values for easier verification
  // A: {2,3,2,2} = 24 elements
  std::vector<float> A = {
    1.f, 0.f, 0.f, 1.f,  // [0,0] = Identity
    1.f, 0.f, 0.f, 1.f,  // [0,1] = Identity
    1.f, 0.f, 0.f, 1.f,  // [0,2] = Identity
    1.f, 0.f, 0.f, 1.f,  // [1,0] = Identity
    1.f, 0.f, 0.f, 1.f,  // [1,1] = Identity
    1.f, 0.f, 0.f, 1.f   // [1,2] = Identity
  };

  // B: {2,3,2,2} = 24 elements
  std::vector<float> B = {
    1.f, 2.f, 3.f, 4.f,  // [0,0]
    5.f, 6.f, 7.f, 8.f,  // [0,1]
    9.f, 10.f, 11.f, 12.f,  // [0,2]
    13.f, 14.f, 15.f, 16.f,  // [1,0]
    17.f, 18.f, 19.f, 20.f,  // [1,1]
    21.f, 22.f, 23.f, 24.f   // [1,2]
  };

  // Y = A @ B, since A is identity, Y should equal B
  // Y: {2,3,2,2} = 24 elements
  std::vector<float> Y = {
    1.f, 2.f, 3.f, 4.f,
    5.f, 6.f, 7.f, 8.f,
    9.f, 10.f, 11.f, 12.f,
    13.f, 14.f, 15.f, 16.f,
    17.f, 18.f, 19.f, 20.f,
    21.f, 22.f, 23.f, 24.f
  };

  test.AddInput<float>("A", {2, 3, 2, 2}, A);
  test.AddInput<float>("B", {2, 3, 2, 2}, B);
  test.AddOutput<float>("Y", {2, 3, 2, 2}, Y);

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

TEST_F(MusaMatMulTest, Test4DAttentionSmall) {
  // Smaller 4D test: {1,2,2,2} @ {1,2,2,2} -> {1,2,2,2}
  // batch dims: (1,2) == (1,2) -> can use optimization
  OpTester test("MatMul", 13);

  std::vector<float> A = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
  std::vector<float> B = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
  // Y[0,0] = [[0*0+1*2, 0*1+1*3], [2*0+3*2, 2*1+3*3]] = [[2,3],[6,11]]
  // Y[0,1] = [[4*4+5*6, 4*5+5*7], [6*4+7*6, 6*5+7*7]] = [[46,55],[66,79]]
  std::vector<float> Y = {2.f, 3.f, 6.f, 11.f, 46.f, 55.f, 66.f, 79.f};

  test.AddInput<float>("A", {1, 2, 2, 2}, A);
  test.AddInput<float>("B", {1, 2, 2, 2}, B);
  test.AddOutput<float>("Y", {1, 2, 2, 2}, Y);

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

TEST_F(MusaMatMulTest, Test4DAttentionPatternFloat16) {
  // Float16 version of Test4DAttentionPattern
  // {2,3,2,2} @ {2,3,2,2} -> {2,3,2,2}
  // batch dims: (2,3) == (2,3) -> can use 4D->3D reshape optimization
  OpTester test("MatMul", 14);

  // A: {2,3,2,2} = 24 elements - Identity matrices
  std::vector<float> A_f = {
    1.f, 0.f, 0.f, 1.f,  // [0,0] = Identity
    1.f, 0.f, 0.f, 1.f,  // [0,1] = Identity
    1.f, 0.f, 0.f, 1.f,  // [0,2] = Identity
    1.f, 0.f, 0.f, 1.f,  // [1,0] = Identity
    1.f, 0.f, 0.f, 1.f,  // [1,1] = Identity
    1.f, 0.f, 0.f, 1.f   // [1,2] = Identity
  };

  // B: {2,3,2,2} = 24 elements
  std::vector<float> B_f = {
    1.f, 2.f, 3.f, 4.f,  // [0,0]
    5.f, 6.f, 7.f, 8.f,  // [0,1]
    9.f, 10.f, 11.f, 12.f,  // [0,2]
    13.f, 14.f, 15.f, 16.f,  // [1,0]
    17.f, 18.f, 19.f, 20.f,  // [1,1]
    21.f, 22.f, 23.f, 24.f   // [1,2]
  };

  // Y = A @ B, since A is identity, Y should equal B
  // Y: {2,3,2,2} = 24 elements
  std::vector<float> Y_f = {
    1.f, 2.f, 3.f, 4.f,
    5.f, 6.f, 7.f, 8.f,
    9.f, 10.f, 11.f, 12.f,
    13.f, 14.f, 15.f, 16.f,
    17.f, 18.f, 19.f, 20.f,
    21.f, 22.f, 23.f, 24.f
  };

  // Convert to MLFloat16
  std::vector<MLFloat16> A_fp16(24), B_fp16(24), Y_fp16(24);
  ConvertFloatToMLFloat16(A_f.data(), A_fp16.data(), 24);
  ConvertFloatToMLFloat16(B_f.data(), B_fp16.data(), 24);
  ConvertFloatToMLFloat16(Y_f.data(), Y_fp16.data(), 24);

  test.AddInput<MLFloat16>("A", {2, 3, 2, 2}, A_fp16);
  test.AddInput<MLFloat16>("B", {2, 3, 2, 2}, B_fp16);
  test.AddOutput<MLFloat16>("Y", {2, 3, 2, 2}, Y_fp16);

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

// Negative test: ensure broadcasting scenarios still work (should fallback to MuBLAS)
TEST_F(MusaMatMulTest, Test4DBroadcastingNotOptimized) {
  // {2,2,2,2} @ {2,1,2,2} -> has broadcasting on dim[1], should NOT use 4D->3D optimization
  // This test ensures the optimization doesn't break broadcasting cases
  OpTester test("MatMul", 13);

  // A: {2,2,2,2} = 16 elements
  std::vector<float> A = {
    0.f, 1.f, 2.f, 3.f,  // [0,0,:,:]
    4.f, 5.f, 6.f, 7.f,  // [0,1,:,:]
    8.f, 9.f, 10.f, 11.f,  // [1,0,:,:]
    12.f, 13.f, 14.f, 15.f   // [1,1,:,:]
  };

  // B: {2,1,2,2} = 8 elements (broadcasting on dim[1])
  std::vector<float> B = {
    0.f, 1.f, 2.f, 3.f,  // [0,0,:,:]
    4.f, 5.f, 6.f, 7.f   // [1,0,:,:]
  };

  // Y shape: {2,2,2,2} = 16 elements
  // For [0,0]: A[0,0]=[[0,1],[2,3]] @ B[0,0]=[[0,1],[2,3]] = [[2,3],[6,11]]
  // For [0,1]: A[0,1]=[[4,5],[6,7]] @ B[0,0]=[[0,1],[2,3]] = [[10,19],[14,27]]
  // For [1,0]: A[1,0]=[[8,9],[10,11]] @ B[1,0]=[[4,5],[6,7]] = [[86,103],[106,127]]
  // For [1,1]: A[1,1]=[[12,13],[14,15]] @ B[1,0]=[[4,5],[6,7]] = [[126,151],[146,175]]
  std::vector<float> Y = {
    2.f, 3.f, 6.f, 11.f,
    10.f, 19.f, 14.f, 27.f,
    86.f, 103.f, 106.f, 127.f,
    126.f, 151.f, 146.f, 175.f
  };

  test.AddInput<float>("A", {2, 2, 2, 2}, A);
  test.AddInput<float>("B", {2, 1, 2, 2}, B);
  test.AddOutput<float>("Y", {2, 2, 2, 2}, Y);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  auto musa_provider = DefaultMusaExecutionProvider();
  if (musa_provider) {
    execution_providers.push_back(std::move(musa_provider));
    // This should still work via MuBLAS fallback
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
             &execution_providers);
  } else {
    GTEST_SKIP() << "MUSA execution provider not available";
  }
}

}  // namespace test
}  // namespace onnxruntime