// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Test fixture for MUSA LayerNormalization
class MusaLayerNormTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Test will skip if MUSA provider is not available
  }

  void RunMusaTest(OpTester& test, OpTester::ExpectResult expected = OpTester::ExpectResult::kExpectSuccess,
                   const std::string& expected_failure_msg = "") {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    auto musa_provider = DefaultMusaExecutionProvider();
    if (musa_provider) {
      execution_providers.push_back(std::move(musa_provider));
      test.Run(expected, expected_failure_msg, {}, nullptr, &execution_providers);
    } else {
      GTEST_SKIP() << "MUSA execution provider not available";
    }
  }
};

// Test basic LayerNormalization with float32
TEST_F(MusaLayerNormTest, LayerNorm_Basic_Float32) {
  OpTester test("LayerNormalization", 17);  // opset 17
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{1, 2, 3};
  test.AddInput<float>("X", dims, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddInput<float>("Scale", {3}, {1.0f, 1.0f, 1.0f});
  test.AddOutput<float>("Y", dims, {-1.2247f, 0.0f, 1.2247f, -1.2247f, 0.0f, 1.2247f});

  RunMusaTest(test);
}

// Test LayerNormalization with Scale and Bias
TEST_F(MusaLayerNormTest, LayerNorm_Scale_Bias_Float32) {
  OpTester test("LayerNormalization", 17);
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{1, 3, 2};
  test.AddInput<float>("X", dims, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddInput<float>("Scale", {2}, {2.0f, 2.0f});
  test.AddInput<float>("Bias", {2}, {0.5f, -0.5f});
  // Expected: norm(x) * scale + bias
  // Mean per row: [1.5, 3.5, 5.5], Std: [0.5, 0.5, 0.5]
  // norm([1,2]) = [(-0.5)/0.5, (0.5)/0.5] = [-1, 1]
  // output = [-1*2+0.5, 1*2-0.5] = [-1.5, 1.5]
  test.AddOutput<float>("Y", dims, {-1.5f, 1.5f, -1.5f, 1.5f, -1.5f, 1.5f});

  RunMusaTest(test);
}

// Test LayerNormalization with Scale only (no Bias)
TEST_F(MusaLayerNormTest, LayerNorm_ScaleOnly_Float32) {
  OpTester test("LayerNormalization", 17);
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{2, 2, 2};
  test.AddInput<float>("X", dims, {-10.264f, 8.6453f, 43.1561f, -0.641239f,
                                   -8.2164f, 0.11412f, 41.3156f, 3.0458f});
  test.AddInput<float>("Scale", {2}, {-0.6953f, 5.1824f});
  // Output computed with standard layernorm
  test.AddOutput<float>("Y", dims, {0.6953f, 5.1824f, -0.6953f, -5.1824f,
                                    0.6953f, 5.1824f, -0.6953f, -5.1824f});

  RunMusaTest(test);
}

// Test BERT-style LayerNormalization (axis=-1)
TEST_F(MusaLayerNormTest, LayerNorm_BERTStyle_Float32) {
  OpTester test("LayerNormalization", 17);
  test.AddAttribute<int64_t>("axis", -1);
  test.AddAttribute<float>("epsilon", 1e-12f);

  std::vector<int64_t> X_dims{2, 4};
  // Simple test data
  test.AddInput<float>("X", X_dims, {1.0f, 2.0f, 3.0f, 4.0f,
                                     5.0f, 6.0f, 7.0f, 8.0f});
  test.AddInput<float>("Scale", {4}, {1.0f, 1.0f, 1.0f, 1.0f});
  test.AddInput<float>("Bias", {4}, {0.0f, 0.0f, 0.0f, 0.0f});

  // Mean of row 1: 2.5, Std: 1.118
  // norm = [-1.342, -0.447, 0.447, 1.342]
  test.AddOutput<float>("Y", X_dims, {-1.3416f, -0.4472f, 0.4472f, 1.3416f,
                                      -1.3416f, -0.4472f, 0.4472f, 1.3416f});

  RunMusaTest(test);
}

// Test LayerNormalization with MLFloat16 input
TEST_F(MusaLayerNormTest, LayerNorm_MLFloat16) {
  OpTester test("LayerNormalization", 17);
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{1, 2, 3};
  test.AddInput<MLFloat16>("X", dims, ToFloat16({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
  test.AddInput<MLFloat16>("Scale", {3}, ToFloat16({1.0f, 1.0f, 1.0f}));
  test.AddOutput<MLFloat16>("Y", dims, ToFloat16({-1.2247f, 0.0f, 1.2247f,
                                                   -1.2247f, 0.0f, 1.2247f}));

  RunMusaTest(test);
}

// Test LayerNormalization with BFloat16 input
TEST_F(MusaLayerNormTest, LayerNorm_BFloat16) {
  OpTester test("LayerNormalization", 17);
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{1, 2, 3};
  test.AddInput<BFloat16>("X", dims, MakeBFloat16({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
  test.AddInput<BFloat16>("Scale", {3}, MakeBFloat16({1.0f, 1.0f, 1.0f}));
  test.AddOutput<BFloat16>("Y", dims, MakeBFloat16({-1.2247f, 0.0f, 1.2247f,
                                                     -1.2247f, 0.0f, 1.2247f}));

  RunMusaTest(test);
}

// Test LayerNormalization with different axis (axis=1 for 3D tensor)
TEST_F(MusaLayerNormTest, LayerNorm_Axis1_3D) {
  OpTester test("LayerNormalization", 17);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddAttribute<float>("epsilon", 1e-05f);

  // Shape: [1, 2, 3], axis=1 means normalize last 2 dims (2*3=6 elements)
  std::vector<int64_t> dims{1, 2, 3};
  test.AddInput<float>("X", dims, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  // Scale/Bias shape: [2, 3]
  test.AddInput<float>("Scale", {2, 3}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
  // Mean=3.5, Std=1.708
  // Expected: (x-mean)/std
  test.AddOutput<float>("Y", dims, {-1.4639f, -0.8783f, -0.2928f,
                                    0.2928f, 0.8783f, 1.4639f});

  RunMusaTest(test);
}

// Test LayerNormalization with axis=0
TEST_F(MusaLayerNormTest, LayerNorm_Axis0_3D) {
  OpTester test("LayerNormalization", 17);
  test.AddAttribute<int64_t>("axis", 0);
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{1, 2, 3};
  test.AddInput<float>("X", dims, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  // Scale shape matches all dims for axis=0
  test.AddInput<float>("Scale", {1, 2, 3}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
  // Mean=3.5, Std=1.708
  test.AddOutput<float>("Y", dims, {-1.4639f, -0.8783f, -0.2928f,
                                    0.2928f, 0.8783f, 1.4639f});

  RunMusaTest(test);
}

// Test LayerNormalization with 4D input (common in CV models)
TEST_F(MusaLayerNormTest, LayerNorm_4D_Float32) {
  OpTester test("LayerNormalization", 17);
  test.AddAttribute<int64_t>("axis", -1);
  test.AddAttribute<float>("epsilon", 1e-05f);

  // Shape: [1, 2, 2, 2], normalize over last dim
  std::vector<int64_t> dims{1, 2, 2, 2};
  test.AddInput<float>("X", dims, {1.0f, 2.0f, 3.0f, 4.0f,
                                   5.0f, 6.0f, 7.0f, 8.0f});
  test.AddInput<float>("Scale", {2}, {1.0f, 1.0f});
  test.AddInput<float>("Bias", {2}, {0.0f, 0.0f});

  // Each pair is normalized: mean=1.5 std=0.5 -> [-1, 1]
  test.AddOutput<float>("Y", dims, {-1.0f, 1.0f, -1.0f, 1.0f,
                                    -1.0f, 1.0f, -1.0f, 1.0f});

  RunMusaTest(test);
}

// Test LayerNormalization with larger batch size
TEST_F(MusaLayerNormTest, LayerNorm_BatchSize_Float32) {
  OpTester test("LayerNormalization", 17);
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{4, 3};
  test.AddInput<float>("X", dims, {1.0f, 2.0f, 3.0f,
                                   4.0f, 5.0f, 6.0f,
                                   7.0f, 8.0f, 9.0f,
                                   10.0f, 11.0f, 12.0f});
  test.AddInput<float>("Scale", {3}, {1.0f, 1.0f, 1.0f});
  test.AddInput<float>("Bias", {3}, {0.0f, 0.0f, 0.0f});

  // Each row normalized: mean(1,2,3)=2, std=0.816
  // norm = [-1.2247, 0, 1.2247]
  test.AddOutput<float>("Y", dims, {-1.2247f, 0.0f, 1.2247f,
                                    -1.2247f, 0.0f, 1.2247f,
                                    -1.2247f, 0.0f, 1.2247f,
                                    -1.2247f, 0.0f, 1.2247f});

  RunMusaTest(test);
}

// Test with non-default epsilon
// Note: Larger epsilon causes slightly different results, so we use SetOutputAbsErr for tolerance
TEST_F(MusaLayerNormTest, LayerNorm_CustomEpsilon) {
  OpTester test("LayerNormalization", 17);
  test.AddAttribute<float>("epsilon", 1e-3f);  // larger epsilon

  std::vector<int64_t> dims{1, 2, 3};
  test.AddInput<float>("X", dims, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddInput<float>("Scale", {3}, {1.0f, 1.0f, 1.0f});
  // Results with larger epsilon are slightly different, use adjusted expected values
  // With epsilon=1e-3, norm factor changes slightly
  test.AddOutput<float>("Y", dims, {-1.2238f, 0.0f, 1.2238f, -1.2238f, 0.0f, 1.2238f});

  RunMusaTest(test);
}

// Test with MLFloat16 Scale and Bias (all same type)
// Note: ONNX standard LayerNormalization requires T and U to match for Scale/Bias
TEST_F(MusaLayerNormTest, LayerNorm_Float16_WithScaleBias) {
  OpTester test("LayerNormalization", 17);
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{2, 2, 2};
  test.AddInput<MLFloat16>("X", dims, ToFloat16({-10.264f, 8.6453f, 43.1561f, -0.641239f,
                                                  -8.2164f, 0.11412f, 41.3156f, 3.0458f}));
  test.AddInput<MLFloat16>("Scale", {2}, ToFloat16({-0.6953f, 5.1824f}));
  // Output matches input type for standard LayerNormalization
  test.AddOutput<MLFloat16>("Y", dims, ToFloat16({0.6953f, 5.1824f, -0.6953f, -5.1824f,
                                                   0.6953f, 5.1824f, -0.6953f, -5.1824f}));

  RunMusaTest(test);
}

}  // namespace test
}  // namespace onnxruntime
