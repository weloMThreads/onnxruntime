// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/onnx_protobuf.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/compare_provider_test_utils.h"
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

void CompareWithMusaNoFallback(CompareOpTester& test,
                               bool need_cpu_cast = false,
                               double abs_error = 1e-4,
                               double rel_error = 1e-4) {
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  test.CompareWithCPU(kMusaExecutionProvider, abs_error, rel_error,
                      need_cpu_cast, {}, true);
}

ONNX_NAMESPACE::TypeProto MakeTensorType(int32_t elem_type,
                                         std::initializer_list<int64_t> dims) {
  ONNX_NAMESPACE::TypeProto type_proto;
  auto* tensor_type = type_proto.mutable_tensor_type();
  tensor_type->set_elem_type(elem_type);
  for (int64_t dim : dims) {
    tensor_type->mutable_shape()->add_dim()->set_dim_value(dim);
  }
  return type_proto;
}

class IsNaNWhereSignReciprocalTester : public CompareOpTester {
 public:
  IsNaNWhereSignReciprocalTester() : CompareOpTester("IsNaNWhereSignReciprocal", 13) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto bool_tensor = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_BOOL, {2, 2});
    auto float_tensor = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2, 2});
    auto& isnan_arg = graph.GetOrCreateNodeArg("IsNaNOut", &bool_tensor);
    auto& clean_arg = graph.GetOrCreateNodeArg("Clean", &float_tensor);
    auto& sign_arg = graph.GetOrCreateNodeArg("SignOut", &float_tensor);

    graph.AddNode("isnan_node", "IsNaN", "Detect NaNs", {graph_input_defs[0]}, {&isnan_arg});
    graph.AddNode("where_node", "Where", "Replace NaNs", {&isnan_arg, graph_input_defs[1], graph_input_defs[0]}, {&clean_arg});
    graph.AddNode("sign_node", "Sign", "Sign after NaN cleanup", {&clean_arg}, {&sign_arg});
    graph.AddNode("reciprocal_node", "Reciprocal", "Reciprocal of sign", {&sign_arg}, {graph_output_defs[0]});
  }
};

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

TEST(MusaUnaryElementwiseExtraTest, ReciprocalFloatDynamicShape) {
  CompareOpTester test("Reciprocal", 13);
  const std::vector<std::string> dim_params{"batch", "seq"};
  test.AddInput<float>("X", {2, 3}, {1.0f, 2.0f, -4.0f, 0.25f, -0.5f, 8.0f}, false, &dim_params);
  test.AddOutput<float>("Y", {2, 3}, {1.0f, 0.5f, -0.25f, 4.0f, -2.0f, 0.125f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaUnaryElementwiseExtraTest, ReciprocalFloat16Scalar) {
  CompareOpTester test("Reciprocal", 13);
  test.AddInput<MLFloat16>("X", {}, ToFloat16({4.0f}));
  test.AddOutput<MLFloat16>("Y", {}, ToFloat16({0.25f}));
  CompareWithMusaNoFallback(test, true, 1e-3, 1e-3);
}

TEST(MusaUnaryElementwiseExtraTest, ReciprocalDoubleEmpty) {
  CompareOpTester test("Reciprocal", 13);
  test.AddInput<double>("X", {0}, {});
  test.AddOutput<double>("Y", {0}, {});
  CompareWithMusaNoFallback(test);
}

TEST(MusaUnaryElementwiseExtraTest, SignFloatDynamicShape) {
  CompareOpTester test("Sign", 13);
  const std::vector<std::string> dim_params{"batch", "seq"};
  test.AddInput<float>("input", {2, 3}, {-3.0f, -0.0f, 0.0f, 2.5f, -7.0f, 9.0f}, false, &dim_params);
  test.AddOutput<float>("output", {2, 3}, {-1.0f, 0.0f, 0.0f, 1.0f, -1.0f, 1.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaUnaryElementwiseExtraTest, SignFloat16Scalar) {
  CompareOpTester test("Sign", 13);
  test.AddInput<MLFloat16>("input", {}, ToFloat16({-2.0f}));
  test.AddOutput<MLFloat16>("output", {}, ToFloat16({-1.0f}));
  CompareWithMusaNoFallback(test, true, 1e-3, 1e-3);
}

TEST(MusaUnaryElementwiseExtraTest, SignInt32) {
  CompareOpTester test("Sign", 13);
  test.AddInput<int32_t>("input", {5}, {-4, -1, 0, 2, 9});
  test.AddOutput<int32_t>("output", {5}, {-1, -1, 0, 1, 1});
  CompareWithMusaNoFallback(test);
}

TEST(MusaUnaryElementwiseExtraTest, SignInt64Empty) {
  CompareOpTester test("Sign", 13);
  test.AddInput<int64_t>("input", {0}, {});
  test.AddOutput<int64_t>("output", {0}, {});
  CompareWithMusaNoFallback(test);
}

TEST(MusaUnaryElementwiseExtraTest, IsNaNFloatDynamicShape) {
  CompareOpTester test("IsNaN", 13);
  const std::vector<std::string> dim_params{"batch", "seq"};
  const float nan = std::numeric_limits<float>::quiet_NaN();
  test.AddInput<float>("X", {2, 3}, {1.0f, nan, -2.0f, nan, 0.0f, 5.0f}, false, &dim_params);
  test.AddOutput<bool>("Y", {2, 3}, {false, true, false, true, false, false});
  CompareWithMusaNoFallback(test);
}

TEST(MusaUnaryElementwiseExtraTest, IsNaNFloat16Scalar) {
  CompareOpTester test("IsNaN", 13);
  test.AddInput<MLFloat16>("X", {}, {MLFloat16::NaN});
  test.AddOutput<bool>("Y", {}, {true});
  CompareWithMusaNoFallback(test);
}

TEST(MusaUnaryElementwiseExtraTest, IsNaNDoubleEmpty) {
  CompareOpTester test("IsNaN", 13);
  test.AddInput<double>("X", {0}, {});
  test.AddOutput<bool>("Y", {0}, {});
  CompareWithMusaNoFallback(test);
}

TEST(MusaUnaryElementwiseExtraTest, IsNaNWhereSignReciprocalNoCpuFallback) {
  IsNaNWhereSignReciprocalTester test;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  test.AddInput<float>("X", {2, 2}, {nan, -4.0f, 0.25f, 8.0f});
  test.AddInput<float>("Replacement", {2, 2}, {2.0f, 2.0f, 2.0f, 2.0f});
  test.AddOutput<float>("Y", {2, 2}, {1.0f, -1.0f, 1.0f, 1.0f});
  CompareWithMusaNoFallback(test);
}

}  // namespace test
}  // namespace onnxruntime
