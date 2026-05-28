// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/onnx_protobuf.h"
#include "core/optimizer/graph_transformer_level.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/compare_provider_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <random>
#include <vector>

namespace onnxruntime {
namespace test {
namespace {

std::vector<MLFloat16> ToFloat16(std::initializer_list<float> values) {
  std::vector<MLFloat16> result;
  result.reserve(values.size());
  std::transform(values.begin(), values.end(), std::back_inserter(result),
                 [](float value) { return MLFloat16(value); });
  return result;
}

std::vector<float> Log1pExpected(std::initializer_list<float> values) {
  std::vector<float> result;
  result.reserve(values.size());
  std::transform(values.begin(), values.end(), std::back_inserter(result),
                 [](float value) { return std::log1p(value); });
  return result;
}

std::vector<float> Expm1Expected(std::initializer_list<float> values) {
  std::vector<float> result;
  result.reserve(values.size());
  std::transform(values.begin(), values.end(), std::back_inserter(result),
                 [](float value) { return std::expm1(value); });
  return result;
}

std::vector<float> FloorExpected(std::initializer_list<float> values) {
  std::vector<float> result;
  result.reserve(values.size());
  std::transform(values.begin(), values.end(), std::back_inserter(result),
                 [](float value) { return std::floor(value); });
  return result;
}

std::vector<float> CeilExpected(std::initializer_list<float> values) {
  std::vector<float> result;
  result.reserve(values.size());
  std::transform(values.begin(), values.end(), std::back_inserter(result),
                 [](float value) { return std::ceil(value); });
  return result;
}

SessionOptions MakeNoFallbackSessionOptions() {
  SessionOptions so;
  so.graph_optimization_level = TransformerLevel::Default;
  ORT_THROW_IF_ERROR(so.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  return so;
}

void RunMusaNoFallback(OpTester& test) {
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
                               double abs_error = 1e-4,
                               double rel_error = 1e-4) {
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  test.CompareWithCPU(kMusaExecutionProvider, abs_error, rel_error, false, {}, true);
}

template <typename T>
std::vector<T> UniformExpected(int64_t count, float low, float high, float seed) {
  std::default_random_engine generator{static_cast<uint32_t>(seed)};
  std::uniform_real_distribution<float> distribution{low, high};
  std::vector<T> output(static_cast<size_t>(count));
  for (auto& value : output) {
    value = static_cast<T>(distribution(generator));
  }
  return output;
}

template <>
std::vector<double> UniformExpected<double>(int64_t count, float low, float high, float seed) {
  std::default_random_engine generator{static_cast<uint32_t>(seed)};
  std::uniform_real_distribution<double> distribution{static_cast<double>(low), static_cast<double>(high)};
  std::vector<double> output(static_cast<size_t>(count));
  for (auto& value : output) {
    value = distribution(generator);
  }
  return output;
}

template <>
std::vector<MLFloat16> UniformExpected<MLFloat16>(int64_t count, float low, float high, float seed) {
  std::default_random_engine generator{static_cast<uint32_t>(seed)};
  std::uniform_real_distribution<float> distribution{low, high};
  std::vector<MLFloat16> output(static_cast<size_t>(count));
  for (auto& value : output) {
    value = MLFloat16(distribution(generator));
  }
  return output;
}

ONNX_NAMESPACE::TypeProto MakeTensorType(int32_t elem_type, const std::vector<int64_t>& dims) {
  ONNX_NAMESPACE::TypeProto type_proto;
  auto* tensor_type = type_proto.mutable_tensor_type();
  tensor_type->set_elem_type(elem_type);
  for (int64_t dim : dims) {
    tensor_type->mutable_shape()->add_dim()->set_dim_value(dim);
  }
  return type_proto;
}

class RandomUniformLikeAddTester : public OpTester {
 public:
  RandomUniformLikeAddTester() : OpTester("RandomUniformLikeAdd", 13) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto float_type = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2, 3});
    auto& random_arg = graph.GetOrCreateNodeArg("RandomOut", &float_type);

    auto& random_node = graph.AddNode("random_node", "RandomUniformLike", "RandomUniformLike node",
                                      {graph_input_defs[0]}, {&random_arg});
    random_node.AddAttribute("low", -1.0f);
    random_node.AddAttribute("high", 1.0f);
    random_node.AddAttribute("seed", 19.0f);
    random_node.AddAttribute("dtype", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));

    graph.AddNode("add_node", "Add", "Add node", {&random_arg, graph_input_defs[1]}, {graph_output_defs[0]});
  }
};

class FloorCeilExpm1Tester : public CompareOpTester {
 public:
  FloorCeilExpm1Tester() : CompareOpTester("FloorCeilExpm1", 17) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto float_type = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2, 3});
    auto& floor_arg = graph.GetOrCreateNodeArg("FloorOut", &float_type);
    auto& ceil_arg = graph.GetOrCreateNodeArg("CeilOut", &float_type);

    graph.AddNode("floor_node", "Floor", "Floor node", {graph_input_defs[0]}, {&floor_arg});
    graph.AddNode("ceil_node", "Ceil", "Ceil node", {&floor_arg}, {&ceil_arg});
    graph.AddNode("expm1_node", "Expm1", "Expm1 node", {&ceil_arg}, {graph_output_defs[0]});
  }
};

}  // namespace

TEST(MusaRandomLog1pTest, Log1pFloatDynamicShapeCpuParity) {
  CompareOpTester test("Log1p", 17);
  const std::vector<std::string> dim_params{"batch", "seq"};
  test.AddInput<float>("X", {2, 3}, {0.0f, 0.25f, 1.0f, 3.0f, -0.5f, 8.0f}, false, &dim_params);
  test.AddOutput<float>("Y", {2, 3}, Log1pExpected({0.0f, 0.25f, 1.0f, 3.0f, -0.5f, 8.0f}));
  CompareWithMusaNoFallback(test, 1e-5, 1e-5);
}

TEST(MusaRandomLog1pTest, Log1pScalarNoCpuFallback) {
  OpTester test("Log1p", 17);
  test.AddInput<float>("X", {}, {3.0f});
  test.AddOutput<float>("Y", {}, Log1pExpected({3.0f}));
  RunMusaNoFallback(test);
}

TEST(MusaRandomLog1pTest, Log1pFloat16NoCpuFallback) {
  OpTester test("Log1p", 17);
  test.AddInput<MLFloat16>("X", {4}, ToFloat16({0.0f, 0.5f, 1.0f, 3.0f}));
  test.AddOutput<MLFloat16>("Y", {4}, ToFloat16({0.0f, 0.4054651f, 0.6931472f, 1.3862944f}));
  RunMusaNoFallback(test);
}

TEST(MusaRandomLog1pTest, Log1pEmptyCpuParity) {
  CompareOpTester test("Log1p", 17);
  test.AddInput<float>("X", {2, 0}, std::vector<float>{});
  test.AddOutput<float>("Y", {2, 0}, std::vector<float>{});
  CompareWithMusaNoFallback(test);
}

TEST(MusaRandomLog1pTest, Expm1FloatDynamicShapeCpuParity) {
  CompareOpTester test("Expm1", 17);
  const std::vector<std::string> dim_params{"batch", "seq"};
  test.AddInput<float>("X", {2, 3}, {0.0f, 0.25f, 1.0f, -0.5f, 2.0f, -1.0f}, false, &dim_params);
  test.AddOutput<float>("Y", {2, 3}, Expm1Expected({0.0f, 0.25f, 1.0f, -0.5f, 2.0f, -1.0f}));
  CompareWithMusaNoFallback(test, 1e-5, 1e-5);
}

TEST(MusaRandomLog1pTest, Expm1ScalarNoCpuFallback) {
  OpTester test("Expm1", 17);
  test.AddInput<float>("X", {}, {1.0f});
  test.AddOutput<float>("Y", {}, Expm1Expected({1.0f}));
  RunMusaNoFallback(test);
}

TEST(MusaRandomLog1pTest, Expm1Float16NoCpuFallback) {
  OpTester test("Expm1", 17);
  test.AddInput<MLFloat16>("X", {4}, ToFloat16({0.0f, 0.5f, 1.0f, -0.5f}));
  test.AddOutput<MLFloat16>("Y", {4}, ToFloat16({0.0f, 0.6487213f, 1.7182819f, -0.3934693f}));
  RunMusaNoFallback(test);
}

TEST(MusaRandomLog1pTest, Expm1EmptyCpuParity) {
  CompareOpTester test("Expm1", 17);
  test.AddInput<float>("X", {2, 0}, std::vector<float>{});
  test.AddOutput<float>("Y", {2, 0}, std::vector<float>{});
  CompareWithMusaNoFallback(test);
}

TEST(MusaRandomLog1pTest, FloorFloatDynamicShapeCpuParity) {
  CompareOpTester test("Floor", 13);
  const std::vector<std::string> dim_params{"batch", "seq"};
  test.AddInput<float>("X", {2, 3}, {-1.2f, -0.0f, 0.2f, 1.0f, 1.7f, 2.9f}, false, &dim_params);
  test.AddOutput<float>("Y", {2, 3}, FloorExpected({-1.2f, -0.0f, 0.2f, 1.0f, 1.7f, 2.9f}));
  CompareWithMusaNoFallback(test);
}

TEST(MusaRandomLog1pTest, FloorScalarNoCpuFallback) {
  OpTester test("Floor", 13);
  test.AddInput<double>("X", {}, {-1.2});
  test.AddOutput<double>("Y", {}, {-2.0});
  RunMusaNoFallback(test);
}

TEST(MusaRandomLog1pTest, FloorFloat16NoCpuFallback) {
  OpTester test("Floor", 13);
  test.AddInput<MLFloat16>("X", {4}, ToFloat16({-1.2f, 0.2f, 1.0f, 1.7f}));
  test.AddOutput<MLFloat16>("Y", {4}, ToFloat16({-2.0f, 0.0f, 1.0f, 1.0f}));
  RunMusaNoFallback(test);
}

TEST(MusaRandomLog1pTest, FloorEmptyCpuParity) {
  CompareOpTester test("Floor", 13);
  test.AddInput<float>("X", {0, 3}, std::vector<float>{});
  test.AddOutput<float>("Y", {0, 3}, std::vector<float>{});
  CompareWithMusaNoFallback(test);
}

TEST(MusaRandomLog1pTest, CeilFloatDynamicShapeCpuParity) {
  CompareOpTester test("Ceil", 13);
  const std::vector<std::string> dim_params{"batch", "seq"};
  test.AddInput<float>("X", {2, 3}, {-1.2f, -0.0f, 0.2f, 1.0f, 1.7f, 2.9f}, false, &dim_params);
  test.AddOutput<float>("Y", {2, 3}, CeilExpected({-1.2f, -0.0f, 0.2f, 1.0f, 1.7f, 2.9f}));
  CompareWithMusaNoFallback(test);
}

TEST(MusaRandomLog1pTest, CeilScalarNoCpuFallback) {
  OpTester test("Ceil", 13);
  test.AddInput<double>("X", {}, {-1.2});
  test.AddOutput<double>("Y", {}, {-1.0});
  RunMusaNoFallback(test);
}

TEST(MusaRandomLog1pTest, CeilFloat16NoCpuFallback) {
  OpTester test("Ceil", 13);
  test.AddInput<MLFloat16>("X", {4}, ToFloat16({-1.2f, 0.2f, 1.0f, 1.7f}));
  test.AddOutput<MLFloat16>("Y", {4}, ToFloat16({-1.0f, 1.0f, 1.0f, 2.0f}));
  RunMusaNoFallback(test);
}

TEST(MusaRandomLog1pTest, CeilEmptyCpuParity) {
  CompareOpTester test("Ceil", 13);
  test.AddInput<float>("X", {0, 3}, std::vector<float>{});
  test.AddOutput<float>("Y", {0, 3}, std::vector<float>{});
  CompareWithMusaNoFallback(test);
}

TEST(MusaRandomLog1pTest, FloorCeilExpm1MultiOpCpuParity) {
  FloorCeilExpm1Tester test;
  test.AddInput<float>("X", {2, 3}, {-1.2f, -0.2f, 0.0f, 0.2f, 1.7f, 2.1f});
  test.AddOutput<float>("Y", {2, 3}, Expm1Expected({-2.0f, -1.0f, 0.0f, 0.0f, 1.0f, 2.0f}));
  CompareWithMusaNoFallback(test, 1e-5, 1e-5);
}

TEST(MusaRandomLog1pTest, RandomUniformLikeFloatCpuParity) {
  CompareOpTester test("RandomUniformLike", 1);
  const std::vector<int64_t> dims{2, 3};
  const std::vector<std::string> dim_params{"batch", "seq"};
  constexpr float low = -2.0f;
  constexpr float high = 3.0f;
  constexpr float seed = 7.0f;
  test.AddAttribute("low", low);
  test.AddAttribute("high", high);
  test.AddAttribute("seed", seed);
  test.AddAttribute<int64_t>("dtype", ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  test.AddInput<float>("X", dims, std::vector<float>(6, 0.0f), false, &dim_params);
  test.AddOutput<float>("Y", dims, UniformExpected<float>(6, low, high, seed));
  CompareWithMusaNoFallback(test, 1e-6, 1e-6);
}

TEST(MusaRandomLog1pTest, RandomUniformLikeDoubleInferDTypeNoCpuFallback) {
  OpTester test("RandomUniformLike", 1);
  const std::vector<int64_t> dims{2, 2};
  constexpr float low = 0.0f;
  constexpr float high = 10.0f;
  constexpr float seed = 123.0f;
  test.AddAttribute("low", low);
  test.AddAttribute("high", high);
  test.AddAttribute("seed", seed);
  test.AddInput<double>("X", dims, std::vector<double>(4, 0.0));
  test.AddOutput<double>("Y", dims, UniformExpected<double>(4, low, high, seed));
  RunMusaNoFallback(test);
}

TEST(MusaRandomLog1pTest, RandomUniformLikeFloat16NoCpuFallback) {
  OpTester test("RandomUniformLike", 1);
  const std::vector<int64_t> dims{4};
  constexpr float low = -1.0f;
  constexpr float high = 1.0f;
  constexpr float seed = 11.0f;
  test.AddAttribute("low", low);
  test.AddAttribute("high", high);
  test.AddAttribute("seed", seed);
  test.AddAttribute<int64_t>("dtype", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  test.AddInput<MLFloat16>("X", dims, ToFloat16({0.0f, 0.0f, 0.0f, 0.0f}));
  test.AddOutput<MLFloat16>("Y", dims, UniformExpected<MLFloat16>(4, low, high, seed));
  RunMusaNoFallback(test);
}

TEST(MusaRandomLog1pTest, RandomUniformLikeEmptyCpuParity) {
  CompareOpTester test("RandomUniformLike", 1);
  constexpr float low = 0.0f;
  constexpr float high = 1.0f;
  constexpr float seed = 5.0f;
  test.AddAttribute("low", low);
  test.AddAttribute("high", high);
  test.AddAttribute("seed", seed);
  test.AddAttribute<int64_t>("dtype", ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  test.AddInput<float>("X", {0, 3}, std::vector<float>{});
  test.AddOutput<float>("Y", {0, 3}, std::vector<float>{});
  CompareWithMusaNoFallback(test);
}

TEST(MusaRandomLog1pTest, RandomUniformLikeAddMultiOpNoCpuFallback) {
  RandomUniformLikeAddTester test;
  constexpr float low = -1.0f;
  constexpr float high = 1.0f;
  constexpr float seed = 19.0f;
  auto expected = UniformExpected<float>(6, low, high, seed);
  const std::vector<float> bias{1.0f, 1.0f, 1.0f, 0.5f, 0.5f, 0.5f};
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] += bias[i];
  }

  test.AddInput<float>("X", {2, 3}, std::vector<float>(6, 0.0f));
  test.AddInput<float>("Bias", {2, 3}, bias);
  test.AddOutput<float>("Y", {2, 3}, expected);
  RunMusaNoFallback(test);
}

}  // namespace test
}  // namespace onnxruntime
