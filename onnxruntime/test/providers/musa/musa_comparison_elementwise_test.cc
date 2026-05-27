// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/onnx_protobuf.h"
#include "test/providers/compare_provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <string>
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

class GreaterMinWhereTester : public CompareOpTester {
 public:
  GreaterMinWhereTester() : CompareOpTester("GreaterMinWhere", 13) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto bool_tensor = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_BOOL, {2, 3});
    auto float_tensor = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2, 3});

    auto& cond_arg = graph.GetOrCreateNodeArg("Cond", &bool_tensor);
    auto& min_arg = graph.GetOrCreateNodeArg("MinOut", &float_tensor);

    graph.AddNode("greater_node", "Greater", "Greater node",
                  {graph_input_defs[0], graph_input_defs[1]}, {&cond_arg});
    graph.AddNode("min_node", "Min", "Min node",
                  {graph_input_defs[2], graph_input_defs[3]}, {&min_arg});
    graph.AddNode("where_node", "Where", "Where node",
                  {&cond_arg, &min_arg, graph_input_defs[3]}, {graph_output_defs[0]});
  }
};

}  // namespace

TEST(MusaComparisonElementwiseTest, GreaterFloatBroadcastDynamicShape) {
  CompareOpTester test("Greater", 13);
  const std::vector<std::string> lhs_dim_params{"batch", "1"};
  const std::vector<std::string> rhs_dim_params{"1", "seq"};
  test.AddInput<float>("A", {2, 1}, {1.0f, 4.0f}, false, &lhs_dim_params);
  test.AddInput<float>("B", {1, 3}, {0.0f, 2.0f, 5.0f}, false, &rhs_dim_params);
  test.AddOutput<bool>("C", {2, 3}, {true, false, false, true, true, false});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, GreaterFloat16) {
  CompareOpTester test("Greater", 13);
  test.AddInput<MLFloat16>("A", {4}, ToFloat16({1.0f, 4.0f, 2.0f, -1.0f}));
  test.AddInput<MLFloat16>("B", {4}, ToFloat16({0.5f, 4.0f, 3.0f, -2.0f}));
  test.AddOutput<bool>("C", {4}, {true, false, false, true});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, GreaterInt32Scalar) {
  CompareOpTester test("Greater", 13);
  test.AddInput<int32_t>("A", {4}, {1, 4, 2, -1});
  test.AddInput<int32_t>("B", {}, {2});
  test.AddOutput<bool>("C", {4}, {false, true, false, false});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, GreaterInt64Empty) {
  CompareOpTester test("Greater", 13);
  const std::vector<int64_t> empty;
  test.AddInput<int64_t>("A", {0}, empty);
  test.AddInput<int64_t>("B", {0}, empty);
  test.AddOutput<bool>("C", {0}, {});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, MinFloatBroadcastDynamicShape) {
  CompareOpTester test("Min", 13);
  const std::vector<std::string> lhs_dim_params{"batch", "1"};
  const std::vector<std::string> rhs_dim_params{"1", "seq"};
  test.AddInput<float>("A", {2, 1}, {1.0f, 4.0f}, false, &lhs_dim_params);
  test.AddInput<float>("B", {1, 3}, {0.0f, 2.0f, 5.0f}, false, &rhs_dim_params);
  test.AddOutput<float>("C", {2, 3}, {0.0f, 1.0f, 1.0f, 0.0f, 2.0f, 4.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, MinFloat16Scalar) {
  CompareOpTester test("Min", 13);
  test.AddInput<MLFloat16>("A", {4}, ToFloat16({1.0f, 4.0f, 2.0f, -1.0f}));
  test.AddInput<MLFloat16>("B", {}, ToFloat16({2.0f}));
  test.AddOutput<MLFloat16>("C", {4}, ToFloat16({1.0f, 2.0f, 2.0f, -1.0f}));
  CompareWithMusaNoFallback(test, true, 1e-3, 1e-3);
}

TEST(MusaComparisonElementwiseTest, MinInt32) {
  CompareOpTester test("Min", 13);
  test.AddInput<int32_t>("A", {4}, {1, 4, 2, -1});
  test.AddInput<int32_t>("B", {4}, {2, 3, 2, -2});
  test.AddOutput<int32_t>("C", {4}, {1, 3, 2, -2});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, MinInt64Empty) {
  CompareOpTester test("Min", 13);
  const std::vector<int64_t> empty;
  test.AddInput<int64_t>("A", {0}, empty);
  test.AddInput<int64_t>("B", {0}, empty);
  test.AddOutput<int64_t>("C", {0}, empty);
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, GreaterMinWhereMultiOpGraph) {
  GreaterMinWhereTester test;
  test.AddInput<float>("A", {2, 3}, {1.0f, 4.0f, 2.0f, 8.0f, 5.0f, 0.0f});
  test.AddInput<float>("B", {2, 3}, {2.0f, 3.0f, 2.0f, 7.0f, 6.0f, -1.0f});
  test.AddInput<float>("X", {2, 3}, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});
  test.AddInput<float>("Y", {2, 3}, {6.0f, 25.0f, 35.0f, 35.0f, 70.0f, 55.0f});
  test.AddOutput<float>("Z", {2, 3}, {6.0f, 20.0f, 35.0f, 35.0f, 70.0f, 55.0f});
  CompareWithMusaNoFallback(test);
}

}  // namespace test
}  // namespace onnxruntime
