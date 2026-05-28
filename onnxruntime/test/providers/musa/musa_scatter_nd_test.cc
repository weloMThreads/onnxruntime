// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/onnx_protobuf.h"
#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/compare_provider_test_utils.h"
#include "test/util/include/default_providers.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <initializer_list>
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
                                         const std::vector<int64_t>& dims) {
  ONNX_NAMESPACE::TypeProto type_proto;
  auto* tensor_type = type_proto.mutable_tensor_type();
  tensor_type->set_elem_type(elem_type);
  for (int64_t dim : dims) {
    tensor_type->mutable_shape()->add_dim()->set_dim_value(dim);
  }
  return type_proto;
}

class ScatterNDAddTester : public CompareOpTester {
 public:
  ScatterNDAddTester() : CompareOpTester("ScatterNDAdd", 18) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto float_tensor = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2, 3});
    auto& scatter_arg = graph.GetOrCreateNodeArg("ScatterOut", &float_tensor);

    graph.AddNode("scatter_nd_node", "ScatterND", "ScatterND node",
                  {graph_input_defs[0], graph_input_defs[1], graph_input_defs[2]}, {&scatter_arg});
    graph.AddNode("add_node", "Add", "Add node",
                  {&scatter_arg, graph_input_defs[3]}, {graph_output_defs[0]});
  }
};

}  // namespace

TEST(MusaScatterNDTest, FloatDynamicShapeRows) {
  CompareOpTester test("ScatterND", 18);
  const std::vector<std::string> data_dim_params{"batch", "seq"};
  const std::vector<std::string> updates_dim_params{"2", "seq"};
  test.AddInput<float>("data", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, false, &data_dim_params);
  test.AddInput<int64_t>("indices", {2, 1}, {1LL, 0LL});
  test.AddInput<float>("updates", {2, 3}, {10.0f, 11.0f, 12.0f, 20.0f, 21.0f, 22.0f}, false, &updates_dim_params);
  test.AddOutput<float>("output", {2, 3}, {20.0f, 21.0f, 22.0f, 10.0f, 11.0f, 12.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaScatterNDTest, FloatNegativeIndices) {
  CompareOpTester test("ScatterND", 13);
  test.AddInput<float>("data", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddInput<int64_t>("indices", {2, 1}, {-1LL, 0LL});
  test.AddInput<float>("updates", {2, 3}, {10.0f, 11.0f, 12.0f, 20.0f, 21.0f, 22.0f});
  test.AddOutput<float>("output", {2, 3}, {20.0f, 21.0f, 22.0f, 10.0f, 11.0f, 12.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaScatterNDTest, Float16ScalarUpdates) {
  CompareOpTester test("ScatterND", 18);
  test.AddInput<MLFloat16>("data", {2, 2}, ToFloat16({1.0f, 2.0f, 3.0f, 4.0f}));
  test.AddInput<int64_t>("indices", {2, 2}, {0LL, 1LL, 1LL, 0LL});
  test.AddInput<MLFloat16>("updates", {2}, ToFloat16({7.0f, 8.0f}));
  test.AddOutput<MLFloat16>("output", {2, 2}, ToFloat16({1.0f, 7.0f, 8.0f, 4.0f}));
  CompareWithMusaNoFallback(test, true, 1e-3, 1e-3);
}

TEST(MusaScatterNDTest, Int32ScalarUpdates) {
  CompareOpTester test("ScatterND", 18);
  test.AddInput<int32_t>("data", {4}, {1, 2, 3, 4});
  test.AddInput<int64_t>("indices", {2, 1}, {1LL, 3LL});
  test.AddInput<int32_t>("updates", {2}, {20, 40});
  test.AddOutput<int32_t>("output", {4}, {1, 20, 3, 40});
  CompareWithMusaNoFallback(test);
}

TEST(MusaScatterNDTest, Int64ScalarUpdates) {
  CompareOpTester test("ScatterND", 18);
  test.AddInput<int64_t>("data", {4}, {1LL, 2LL, 3LL, 4LL});
  test.AddInput<int64_t>("indices", {2, 1}, {0LL, 2LL});
  test.AddInput<int64_t>("updates", {2}, {10LL, 30LL});
  test.AddOutput<int64_t>("output", {4}, {10LL, 2LL, 30LL, 4LL});
  CompareWithMusaNoFallback(test);
}

TEST(MusaScatterNDTest, BoolScalarUpdates) {
  CompareOpTester test("ScatterND", 18);
  test.AddInput<bool>("data", {4}, {true, true, false, false});
  test.AddInput<int64_t>("indices", {2, 1}, {1LL, 2LL});
  test.AddInput<bool>("updates", {2}, {false, true});
  test.AddOutput<bool>("output", {4}, {true, false, true, false});
  CompareWithMusaNoFallback(test);
}

TEST(MusaScatterNDTest, EmptyIndicesFloat) {
  CompareOpTester test("ScatterND", 18);
  test.AddInput<float>("data", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddInput<int64_t>("indices", {0, 1}, std::vector<int64_t>{});
  test.AddInput<float>("updates", {0, 3}, std::vector<float>{});
  test.AddOutput<float>("output", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaScatterNDTest, ScatterNDAddMultiOpNoCpuFallback) {
  ScatterNDAddTester test;
  test.AddInput<float>("data", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddInput<int64_t>("indices", {1, 1}, {1LL});
  test.AddInput<float>("updates", {1, 3}, {10.0f, 11.0f, 12.0f});
  test.AddInput<float>("bias", {2, 3}, {0.5f, 0.5f, 0.5f, 1.0f, 1.0f, 1.0f});
  test.AddOutput<float>("output", {2, 3}, {1.5f, 2.5f, 3.5f, 11.0f, 12.0f, 13.0f});
  CompareWithMusaNoFallback(test);
}

}  // namespace test
}  // namespace onnxruntime
