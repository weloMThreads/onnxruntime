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

class GatherNDAddTester : public CompareOpTester {
 public:
  GatherNDAddTester() : CompareOpTester("GatherNDAdd", 13) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto gathered_type = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2, 3});
    auto& gathered_arg = graph.GetOrCreateNodeArg("Gathered", &gathered_type);

    graph.AddNode("gather_nd_node", "GatherND", "GatherND node",
                  {graph_input_defs[0], graph_input_defs[1]}, {&gathered_arg});
    graph.AddNode("add_node", "Add", "Add node",
                  {&gathered_arg, graph_input_defs[2]}, {graph_output_defs[0]});
  }
};

}  // namespace

TEST(MusaGatherNDTest, FloatSliceDynamicShapeDefaultBatchDims) {
  CompareOpTester test("GatherND", 13);
  const std::vector<std::string> data_dim_params{"batch", "seq", "4"};
  const std::vector<std::string> indices_dim_params{"seq", "2", "2"};
  test.AddInput<float>("data", {2, 3, 4}, ValueRange(24, 1.0f), false, &data_dim_params);
  test.AddInput<int64_t>("indices", {3, 2, 2},
                         {0LL, 1LL, 0LL, 2LL,
                          1LL, 0LL, 0LL, 0LL,
                          1LL, 1LL, 1LL, 2LL},
                         false, &indices_dim_params);
  test.AddOutput<float>("output", {3, 2, 4},
                        {5.0f, 6.0f, 7.0f, 8.0f,
                         9.0f, 10.0f, 11.0f, 12.0f,
                         13.0f, 14.0f, 15.0f, 16.0f,
                         1.0f, 2.0f, 3.0f, 4.0f,
                         17.0f, 18.0f, 19.0f, 20.0f,
                         21.0f, 22.0f, 23.0f, 24.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaGatherNDTest, FloatNegativeIndicesBatchDimsOne) {
  CompareOpTester test("GatherND", 12);
  test.AddAttribute<int64_t>("batch_dims", 1);
  test.AddInput<float>("data", {2, 3, 4}, ValueRange(24, 1.0f));
  test.AddInput<int64_t>("indices", {2, 2, 2},
                         {0LL, -3LL, -1LL, 2LL,
                          -1LL, 0LL, 0LL, -2LL});
  test.AddOutput<float>("output", {2, 2}, {2.0f, 11.0f, 21.0f, 15.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaGatherNDTest, ScalarOutputInt64) {
  CompareOpTester test("GatherND", 13);
  test.AddInput<int64_t>("data", {2, 2}, {0LL, 1LL, 2LL, 3LL});
  test.AddInput<int64_t>("indices", {2}, {1LL, 0LL});
  test.AddOutput<int64_t>("output", {}, {2LL});
  CompareWithMusaNoFallback(test);
}

TEST(MusaGatherNDTest, EmptyIndicesFloat) {
  CompareOpTester test("GatherND", 13);
  test.AddInput<float>("data", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddInput<int64_t>("indices", {0, 1}, std::vector<int64_t>{});
  test.AddOutput<float>("output", {0, 3}, std::vector<float>{});
  CompareWithMusaNoFallback(test);
}

TEST(MusaGatherNDTest, Float16Slice) {
  CompareOpTester test("GatherND", 13);
  test.AddInput<MLFloat16>("data", {2, 2}, ToFloat16({0.0f, 0.5f, 1.0f, 1.5f}));
  test.AddInput<int64_t>("indices", {2, 1}, {1LL, 0LL});
  test.AddOutput<MLFloat16>("output", {2, 2}, ToFloat16({1.0f, 1.5f, 0.0f, 0.5f}));
  CompareWithMusaNoFallback(test, true, 1e-3, 1e-3);
}

TEST(MusaGatherNDTest, Int32Values) {
  CompareOpTester test("GatherND", 13);
  test.AddInput<int32_t>("data", {2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7});
  test.AddInput<int64_t>("indices", {2, 3}, {0LL, 0LL, 1LL, 1LL, 0LL, 1LL});
  test.AddOutput<int32_t>("output", {2}, {1, 5});
  CompareWithMusaNoFallback(test);
}

TEST(MusaGatherNDTest, BoolValues) {
  CompareOpTester test("GatherND", 13);
  test.AddInput<bool>("data", {2, 2}, {true, false, false, true});
  test.AddInput<int64_t>("indices", {2, 1, 2}, {0LL, 0LL, 0LL, 1LL});
  test.AddOutput<bool>("output", {2, 1}, {true, false});
  CompareWithMusaNoFallback(test);
}

TEST(MusaGatherNDTest, GatherNDAddMultiOpNoCpuFallback) {
  GatherNDAddTester test;
  test.AddInput<float>("data", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddInput<int64_t>("indices", {2, 1}, {1LL, 0LL});
  test.AddInput<float>("bias", {2, 3}, {10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f});
  test.AddOutput<float>("output", {2, 3}, {14.0f, 15.0f, 16.0f, 11.0f, 12.0f, 13.0f});
  CompareWithMusaNoFallback(test);
}

}  // namespace test
}  // namespace onnxruntime
