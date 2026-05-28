// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/compare_provider_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

#include <vector>

namespace onnxruntime {
namespace test {

namespace {

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

class BatchNormReluTester : public CompareOpTester {
 public:
  BatchNormReluTester() : CompareOpTester("BatchNormRelu", 15) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto bn_tensor = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {1, 2, 2, 1});
    auto& bn_arg = graph.GetOrCreateNodeArg("BatchNormY", &bn_tensor);

    auto& bn_node = graph.AddNode("batchnorm_node", "BatchNormalization", "BatchNormalization node",
                                  {graph_input_defs[0], graph_input_defs[1], graph_input_defs[2],
                                   graph_input_defs[3], graph_input_defs[4]},
                                  {&bn_arg});
    bn_node.AddAttribute("epsilon", 1e-5f);
    graph.AddNode("relu_node", "Relu", "Relu node", {&bn_arg}, {graph_output_defs[0]});
  }
};

}  // namespace

TEST(MusaBatchNormTest, BatchNormalizationFloatNoCpuFallback) {
  CompareOpTester test("BatchNormalization", 15);
  test.AddAttribute("epsilon", 1e-5f);
  test.AddInput<float>("X", {1, 2, 2, 1}, {1.0f, 3.0f, 4.0f, 7.0f});
  test.AddInput<float>("scale", {2}, {1.0f, 1.5f});
  test.AddInput<float>("B", {2}, {0.5f, -1.0f});
  test.AddInput<float>("mean", {2}, {1.0f, 4.0f});
  test.AddInput<float>("var", {2}, {4.0f, 9.0f});
  test.AddOutput<float>("Y", {1, 2, 2, 1}, {0.5f, 1.4999988f, -1.0f, 0.4999992f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaBatchNormTest, BatchNormalizationFloat16NoCpuFallback) {
  CompareOpTester test("BatchNormalization", 15);
  test.AddAttribute("epsilon", 1e-3f);
  test.AddInput<MLFloat16>("X", {1, 2, 2, 1}, ToFloat16({1.0f, 3.0f, 4.0f, 7.0f}));
  test.AddInput<MLFloat16>("scale", {2}, ToFloat16({1.0f, 1.5f}));
  test.AddInput<MLFloat16>("B", {2}, ToFloat16({0.5f, -1.0f}));
  test.AddInput<MLFloat16>("mean", {2}, ToFloat16({1.0f, 4.0f}));
  test.AddInput<MLFloat16>("var", {2}, ToFloat16({4.0f, 9.0f}));
  test.AddOutput<MLFloat16>("Y", {1, 2, 2, 1}, ToFloat16({0.5f, 1.4998751f, -1.0f, 0.4999167f}));
  CompareWithMusaNoFallback(test, true, 2e-3, 2e-3);
}

TEST(MusaBatchNormTest, BatchNormalizationDynamicShapeNoCpuFallback) {
  CompareOpTester test("BatchNormalization", 15);
  const std::vector<std::string> dim_params{"batch", "2", "seq", "1"};
  test.AddAttribute("epsilon", 1e-5f);
  test.AddInput<float>("X", {1, 2, 2, 1}, {1.0f, 3.0f, 4.0f, 7.0f}, false, &dim_params);
  test.AddInput<float>("scale", {2}, {1.0f, 1.5f});
  test.AddInput<float>("B", {2}, {0.5f, -1.0f});
  test.AddInput<float>("mean", {2}, {1.0f, 4.0f});
  test.AddInput<float>("var", {2}, {4.0f, 9.0f});
  test.AddOutput<float>("Y", {1, 2, 2, 1}, {0.5f, 1.4999988f, -1.0f, 0.4999992f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaBatchNormTest, BatchNormalizationReluMultiOpNoCpuFallback) {
  BatchNormReluTester test;
  test.AddInput<float>("X", {1, 2, 2, 1}, {1.0f, 3.0f, 4.0f, 7.0f});
  test.AddInput<float>("scale", {2}, {1.0f, 1.5f});
  test.AddInput<float>("B", {2}, {0.5f, -1.0f});
  test.AddInput<float>("mean", {2}, {1.0f, 4.0f});
  test.AddInput<float>("var", {2}, {4.0f, 9.0f});
  test.AddOutput<float>("Y", {1, 2, 2, 1}, {0.5f, 1.4999988f, 0.0f, 0.4999992f});
  CompareWithMusaNoFallback(test);
}

}  // namespace test
}  // namespace onnxruntime
