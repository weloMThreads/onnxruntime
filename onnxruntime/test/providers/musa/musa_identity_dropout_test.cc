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
#include <functional>
#include <initializer_list>
#include <memory>
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

  test.CompareWithCPU(kMusaExecutionProvider, abs_error, rel_error,
                      false, {}, true);
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

class IdentityAddTester : public OpTester {
 public:
  IdentityAddTester() : OpTester("IdentityAdd", 13) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto float_type = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2, 3});
    auto& identity_arg = graph.GetOrCreateNodeArg("IdentityOut", &float_type);

    graph.AddNode("identity_node", "Identity", "Identity node", {graph_input_defs[0]}, {&identity_arg});
    graph.AddNode("add_node", "Add", "Add node", {&identity_arg, graph_input_defs[1]}, {graph_output_defs[0]});
  }
};

}  // namespace

TEST(MusaIdentityDropoutTest, IdentityFloatDynamicShapeCpuParity) {
  CompareOpTester test("Identity", 13);
  const std::vector<std::string> dim_params{"batch", "seq"};
  test.AddInput<float>("X", {2, 3}, {1.0f, 0.0f, 2.0f, -1.0f, 3.0f, 4.0f}, false, &dim_params);
  test.AddOutput<float>("Y", {2, 3}, {1.0f, 0.0f, 2.0f, -1.0f, 3.0f, 4.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaIdentityDropoutTest, IdentityInt32ScalarNoCpuFallback) {
  OpTester test("Identity", 13);
  test.AddInput<int32_t>("X", {}, {7});
  test.AddOutput<int32_t>("Y", {}, {7});
  RunMusaNoFallback(test);
}

TEST(MusaIdentityDropoutTest, IdentityBoolNoCpuFallback) {
  OpTester test("Identity", 13);
  test.AddInput<bool>("X", {4}, {false, true, true, false});
  test.AddOutput<bool>("Y", {4}, {false, true, true, false});
  RunMusaNoFallback(test);
}

TEST(MusaIdentityDropoutTest, IdentityInt64EmptyCpuParity) {
  CompareOpTester test("Identity", 13);
  test.AddInput<int64_t>("X", {2, 0}, std::vector<int64_t>{});
  test.AddOutput<int64_t>("Y", {2, 0}, std::vector<int64_t>{});
  CompareWithMusaNoFallback(test);
}

TEST(MusaIdentityDropoutTest, IdentityFloat16NoCpuFallback) {
  OpTester test("Identity", 13);
  test.AddInput<MLFloat16>("X", {4}, ToFloat16({2.0f, -1.0f, 0.5f, 3.0f}));
  test.AddOutput<MLFloat16>("Y", {4}, ToFloat16({2.0f, -1.0f, 0.5f, 3.0f}));
  RunMusaNoFallback(test);
}

TEST(MusaIdentityDropoutTest, IdentityAddMultiOpNoCpuFallback) {
  IdentityAddTester test;
  const std::vector<std::string> dim_params{"batch", "seq"};
  test.AddInput<float>("X", {2, 3}, {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f}, false, &dim_params);
  test.AddInput<float>("Bias", {2, 3}, {0.5f, 0.5f, 0.5f, 1.0f, 1.0f, 1.0f});
  test.AddOutput<float>("Y", {2, 3}, {1.5f, 2.5f, 3.5f, 0.0f, -1.0f, -2.0f});
  RunMusaNoFallback(test);
}

TEST(MusaIdentityDropoutTest, DropoutFloatDynamicShapeCpuParity) {
  CompareOpTester test("Dropout", 10);
  const std::vector<std::string> dim_params{"batch", "seq"};
  test.AddInput<float>("X", {2, 3}, {1.0f, 0.0f, 2.0f, -1.0f, 3.0f, 4.0f}, false, &dim_params);
  test.AddOutput<float>("Y", {2, 3}, {1.0f, 0.0f, 2.0f, -1.0f, 3.0f, 4.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaIdentityDropoutTest, DropoutFloat16NoCpuFallback) {
  OpTester test("Dropout", 10);
  test.AddInput<MLFloat16>("X", {4}, ToFloat16({1.0f, 2.0f, -1.0f, 0.0f}));
  test.AddOutput<MLFloat16>("Y", {4}, ToFloat16({1.0f, 2.0f, -1.0f, 0.0f}));
  RunMusaNoFallback(test);
}

TEST(MusaIdentityDropoutTest, DropoutOptionalMaskNoCpuFallback) {
  OpTester test("Dropout", 10);
  test.AddInput<float>("X", {2, 2}, {1.0f, 2.0f, 3.0f, 5.0f});
  test.AddOutput<float>("Y", {2, 2}, {1.0f, 2.0f, 3.0f, 5.0f});
  test.AddOutput<bool>("mask", {2, 2}, {false, false, false, false});
  RunMusaNoFallback(test);
}

}  // namespace test
}  // namespace onnxruntime
