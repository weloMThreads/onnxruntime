// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/onnx_protobuf.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/compare_provider_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

#include <algorithm>
#include <cstdint>
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

class NonZeroCastAddTester : public CompareOpTester {
 public:
  NonZeroCastAddTester() : CompareOpTester("NonZeroCastAdd", 13) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto int64_indices = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_INT64, {2, 2});
    auto float_indices = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2, 2});
    auto& nonzero_arg = graph.GetOrCreateNodeArg("NonZeroOut", &int64_indices);
    auto& cast_arg = graph.GetOrCreateNodeArg("CastOut", &float_indices);

    graph.AddNode("nonzero_node", "NonZero", "NonZero node", {graph_input_defs[0]}, {&nonzero_arg});
    auto& cast_node = graph.AddNode("cast_node", "Cast", "Cast node", {&nonzero_arg}, {&cast_arg});
    cast_node.AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
    graph.AddNode("add_node", "Add", "Add node", {&cast_arg, graph_input_defs[1]}, {graph_output_defs[0]});
  }
};

}  // namespace

TEST(MusaNonZeroTest, FloatDynamicShape2D) {
  CompareOpTester test("NonZero", 13);
  const std::vector<std::string> dim_params{"batch", "seq"};
  test.AddInput<float>("X", {2, 3}, {1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f}, false, &dim_params);
  test.AddOutput<int64_t>("Y", {2, 3}, {0LL, 0LL, 1LL, 0LL, 2LL, 1LL});
  CompareWithMusaNoFallback(test);
}

TEST(MusaNonZeroTest, BoolScalarTrue) {
  CompareOpTester test("NonZero", 13);
  test.AddInput<bool>("X", {}, {true});
  test.AddOutput<int64_t>("Y", {1, 1}, {0LL});
  CompareWithMusaNoFallback(test);
}

TEST(MusaNonZeroTest, Int32ScalarZero) {
  CompareOpTester test("NonZero", 13);
  test.AddInput<int32_t>("X", {}, {0});
  test.AddOutput<int64_t>("Y", {1, 0}, std::vector<int64_t>{});
  CompareWithMusaNoFallback(test);
}

TEST(MusaNonZeroTest, Int64Vector) {
  CompareOpTester test("NonZero", 13);
  test.AddInput<int64_t>("X", {5}, {-1LL, 0LL, 2LL, 0LL, 7LL});
  test.AddOutput<int64_t>("Y", {1, 3}, {0LL, 2LL, 4LL});
  CompareWithMusaNoFallback(test);
}

TEST(MusaNonZeroTest, Uint8Empty) {
  CompareOpTester test("NonZero", 13);
  test.AddInput<uint8_t>("X", {2, 0}, std::vector<uint8_t>{});
  test.AddOutput<int64_t>("Y", {2, 0}, std::vector<int64_t>{});
  CompareWithMusaNoFallback(test);
}

TEST(MusaNonZeroTest, Float16NoCpuFallback) {
  OpTester test("NonZero", 13);
  test.AddInput<MLFloat16>("X", {2, 3}, ToFloat16({0.0f, 4.0f, 0.0f, -1.0f, 0.0f, 2.0f}));
  test.AddOutput<int64_t>("Y", {2, 3}, {0LL, 1LL, 1LL, 1LL, 0LL, 2LL});
  RunMusaNoFallback(test);
}

TEST(MusaNonZeroTest, NonZeroCastAddMultiOpNoCpuFallback) {
  NonZeroCastAddTester test;
  test.AddInput<float>("X", {2, 2}, {0.0f, 5.0f, 6.0f, 0.0f});
  test.AddInput<float>("Bias", {2, 2}, {0.5f, 0.5f, 1.0f, 1.0f});
  test.AddOutput<float>("Y", {2, 2}, {0.5f, 1.5f, 2.0f, 1.0f});
  CompareWithMusaNoFallback(test);
}

}  // namespace test
}  // namespace onnxruntime
