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

class UniqueAddTester : public CompareOpTester {
 public:
  UniqueAddTester() : CompareOpTester("UniqueAdd", 11) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto unique_type = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {3});
    auto& unique_arg = graph.GetOrCreateNodeArg("UniqueOut", &unique_type);

    auto& unique_node = graph.AddNode("unique_node", "Unique", "Unique node", {graph_input_defs[0]}, {&unique_arg});
    unique_node.AddAttribute("sorted", static_cast<int64_t>(1));
    graph.AddNode("add_node", "Add", "Add node", {&unique_arg, graph_input_defs[1]}, {graph_output_defs[0]});
  }
};

}  // namespace

TEST(MusaUniqueTest, FloatFlattenUnsortedDynamicShape) {
  CompareOpTester test("Unique", 11);
  const std::vector<std::string> dim_params{"batch", "seq"};
  test.AddAttribute("sorted", static_cast<int64_t>(0));
  test.AddInput<float>("X", {2, 3}, {1.0f, 4.0f, 1.0f, 2.0f, 2.0f, 0.0f}, false, &dim_params);
  test.AddOutput<float>("Y", {4}, {1.0f, 4.0f, 2.0f, 0.0f});
  test.AddOutput<int64_t>("indices", {4}, {0LL, 1LL, 3LL, 5LL});
  test.AddOutput<int64_t>("inverse_indices", {6}, {0LL, 1LL, 0LL, 2LL, 2LL, 3LL});
  test.AddOutput<int64_t>("counts", {4}, {2LL, 1LL, 2LL, 1LL});
  CompareWithMusaNoFallback(test);
}

TEST(MusaUniqueTest, FloatFlattenSorted) {
  CompareOpTester test("Unique", 11);
  test.AddAttribute("sorted", static_cast<int64_t>(1));
  test.AddInput<float>("X", {2, 3}, {1.0f, 4.0f, 1.0f, 2.0f, 2.0f, 0.0f});
  test.AddOutput<float>("Y", {4}, {0.0f, 1.0f, 2.0f, 4.0f});
  test.AddOutput<int64_t>("indices", {4}, {5LL, 0LL, 3LL, 1LL});
  test.AddOutput<int64_t>("inverse_indices", {6}, {1LL, 3LL, 1LL, 2LL, 2LL, 0LL});
  test.AddOutput<int64_t>("counts", {4}, {1LL, 2LL, 2LL, 1LL});
  CompareWithMusaNoFallback(test);
}

TEST(MusaUniqueTest, DoubleFlattenUnsorted) {
  CompareOpTester test("Unique", 11);
  test.AddAttribute("sorted", static_cast<int64_t>(0));
  test.AddInput<double>("X", {2, 2}, {3.14, -1.3, 3.14, -1.3});
  test.AddOutput<double>("Y", {2}, {3.14, -1.3});
  test.AddOutput<int64_t>("indices", {2}, {0LL, 1LL});
  test.AddOutput<int64_t>("inverse_indices", {4}, {0LL, 1LL, 0LL, 1LL});
  test.AddOutput<int64_t>("counts", {2}, {2LL, 2LL});
  CompareWithMusaNoFallback(test);
}

TEST(MusaUniqueTest, Int64Axis0OneDim) {
  CompareOpTester test("Unique", 11);
  test.AddAttribute("axis", static_cast<int64_t>(0));
  test.AddAttribute("sorted", static_cast<int64_t>(1));
  test.AddInput<int64_t>("X", {6}, {2LL, 1LL, 1LL, 3LL, 4LL, 3LL});
  test.AddOutput<int64_t>("Y", {4}, {1LL, 2LL, 3LL, 4LL});
  test.AddOutput<int64_t>("indices", {4}, {1LL, 0LL, 3LL, 4LL});
  test.AddOutput<int64_t>("inverse_indices", {6}, {1LL, 0LL, 0LL, 2LL, 3LL, 2LL});
  test.AddOutput<int64_t>("counts", {4}, {2LL, 1LL, 2LL, 1LL});
  CompareWithMusaNoFallback(test);
}

TEST(MusaUniqueTest, Int8NoOptionalOutputs) {
  CompareOpTester test("Unique", 11);
  test.AddAttribute("sorted", static_cast<int64_t>(1));
  test.AddInput<int8_t>("X", {2, 4}, {1, 4, -1, 2, 2, 0, -1, 4});
  test.AddOutput<int8_t>("Y", {5}, {-1, 0, 1, 2, 4});
  CompareWithMusaNoFallback(test);
}

TEST(MusaUniqueTest, FloatEmpty) {
  CompareOpTester test("Unique", 11);
  test.AddInput<float>("X", {0}, std::vector<float>{});
  test.AddOutput<float>("Y", {0}, std::vector<float>{});
  test.AddOutput<int64_t>("indices", {0}, std::vector<int64_t>{});
  test.AddOutput<int64_t>("inverse_indices", {0}, std::vector<int64_t>{});
  test.AddOutput<int64_t>("counts", {0}, std::vector<int64_t>{});
  CompareWithMusaNoFallback(test);
}

TEST(MusaUniqueTest, Float16NoCpuFallback) {
  OpTester test("Unique", 11);
  test.AddAttribute("sorted", static_cast<int64_t>(0));
  test.AddInput<MLFloat16>("X", {5}, ToFloat16({2.0f, -1.0f, 2.0f, 0.5f, -1.0f}));
  test.AddOutput<MLFloat16>("Y", {3}, ToFloat16({2.0f, -1.0f, 0.5f}));
  test.AddOutput<int64_t>("indices", {3}, {0LL, 1LL, 3LL});
  test.AddOutput<int64_t>("inverse_indices", {5}, {0LL, 1LL, 0LL, 2LL, 1LL});
  test.AddOutput<int64_t>("counts", {3}, {2LL, 2LL, 1LL});
  RunMusaNoFallback(test);
}

TEST(MusaUniqueTest, Int32ScalarNoCpuFallback) {
  OpTester test("Unique", 11);
  test.AddInput<int32_t>("X", {}, {7});
  test.AddOutput<int32_t>("Y", {1}, {7});
  test.AddOutput<int64_t>("indices", {1}, {0LL});
  test.AddOutput<int64_t>("inverse_indices", {1}, {0LL});
  test.AddOutput<int64_t>("counts", {1}, {1LL});
  RunMusaNoFallback(test);
}

TEST(MusaUniqueTest, UniqueAddMultiOpNoCpuFallback) {
  UniqueAddTester test;
  test.AddInput<float>("X", {5}, {3.0f, 1.0f, 3.0f, 2.0f, 1.0f});
  test.AddInput<float>("Bias", {3}, {0.5f, 1.0f, 1.5f});
  test.AddOutput<float>("Y", {3}, {1.5f, 3.0f, 4.5f});
  CompareWithMusaNoFallback(test);
}

}  // namespace test
}  // namespace onnxruntime
