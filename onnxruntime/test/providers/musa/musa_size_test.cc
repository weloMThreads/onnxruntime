// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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

class SizeAddTester : public CompareOpTester {
 public:
  SizeAddTester() : CompareOpTester("SizeAdd", 13) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    graph.AddNode("size_node", "Size", "Size node", {graph_input_defs[0]}, {graph_output_defs[0]});
    graph.AddNode("add_node", "Add", "Add node", {graph_input_defs[0], graph_input_defs[1]}, {graph_output_defs[1]});
  }
};

}  // namespace

TEST(MusaSizeTest, FloatDynamicShapeCpuParity) {
  CompareOpTester test("Size", 13);
  const std::vector<std::string> dim_params{"batch", "seq"};
  test.AddInput<float>("X", {2, 3}, {1.0f, 0.0f, 2.0f, -1.0f, 3.0f, 4.0f}, false, &dim_params);
  test.AddOutput<int64_t>("Y", {}, {6LL});
  CompareWithMusaNoFallback(test);
}

TEST(MusaSizeTest, Int32ScalarCpuParity) {
  CompareOpTester test("Size", 13);
  test.AddInput<int32_t>("X", {}, {7});
  test.AddOutput<int64_t>("Y", {}, {1LL});
  CompareWithMusaNoFallback(test);
}

TEST(MusaSizeTest, Int64EmptyCpuParity) {
  CompareOpTester test("Size", 13);
  test.AddInput<int64_t>("X", {2, 0}, std::vector<int64_t>{});
  test.AddOutput<int64_t>("Y", {}, {0LL});
  CompareWithMusaNoFallback(test);
}

TEST(MusaSizeTest, BoolNoCpuFallback) {
  OpTester test("Size", 13);
  test.AddInput<bool>("X", {2, 2}, {true, false, true, true});
  test.AddOutput<int64_t>("Y", {}, {4LL});
  RunMusaNoFallback(test);
}

TEST(MusaSizeTest, Float16NoCpuFallback) {
  OpTester test("Size", 13);
  test.AddInput<MLFloat16>("X", {4}, ToFloat16({2.0f, -1.0f, 0.5f, 3.0f}));
  test.AddOutput<int64_t>("Y", {}, {4LL});
  RunMusaNoFallback(test);
}

TEST(MusaSizeTest, SizeAndAddMultiOpNoCpuFallback) {
  SizeAddTester test;
  const std::vector<std::string> dim_params{"batch", "seq"};
  test.AddInput<float>("X", {2, 3}, {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f}, false, &dim_params);
  test.AddInput<float>("Bias", {2, 3}, {0.5f, 0.5f, 0.5f, 1.0f, 1.0f, 1.0f});
  test.AddOutput<int64_t>("SizeY", {}, {6LL});
  test.AddOutput<float>("AddY", {2, 3}, {1.5f, 2.5f, 3.5f, 0.0f, -1.0f, -2.0f});
  CompareWithMusaNoFallback(test);
}

}  // namespace test
}  // namespace onnxruntime
