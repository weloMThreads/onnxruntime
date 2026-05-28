// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_session_options_config_keys.h"
#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/compare_provider_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

#include <cmath>
#include <vector>

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

class ReduceProdAddLogTester : public CompareOpTester {
 public:
  ReduceProdAddLogTester() : CompareOpTester("ReduceProdAddLog", 13) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto reduced_tensor = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2, 1});
    auto& reduced_arg = graph.GetOrCreateNodeArg("Reduced", &reduced_tensor);
    auto& sum_arg = graph.GetOrCreateNodeArg("Sum", &reduced_tensor);

    auto& reduce_node = graph.AddNode("reduceprod_node", "ReduceProd", "ReduceProd node",
                                      {graph_input_defs[0]}, {&reduced_arg});
    reduce_node.AddAttribute("axes", std::vector<int64_t>{1});
    reduce_node.AddAttribute("keepdims", int64_t{1});
    graph.AddNode("add_node", "Add", "Add node",
                  {&reduced_arg, graph_input_defs[1]}, {&sum_arg});
    graph.AddNode("log_node", "Log", "Log node",
                  {&sum_arg}, {graph_output_defs[0]});
  }
};


class ReduceSumSquareAddLogTester : public CompareOpTester {
 public:
  ReduceSumSquareAddLogTester() : CompareOpTester("ReduceSumSquareAddLog", 17) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto reduced_tensor = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2, 1});
    auto& reduced_arg = graph.GetOrCreateNodeArg("ReducedSumSquare", &reduced_tensor);
    auto& sum_arg = graph.GetOrCreateNodeArg("SumSquarePlusBias", &reduced_tensor);

    auto& reduce_node = graph.AddNode("reducesumsquare_node", "ReduceSumSquare", "ReduceSumSquare node",
                                      {graph_input_defs[0]}, {&reduced_arg});
    reduce_node.AddAttribute("axes", std::vector<int64_t>{1});
    reduce_node.AddAttribute("keepdims", int64_t{1});
    graph.AddNode("add_node", "Add", "Add node",
                  {&reduced_arg, graph_input_defs[1]}, {&sum_arg});
    graph.AddNode("log_node", "Log", "Log node",
                  {&sum_arg}, {graph_output_defs[0]});
  }
};

class ReduceMaxAddLogTester : public CompareOpTester {
 public:
  ReduceMaxAddLogTester() : CompareOpTester("ReduceMaxAddLog", 17) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto reduced_tensor = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2, 1});
    auto& reduced_arg = graph.GetOrCreateNodeArg("ReducedMax", &reduced_tensor);
    auto& sum_arg = graph.GetOrCreateNodeArg("MaxPlusBias", &reduced_tensor);

    auto& reduce_node = graph.AddNode("reducemax_node", "ReduceMax", "ReduceMax node",
                                      {graph_input_defs[0]}, {&reduced_arg});
    reduce_node.AddAttribute("axes", std::vector<int64_t>{1});
    reduce_node.AddAttribute("keepdims", int64_t{1});
    graph.AddNode("add_node", "Add", "Add node",
                  {&reduced_arg, graph_input_defs[1]}, {&sum_arg});
    graph.AddNode("log_node", "Log", "Log node",
                  {&sum_arg}, {graph_output_defs[0]});
  }
};

}  // namespace

TEST(MusaReduceTest, ReduceMeanBceHiddenFp16NoCpuFallback) {
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  constexpr int64_t batch = 8;
  constexpr int64_t seq = 1;
  constexpr int64_t hidden = 768;
  std::vector<float> input(static_cast<size_t>(batch * seq * hidden));
  std::vector<float> expected(static_cast<size_t>(batch * seq));

  for (int64_t b = 0; b < batch; ++b) {
    float sum = 0.0f;
    for (int64_t h = 0; h < hidden; ++h) {
      const float value = static_cast<float>((b * 17 + h * 13) % 251 - 125) / 64.0f;
      input[static_cast<size_t>(b * hidden + h)] = value;
      sum += value;
    }
    expected[static_cast<size_t>(b)] = sum / static_cast<float>(hidden);
  }

  OpTester test("ReduceMean", 13);
  test.AddAttribute("axes", std::vector<int64_t>{-1});
  test.AddAttribute("keepdims", static_cast<int64_t>(1));
  test.AddInput<MLFloat16>("data", {batch, seq, hidden}, ToFloat16(input));
  test.AddOutput<MLFloat16>("reduced", {batch, seq, 1}, ToFloat16(expected));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(musa_provider));
  test.Run(MakeNoFallbackSessionOptions(),
           OpTester::ExpectResult::kExpectSuccess,
           "",
           {},
           nullptr,
           &execution_providers);
}

TEST(MusaReduceTest, ReduceL2KeepDimsNoCpuFallback) {
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  OpTester test("ReduceL2", 13);
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", static_cast<int64_t>(1));
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,
                        5.0f, 6.0f,
                        7.0f, 8.0f,
                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 2, 1},
                        {15.71623325f, 20.07485962f});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(musa_provider));
  test.Run(MakeNoFallbackSessionOptions(),
           OpTester::ExpectResult::kExpectSuccess,
           "",
           {},
           nullptr,
           &execution_providers);
}

TEST(MusaReduceTest, ReduceL2AxesInputNoCpuFallback) {
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  OpTester test("ReduceL2", 18);
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddAttribute("noop_with_empty_axes", static_cast<int64_t>(0));
  test.AddInput<float>("data", {2, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,
                        5.0f, 6.0f,
                        7.0f, 8.0f});
  test.AddInput<int64_t>("axes", {2}, {0, 2});
  test.AddOutput<float>("reduced", {2},
                        {8.12403870f, 11.74734020f});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(musa_provider));
  test.Run(MakeNoFallbackSessionOptions(),
           OpTester::ExpectResult::kExpectSuccess,
           "",
           {},
           nullptr,
           &execution_providers);
}


TEST(MusaReduceTest, ReduceSumSquareFloatOpset17NoCpuFallback) {
  CompareOpTester test("ReduceSumSquare", 17);
  test.AddAttribute("axes", std::vector<int64_t>{0, 2, 3});
  test.AddAttribute("keepdims", int64_t{0});
  test.AddInput<float>("data", {1, 2, 3, 1}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddOutput<float>("reduced", {2}, {14.0f, 77.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaReduceTest, ReduceSumSquareFloat16Opset18NoCpuFallback) {
  CompareOpTester test("ReduceSumSquare", 18);
  test.AddAttribute("keepdims", int64_t{1});
  test.AddInput<MLFloat16>("data", {2, 2}, ToFloat16({1.0f, 2.0f, 3.0f, 4.0f}));
  test.AddInput<int64_t>("axes", {1}, {0});
  test.AddOutput<MLFloat16>("reduced", {1, 2}, ToFloat16({10.0f, 20.0f}));
  CompareWithMusaNoFallback(test, true, 1e-3, 1e-3);
}

TEST(MusaReduceTest, ReduceSumSquareEmptyInputNoCpuFallback) {
  CompareOpTester test("ReduceSumSquare", 17);
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", int64_t{1});
  test.AddInput<float>("data", {0, 2, 3}, {});
  test.AddOutput<float>("reduced", {1, 2, 3}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaReduceTest, ReduceSumSquareDynamicShapeNoCpuFallback) {
  CompareOpTester test("ReduceSumSquare", 17);
  const std::vector<std::string> dim_params{"batch", "seq"};
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", int64_t{0});
  test.AddInput<float>("data", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, false, &dim_params);
  test.AddOutput<float>("reduced", {2}, {14.0f, 77.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaReduceTest, ReduceSumSquareAddLogMultiOpNoCpuFallback) {
  ReduceSumSquareAddLogTester test;
  test.AddInput<float>("data", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddInput<float>("bias", {2, 1}, {1.0f, 2.0f});
  test.AddOutput<float>("log_out", {2, 1}, {static_cast<float>(std::log(15.0f)), static_cast<float>(std::log(79.0f))});
  CompareWithMusaNoFallback(test);
}

TEST(MusaPoolTest, GlobalAveragePool3DNoCpuFallback) {
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  OpTester test("GlobalAveragePool", 1);
  test.AddInput<float>("X", {1, 2, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        5.0f, 6.0f, 7.0f, 8.0f});
  test.AddOutput<float>("Y", {1, 2, 1}, {2.5f, 6.5f});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(musa_provider));
  test.Run(MakeNoFallbackSessionOptions(),
           OpTester::ExpectResult::kExpectSuccess,
           "",
           {},
           nullptr,
           &execution_providers);
}

TEST(MusaPoolTest, MaxPool2DNoCpuFallback) {
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  OpTester test("MaxPool", 12);
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});
  test.AddAttribute("strides", std::vector<int64_t>{2, 2});
  test.AddInput<float>("X", {1, 1, 4, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        5.0f, 6.0f, 7.0f, 8.0f,
                        9.0f, 10.0f, 11.0f, 12.0f,
                        13.0f, 14.0f, 15.0f, 16.0f});
  test.AddOutput<float>("Y", {1, 1, 2, 2}, {6.0f, 8.0f, 14.0f, 16.0f});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(musa_provider));
  test.Run(MakeNoFallbackSessionOptions(),
           OpTester::ExpectResult::kExpectSuccess,
           "",
           {},
           nullptr,
           &execution_providers);
}

TEST(MusaPoolTest, GlobalMaxPoolNoCpuFallback) {
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  OpTester test("GlobalMaxPool", 1);
  test.AddInput<float>("X", {1, 2, 4}, {1.0f, 3.0f, 2.0f, 4.0f, 8.0f, 5.0f, 7.0f, 6.0f});
  test.AddOutput<float>("Y", {1, 2, 1}, {4.0f, 8.0f});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(musa_provider));
  test.Run(MakeNoFallbackSessionOptions(),
           OpTester::ExpectResult::kExpectSuccess,
           "",
           {},
           nullptr,
           &execution_providers);
}

TEST(MusaTensorTest, FlattenNoCpuFallback) {
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  OpTester test("Flatten", 13);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddInput<float>("input", {2, 3, 4}, 
                       {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
                        13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f});
  test.AddOutput<float>("output", {2, 12},
                        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
                         13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(musa_provider));
  test.Run(MakeNoFallbackSessionOptions(),
           OpTester::ExpectResult::kExpectSuccess,
           "",
           {},
           nullptr,
           &execution_providers);
}


TEST(MusaReduceTest, ReduceMaxFloatOpset17NoCpuFallback) {
  CompareOpTester test("ReduceMax", 17);
  test.AddAttribute("axes", std::vector<int64_t>{2});
  test.AddAttribute("keepdims", int64_t{0});
  test.AddInput<float>("data", {1, 2, 2, 1}, {1.0f, 3.0f, 2.0f, 4.0f});
  test.AddOutput<float>("reduced", {1, 2, 1}, {3.0f, 4.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaReduceTest, ReduceMaxFloat16Opset17NoCpuFallback) {
  CompareOpTester test("ReduceMax", 17);
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", int64_t{1});
  test.AddInput<MLFloat16>("data", {2, 3}, ToFloat16({1.0f, 5.0f, 3.0f, -2.0f, -4.0f, -1.0f}));
  test.AddOutput<MLFloat16>("reduced", {2, 1}, ToFloat16({5.0f, -1.0f}));
  CompareWithMusaNoFallback(test, true, 1e-3, 1e-3);
}

TEST(MusaReduceTest, ReduceMaxScalarOpset17NoCpuFallback) {
  CompareOpTester test("ReduceMax", 17);
  test.AddAttribute("keepdims", int64_t{0});
  test.AddInput<float>("data", {}, {2.5f});
  test.AddOutput<float>("reduced", {}, {2.5f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaReduceTest, ReduceMaxAddLogMultiOpOpset17NoCpuFallback) {
  ReduceMaxAddLogTester test;
  test.AddInput<float>("data", {2, 3}, {1.0f, 5.0f, 3.0f, 2.0f, 4.0f, 6.0f});
  test.AddInput<float>("bias", {2, 1}, {1.0f, 2.0f});
  test.AddOutput<float>("log_out", {2, 1}, {1.79175949f, 2.07944154f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaReduceTest, ReduceProdFloatNoCpuFallback) {
  CompareOpTester test("ReduceProd", 13);
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", int64_t{1});
  test.AddInput<float>("data", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 1.0f});
  test.AddOutput<float>("reduced", {2, 1}, {6.0f, 20.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaReduceTest, ReduceProdFloat16NoCpuFallback) {
  CompareOpTester test("ReduceProd", 13);
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", int64_t{0});
  test.AddInput<MLFloat16>("data", {2, 3}, ToFloat16({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
  test.AddOutput<MLFloat16>("reduced", {3}, ToFloat16({4.0f, 10.0f, 18.0f}));
  CompareWithMusaNoFallback(test, true, 1e-3, 1e-3);
}

TEST(MusaReduceTest, ReduceProdInt32DynamicShapeNoCpuFallback) {
  CompareOpTester test("ReduceProd", 13);
  const std::vector<std::string> dim_params{"batch", "seq"};
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", int64_t{0});
  test.AddInput<int32_t>("data", {2, 3}, {1, 2, 3, 4, 5, 1}, false, &dim_params);
  test.AddOutput<int32_t>("reduced", {2}, {6, 20});
  CompareWithMusaNoFallback(test);
}

TEST(MusaReduceTest, ReduceProdInt64ScalarOutputNoCpuFallback) {
  CompareOpTester test("ReduceProd", 13);
  test.AddAttribute("keepdims", int64_t{0});
  test.AddInput<int64_t>("data", {3}, {2, 3, 4});
  test.AddOutput<int64_t>("reduced", {}, {24});
  CompareWithMusaNoFallback(test);
}

TEST(MusaReduceTest, ReduceProdFloatEmptyNoCpuFallback) {
  CompareOpTester test("ReduceProd", 13);
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", int64_t{0});
  const std::vector<float> empty;
  test.AddInput<float>("data", {0, 2}, empty);
  test.AddOutput<float>("reduced", {0}, empty);
  CompareWithMusaNoFallback(test);
}

TEST(MusaReduceTest, ReduceProdAddLogMultiOpNoCpuFallback) {
  ReduceProdAddLogTester test;
  test.AddInput<float>("data", {2, 3}, {1.0f, 2.0f, 3.0f, 2.0f, 2.0f, 2.0f});
  test.AddInput<float>("bias", {2, 1}, {1.0f, 1.0f});
  test.AddOutput<float>("log_out", {2, 1}, {1.94591015f, 2.19722462f});
  CompareWithMusaNoFallback(test);
}

}  // namespace test
}  // namespace onnxruntime
