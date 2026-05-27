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
#include <utility>
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

class RoundAddRoundTester : public CompareOpTester {
 public:
  RoundAddRoundTester() : CompareOpTester("RoundAddRound", 13) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto float_tensor = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2, 3});
    auto& rounded_arg = graph.GetOrCreateNodeArg("Rounded", &float_tensor);
    auto& sum_arg = graph.GetOrCreateNodeArg("Sum", &float_tensor);

    graph.AddNode("round1_node", "Round", "Round node",
                  {graph_input_defs[0]}, {&rounded_arg});
    graph.AddNode("add_node", "Add", "Add node",
                  {&rounded_arg, graph_input_defs[1]}, {&sum_arg});
    graph.AddNode("round2_node", "Round", "Round node",
                  {&sum_arg}, {graph_output_defs[0]});
  }
};


class RsqrtSquareLog1pTester : public CompareOpTester {
 public:
  explicit RsqrtSquareLog1pTester(std::vector<int64_t> dims,
                                  int32_t elem_type = ONNX_NAMESPACE::TensorProto_DataType_FLOAT)
      : CompareOpTester("RsqrtSquareLog1p", 13), dims_(std::move(dims)), elem_type_(elem_type) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto tensor_type = MakeTensorType(elem_type_, dims_);
    auto& sqrt_arg = graph.GetOrCreateNodeArg("SqrtOut", &tensor_type);
    auto& rsqrt_arg = graph.GetOrCreateNodeArg("RsqrtOut", &tensor_type);
    auto& square_arg = graph.GetOrCreateNodeArg("SquareOut", &tensor_type);
    auto& plus_one_arg = graph.GetOrCreateNodeArg("PlusOne", &tensor_type);
    auto& log1p_arg = graph.GetOrCreateNodeArg("Log1pOut", &tensor_type);

    graph.AddNode("sqrt_node", "Sqrt", "TF Rsqrt decomposition sqrt",
                  {graph_input_defs[0]}, {&sqrt_arg});
    graph.AddNode("div_node", "Div", "TF Rsqrt decomposition reciprocal",
                  {graph_input_defs[1], &sqrt_arg}, {&rsqrt_arg});
    graph.AddNode("square_node", "Mul", "TF Square decomposition",
                  {&rsqrt_arg, &rsqrt_arg}, {&square_arg});
    graph.AddNode("add_one_node", "Add", "TF Log1p decomposition add one",
                  {graph_input_defs[0], graph_input_defs[1]}, {&plus_one_arg});
    graph.AddNode("log_node", "Log", "TF Log1p decomposition log",
                  {&plus_one_arg}, {&log1p_arg});
    graph.AddNode("sum_node", "Add", "Combine model3 unary decompositions",
                  {&square_arg, &log1p_arg}, {graph_output_defs[0]});
  }

 private:
  std::vector<int64_t> dims_;
  int32_t elem_type_;
};

class SplitSqueezeConcatTester : public CompareOpTester {
 public:
  explicit SplitSqueezeConcatTester(int32_t elem_type)
      : CompareOpTester("SplitSqueezeConcat", 13), elem_type_(elem_type) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto split0_type = MakeTensorType(elem_type_, {2, 2});
    auto split1_type = MakeTensorType(elem_type_, {2, 1});
    auto split2_type = MakeTensorType(elem_type_, {2, 3});
    auto squeezed_type = MakeTensorType(elem_type_, {2});
    auto& split0_arg = graph.GetOrCreateNodeArg("SplitOut0", &split0_type);
    auto& split1_arg = graph.GetOrCreateNodeArg("SplitOut1", &split1_type);
    auto& split2_arg = graph.GetOrCreateNodeArg("SplitOut2", &split2_type);
    auto& squeezed_arg = graph.GetOrCreateNodeArg("Squeezed", &squeezed_type);
    auto& restored_arg = graph.GetOrCreateNodeArg("Restored", &split1_type);

    auto& split_node = graph.AddNode("split_node", "Split", "TF SplitV decomposition",
                                     {graph_input_defs[0], graph_input_defs[1]},
                                     {&split0_arg, &split1_arg, &split2_arg});
    split_node.AddAttribute("axis", int64_t{1});
    graph.AddNode("squeeze_node", "Squeeze", "TF Unpack decomposition squeeze",
                  {&split1_arg, graph_input_defs[2]}, {&squeezed_arg});
    graph.AddNode("unsqueeze_node", "Unsqueeze", "TF Pack decomposition unsqueeze",
                  {&squeezed_arg, graph_input_defs[2]}, {&restored_arg});
    auto& concat_node = graph.AddNode("concat_node", "Concat", "TF Pack decomposition concat",
                                      {&split0_arg, &restored_arg, &split2_arg}, {graph_output_defs[0]});
    concat_node.AddAttribute("axis", int64_t{1});
  }

 private:
  int32_t elem_type_;
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

TEST(MusaComparisonElementwiseTest, BitwiseAndInt32BroadcastDynamicShape) {
  CompareOpTester test("BitwiseAnd", 18);
  const std::vector<std::string> lhs_dim_params{"batch", "1"};
  const std::vector<std::string> rhs_dim_params{"1", "seq"};
  test.AddInput<int32_t>("A", {2, 1}, {7, 12}, false, &lhs_dim_params);
  test.AddInput<int32_t>("B", {1, 3}, {3, 10, 15}, false, &rhs_dim_params);
  test.AddOutput<int32_t>("C", {2, 3}, {3, 2, 7, 0, 8, 12});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, BitwiseAndInt64Scalar) {
  CompareOpTester test("BitwiseAnd", 18);
  test.AddInput<int64_t>("A", {4}, {1, 2, 3, 4});
  test.AddInput<int64_t>("B", {}, {3});
  test.AddOutput<int64_t>("C", {4}, {1, 2, 3, 0});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, BitwiseAndUInt8) {
  CompareOpTester test("BitwiseAnd", 18);
  test.AddInput<uint8_t>("A", {4}, {0xff, 0x0f, 0xf0, 0x55});
  test.AddInput<uint8_t>("B", {4}, {0x0f, 0xf0, 0xff, 0x33});
  test.AddOutput<uint8_t>("C", {4}, {0x0f, 0x00, 0xf0, 0x11});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, BitwiseAndInt32Empty) {
  CompareOpTester test("BitwiseAnd", 18);
  const std::vector<int32_t> empty;
  test.AddInput<int32_t>("A", {0}, empty);
  test.AddInput<int32_t>("B", {0}, empty);
  test.AddOutput<int32_t>("C", {0}, empty);
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, RoundFloatDynamicShape) {
  CompareOpTester test("Round", 13);
  const std::vector<std::string> dim_params{"batch", "seq"};
  test.AddInput<float>("X", {2, 3}, {0.9f, 2.5f, 2.3f, 1.5f, -4.5f, -0.6f}, false, &dim_params);
  test.AddOutput<float>("Y", {2, 3}, {1.0f, 2.0f, 2.0f, 2.0f, -4.0f, -1.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, RoundFloat16Scalar) {
  CompareOpTester test("Round", 13);
  test.AddInput<MLFloat16>("X", {}, ToFloat16({2.5f}));
  test.AddOutput<MLFloat16>("Y", {}, ToFloat16({2.0f}));
  CompareWithMusaNoFallback(test, true, 1e-3, 1e-3);
}

TEST(MusaComparisonElementwiseTest, RoundFloatEmpty) {
  CompareOpTester test("Round", 13);
  const std::vector<float> empty;
  test.AddInput<float>("X", {0}, empty);
  test.AddOutput<float>("Y", {0}, empty);
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, RoundAddRoundMultiOpGraph) {
  RoundAddRoundTester test;
  test.AddInput<float>("X", {2, 3}, {0.5f, 1.5f, 2.5f, -0.5f, -1.5f, 3.2f});
  test.AddInput<float>("Bias", {2, 3}, {0.6f, 0.5f, 1.5f, -0.6f, -0.5f, -3.6f});
  test.AddOutput<float>("Y", {2, 3}, {1.0f, 2.0f, 4.0f, -1.0f, -2.0f, -1.0f});
  CompareWithMusaNoFallback(test);
}


TEST(MusaComparisonElementwiseTest, WhereFloat16BroadcastDynamicShape) {
  CompareOpTester test("Where", 16);
  const std::vector<std::string> cond_dim_params{"batch", "1"};
  const std::vector<std::string> x_dim_params{"batch", "seq"};
  test.AddInput<bool>("Cond", {2, 1}, {true, false}, false, &cond_dim_params);
  test.AddInput<MLFloat16>("X", {2, 3}, ToFloat16({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}), false, &x_dim_params);
  test.AddInput<MLFloat16>("Y", {}, ToFloat16({9.0f}));
  test.AddOutput<MLFloat16>("Z", {2, 3}, ToFloat16({1.0f, 2.0f, 3.0f, 9.0f, 9.0f, 9.0f}));
  CompareWithMusaNoFallback(test, true, 1e-3, 1e-3);
}

TEST(MusaComparisonElementwiseTest, WhereInt32BroadcastScalar) {
  CompareOpTester test("Where", 16);
  test.AddInput<bool>("Cond", {1, 3}, {true, false, true});
  test.AddInput<int32_t>("X", {}, {7});
  test.AddInput<int32_t>("Y", {2, 3}, {1, 2, 3, 4, 5, 6});
  test.AddOutput<int32_t>("Z", {2, 3}, {7, 2, 7, 7, 5, 7});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, RsqrtSquareLog1pFloatDynamicShape) {
  RsqrtSquareLog1pTester test({2, 3});
  const std::vector<std::string> dim_params{"batch", "seq"};
  test.AddInput<float>("X", {2, 3}, {1.0f, 3.0f, 8.0f, 15.0f, 24.0f, 35.0f}, false, &dim_params);
  test.AddInput<float>("One", {}, {1.0f}, true);
  test.AddOutput<float>("Y", {2, 3}, {1.6931472f, 1.7196277f, 2.3222246f, 2.8392551f, 3.2605429f, 3.61209f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, RsqrtSquareLog1pFloatScalar) {
  RsqrtSquareLog1pTester test({});
  test.AddInput<float>("X", {}, {4.0f});
  test.AddInput<float>("One", {}, {1.0f}, true);
  test.AddOutput<float>("Y", {}, {1.859438f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, RsqrtSquareLog1pFloat16) {
  RsqrtSquareLog1pTester test({2, 2}, ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  test.AddInput<MLFloat16>("X", {2, 2}, ToFloat16({1.0f, 4.0f, 16.0f, 64.0f}));
  test.AddInput<MLFloat16>("One", {}, ToFloat16({1.0f}), true);
  test.AddOutput<MLFloat16>("Y", {2, 2}, ToFloat16({1.6931472f, 1.859438f, 2.895213f, 4.1899705f}));
  CompareWithMusaNoFallback(test, true, 1e-3, 1e-3);
}

TEST(MusaComparisonElementwiseTest, SplitSqueezeConcatFloatNoCpuFallback) {
  SplitSqueezeConcatTester test(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  test.AddInput<float>("X", {2, 6}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                    7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
  test.AddInput<int64_t>("SplitSizes", {3}, {2, 1, 3}, true);
  test.AddInput<int64_t>("Axes", {1}, {1}, true);
  test.AddOutput<float>("Y", {2, 6}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                    7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, SplitSqueezeConcatInt32NoCpuFallback) {
  SplitSqueezeConcatTester test(ONNX_NAMESPACE::TensorProto_DataType_INT32);
  test.AddInput<int32_t>("X", {2, 6}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  test.AddInput<int64_t>("SplitSizes", {3}, {2, 1, 3}, true);
  test.AddInput<int64_t>("Axes", {1}, {1}, true);
  test.AddOutput<int32_t>("Y", {2, 6}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  CompareWithMusaNoFallback(test);
}

}  // namespace test
}  // namespace onnxruntime
