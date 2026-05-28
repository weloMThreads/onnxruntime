// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/onnx_protobuf.h"
#include "test/providers/compare_provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <limits>
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

class CompatComparisonChainTester : public CompareOpTester {
 public:
  CompatComparisonChainTester() : CompareOpTester("CompatComparisonChain", 17) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto bool_tensor = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_BOOL, {2, 3});
    auto& ge_arg = graph.GetOrCreateNodeArg("GreaterEqualOut", &bool_tensor);
    auto& le_arg = graph.GetOrCreateNodeArg("LessEqualOut", &bool_tensor);

    graph.AddNode("greater_equal_node", "GreaterEqual", "TF GreaterEqual compatibility",
                  {graph_input_defs[0], graph_input_defs[1]}, {&ge_arg});
    graph.AddNode("less_equal_node", "LessEqual", "TF LessEqual compatibility",
                  {graph_input_defs[2], graph_input_defs[3]}, {&le_arg});
    graph.AddNode("not_equal_node", "NotEqual", "TF NotEqual compatibility",
                  {&ge_arg, &le_arg}, {graph_output_defs[0]});
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

class MaximumMinimumAddTester : public CompareOpTester {
 public:
  MaximumMinimumAddTester() : CompareOpTester("MaximumMinimumAdd", 17) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto float_tensor = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2, 3});
    auto& max_arg = graph.GetOrCreateNodeArg("MaximumOut", &float_tensor);
    auto& min_arg = graph.GetOrCreateNodeArg("MinimumOut", &float_tensor);

    graph.AddNode("maximum_node", "Maximum", "TF Maximum compatibility",
                  {graph_input_defs[0], graph_input_defs[1]}, {&max_arg});
    graph.AddNode("minimum_node", "Minimum", "TF Minimum compatibility",
                  {&max_arg, graph_input_defs[2]}, {&min_arg});
    graph.AddNode("add_node", "Add", "Combine Maximum and Minimum",
                  {&min_arg, graph_input_defs[3]}, {graph_output_defs[0]});
  }
};

class TfMathAliasChainTester : public CompareOpTester {
 public:
  TfMathAliasChainTester() : CompareOpTester("TfMathAliasChain", 17) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto float_tensor = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2, 2});
    auto& add_arg = graph.GetOrCreateNodeArg("AddV2Out", &float_tensor);
    auto& sub_arg = graph.GetOrCreateNodeArg("SubV2Out", &float_tensor);
    auto& div_arg = graph.GetOrCreateNodeArg("RealDivOut", &float_tensor);

    graph.AddNode("addv2_node", "AddV2", "TF AddV2 compatibility",
                  {graph_input_defs[0], graph_input_defs[1]}, {&add_arg});
    graph.AddNode("subv2_node", "SubV2", "TF SubV2 compatibility",
                  {&add_arg, graph_input_defs[2]}, {&sub_arg});
    graph.AddNode("realdiv_node", "RealDiv", "TF RealDiv compatibility",
                  {&sub_arg, graph_input_defs[3]}, {&div_arg});
    graph.AddNode("addn_node", "AddN", "TF AddN compatibility",
                  {&div_arg, graph_input_defs[4], graph_input_defs[5]}, {graph_output_defs[0]});
  }
};

class BiasAddAddChainTester : public CompareOpTester {
 public:
  BiasAddAddChainTester() : CompareOpTester("BiasAddAddChain", 17) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto float_tensor = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2, 3});
    auto& bias_arg = graph.GetOrCreateNodeArg("BiasAddOut", &float_tensor);

    graph.AddNode("biasadd_node", "BiasAdd", "TF BiasAdd compatibility",
                  {graph_input_defs[0], graph_input_defs[1]}, {&bias_arg});
    graph.AddNode("add_node", "Add", "Combine BiasAdd output",
                  {&bias_arg, graph_input_defs[2]}, {graph_output_defs[0]});
  }
};

class SquareRsqrtSelectTester : public CompareOpTester {
 public:
  SquareRsqrtSelectTester() : CompareOpTester("SquareRsqrtSelect", 17) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto float_tensor = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {4});
    auto bool_tensor = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_BOOL, {4});
    auto& square_arg = graph.GetOrCreateNodeArg("SquareOut", &float_tensor);
    auto& rsqrt_arg = graph.GetOrCreateNodeArg("RsqrtOut", &float_tensor);
    auto& cond_arg = graph.GetOrCreateNodeArg("Cond", &bool_tensor);
    auto& select_arg = graph.GetOrCreateNodeArg("SelectOut", &float_tensor);

    graph.AddNode("square_node", "Square", "TF Square compatibility",
                  {graph_input_defs[0]}, {&square_arg});
    graph.AddNode("rsqrt_node", "Rsqrt", "TF Rsqrt compatibility",
                  {graph_input_defs[0]}, {&rsqrt_arg});
    graph.AddNode("greater_node", "Greater", "Build Select condition",
                  {graph_input_defs[0], graph_input_defs[1]}, {&cond_arg});
    graph.AddNode("select_node", "Select", "TF Select compatibility",
                  {&cond_arg, &square_arg, &rsqrt_arg}, {&select_arg});
    graph.AddNode("selectv2_node", "SelectV2", "TF SelectV2 compatibility",
                  {&cond_arg, &select_arg, &square_arg}, {graph_output_defs[0]});
  }
};

class DivNoNanSquaredDifferenceTester : public CompareOpTester {
 public:
  DivNoNanSquaredDifferenceTester() : CompareOpTester("DivNoNanSquaredDifference", 17) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto float_tensor = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2, 2});
    auto& div_arg = graph.GetOrCreateNodeArg("DivNoNanOut", &float_tensor);

    graph.AddNode("div_no_nan_node", "DivNoNan", "TF DivNoNan compatibility",
                  {graph_input_defs[0], graph_input_defs[1]}, {&div_arg});
    graph.AddNode("squared_difference_node", "SquaredDifference", "TF SquaredDifference compatibility",
                  {&div_arg, graph_input_defs[2]}, {graph_output_defs[0]});
  }
};

class FloorDivModTester : public CompareOpTester {
 public:
  FloorDivModTester() : CompareOpTester("FloorDivMod", 17) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto float_tensor = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {4});
    auto& div_arg = graph.GetOrCreateNodeArg("FloorDivOut", &float_tensor);
    auto& mod_arg = graph.GetOrCreateNodeArg("FloorModOut", &float_tensor);

    graph.AddNode("floor_div_node", "FloorDiv", "TF FloorDiv compatibility",
                  {graph_input_defs[0], graph_input_defs[1]}, {&div_arg});
    graph.AddNode("floor_mod_node", "FloorMod", "TF FloorMod compatibility",
                  {graph_input_defs[0], graph_input_defs[1]}, {&mod_arg});
    graph.AddNode("add_node", "Add", "Combine FloorDiv and FloorMod",
                  {&div_arg, &mod_arg}, {graph_output_defs[0]});
  }
};

class ZerosLikeAddTester : public CompareOpTester {
 public:
  ZerosLikeAddTester() : CompareOpTester("ZerosLikeAdd", 17) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto float_tensor = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2, 3});
    auto& zeros_arg = graph.GetOrCreateNodeArg("Zeros", &float_tensor);

    graph.AddNode("zeros_like_node", "ZerosLike", "TF ZerosLike compatibility",
                  {graph_input_defs[0]}, {&zeros_arg});
    graph.AddNode("add_node", "Add", "Add zeros to bias",
                  {&zeros_arg, graph_input_defs[1]}, {graph_output_defs[0]});
  }
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

class ConcatV2SplitVChainTester : public CompareOpTester {
 public:
  ConcatV2SplitVChainTester() : CompareOpTester("ConcatV2SplitVChain", 17) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto concat_type = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2, 4});
    auto split0_type = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2, 1});
    auto split1_type = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2, 3});
    auto& concat_arg = graph.GetOrCreateNodeArg("ConcatV2Out", &concat_type);
    auto& split0_arg = graph.GetOrCreateNodeArg("SplitVOut0", &split0_type);
    auto& split1_arg = graph.GetOrCreateNodeArg("SplitVOut1", &split1_type);

    graph.AddNode("concatv2_node", "ConcatV2", "TF ConcatV2 compatibility",
                  {graph_input_defs[0], graph_input_defs[1], graph_input_defs[2]}, {&concat_arg});
    graph.AddNode("splitv_node", "SplitV", "TF SplitV compatibility",
                  {&concat_arg, graph_input_defs[3], graph_input_defs[4]}, {&split0_arg, &split1_arg});
    graph.AddNode("restore_concatv2_node", "ConcatV2", "Restore split pieces",
                  {&split0_arg, &split1_arg, graph_input_defs[2]}, {graph_output_defs[0]});
  }
};

class ReverseV2InvertPermutationChainTester : public CompareOpTester {
 public:
  ReverseV2InvertPermutationChainTester() : CompareOpTester("ReverseV2InvertPermutationChain", 17) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto axes_type = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_INT64, {2});
    auto& axes_arg = graph.GetOrCreateNodeArg("InvertedAxes", &axes_type);

    graph.AddNode("invert_permutation_node", "InvertPermutation", "TF InvertPermutation compatibility",
                  {graph_input_defs[1]}, {&axes_arg});
    graph.AddNode("reversev2_node", "ReverseV2", "TF ReverseV2 compatibility",
                  {graph_input_defs[0], &axes_arg}, {graph_output_defs[0]});
  }
};

class LogicalChainTester : public CompareOpTester {
 public:
  LogicalChainTester() : CompareOpTester("LogicalChain", 13) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto bool_tensor = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_BOOL, {2, 3});
    auto& and_arg = graph.GetOrCreateNodeArg("AndOut", &bool_tensor);
    auto& or_arg = graph.GetOrCreateNodeArg("OrOut", &bool_tensor);

    graph.AddNode("and_node", "And", "LogicalAnd node",
                  {graph_input_defs[0], graph_input_defs[1]}, {&and_arg});
    graph.AddNode("or_node", "Or", "LogicalOr node",
                  {&and_arg, graph_input_defs[2]}, {&or_arg});
    graph.AddNode("not_node", "Not", "LogicalNot node", {&or_arg}, {graph_output_defs[0]});
  }
};

class FillAddChainTester : public CompareOpTester {
 public:
  FillAddChainTester() : CompareOpTester("FillAddChain", 17) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto float_tensor = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2, 3});
    auto& fill_arg = graph.GetOrCreateNodeArg("FillOut", &float_tensor);

    graph.AddNode("fill_node", "Fill", "TF Fill compatibility",
                  {graph_input_defs[0], graph_input_defs[1]}, {&fill_arg});
    graph.AddNode("add_node", "Add", "Combine Fill output",
                  {&fill_arg, graph_input_defs[2]}, {graph_output_defs[0]});
  }
};

class GatherV2AddChainTester : public CompareOpTester {
 public:
  GatherV2AddChainTester() : CompareOpTester("GatherV2AddChain", 17) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto float_tensor = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2, 2});
    auto& gather_arg = graph.GetOrCreateNodeArg("GatherV2Out", &float_tensor);

    graph.AddNode("gatherv2_node", "GatherV2", "TF GatherV2 compatibility",
                  {graph_input_defs[0], graph_input_defs[1], graph_input_defs[2]}, {&gather_arg});
    graph.AddNode("add_node", "Add", "Combine GatherV2 output",
                  {&gather_arg, graph_input_defs[3]}, {graph_output_defs[0]});
  }
};

class PadV2AddChainTester : public CompareOpTester {
 public:
  PadV2AddChainTester() : CompareOpTester("PadV2AddChain", 17) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto float_tensor = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {3, 3});
    auto& padded_arg = graph.GetOrCreateNodeArg("Padded", &float_tensor);

    graph.AddNode("padv2_node", "PadV2", "TF PadV2 compatibility",
                  {graph_input_defs[0], graph_input_defs[1], graph_input_defs[2]}, {&padded_arg});
    graph.AddNode("add_node", "Add", "Combine PadV2 output",
                  {&padded_arg, graph_input_defs[3]}, {graph_output_defs[0]});
  }
};

class ExpandDimsIdentityChainTester : public CompareOpTester {
 public:
  ExpandDimsIdentityChainTester() : CompareOpTester("ExpandDimsIdentityChain", 17) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto expanded_type = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2, 1, 2});
    auto& expanded_arg = graph.GetOrCreateNodeArg("Expanded", &expanded_type);
    auto& stop_arg = graph.GetOrCreateNodeArg("StopGradientOut", &expanded_type);

    graph.AddNode("expand_dims_node", "ExpandDims", "TF ExpandDims compatibility",
                  {graph_input_defs[0], graph_input_defs[1]}, {&expanded_arg});
    graph.AddNode("stop_gradient_node", "StopGradient", "TF StopGradient compatibility",
                  {&expanded_arg}, {&stop_arg});
    graph.AddNode("snapshot_node", "Snapshot", "TF Snapshot compatibility",
                  {&stop_arg}, {graph_output_defs[0]});
  }
};

class TfArrayLogicalAliasChainTester : public CompareOpTester {
 public:
  TfArrayLogicalAliasChainTester() : CompareOpTester("TfArrayLogicalAliasChain", 17) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto float_tensor = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2, 2});
    auto bool_tensor = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_BOOL, {2, 2});
    auto& broadcast_arg = graph.GetOrCreateNodeArg("BroadcastToOut", &float_tensor);
    auto& isnan_arg = graph.GetOrCreateNodeArg("IsNanOut", &bool_tensor);
    auto& not_arg = graph.GetOrCreateNodeArg("LogicalNotOut", &bool_tensor);
    auto& and_arg = graph.GetOrCreateNodeArg("LogicalAndOut", &bool_tensor);

    graph.AddNode("broadcast_to_node", "BroadcastTo", "TF BroadcastTo compatibility",
                  {graph_input_defs[0], graph_input_defs[1]}, {&broadcast_arg});
    graph.AddNode("isnan_node", "IsNan", "TF IsNan compatibility",
                  {&broadcast_arg}, {&isnan_arg});
    graph.AddNode("logical_not_node", "LogicalNot", "TF LogicalNot compatibility",
                  {&isnan_arg}, {&not_arg});
    graph.AddNode("logical_and_node", "LogicalAnd", "TF LogicalAnd compatibility",
                  {&not_arg, graph_input_defs[2]}, {&and_arg});
    graph.AddNode("logical_or_node", "LogicalOr", "TF LogicalOr compatibility",
                  {&and_arg, graph_input_defs[3]}, {graph_output_defs[0]});
  }
};

class BitwiseExtraChainTester : public CompareOpTester {
 public:
  BitwiseExtraChainTester() : CompareOpTester("BitwiseExtraChain", 18) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto int32_tensor = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_INT32, {4});
    auto& or_arg = graph.GetOrCreateNodeArg("OrOut", &int32_tensor);
    auto& xor_arg = graph.GetOrCreateNodeArg("XorOut", &int32_tensor);

    graph.AddNode("or_node", "BitwiseOr", "BitwiseOr node",
                  {graph_input_defs[0], graph_input_defs[1]}, {&or_arg});
    graph.AddNode("xor_node", "BitwiseXor", "BitwiseXor node",
                  {&or_arg, graph_input_defs[2]}, {&xor_arg});
    graph.AddNode("not_node", "BitwiseNot", "BitwiseNot node",
                  {&xor_arg}, {graph_output_defs[0]});
  }
};

class BitwiseCompatChainTester : public CompareOpTester {
 public:
  BitwiseCompatChainTester() : CompareOpTester("BitwiseCompatChain", 17) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto int32_tensor = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_INT32, {4});
    auto& and_arg = graph.GetOrCreateNodeArg("AndOut", &int32_tensor);
    auto& or_arg = graph.GetOrCreateNodeArg("OrOut", &int32_tensor);
    auto& xor_arg = graph.GetOrCreateNodeArg("XorOut", &int32_tensor);

    graph.AddNode("and_node", "BitwiseAnd", "Opset17 BitwiseAnd node",
                  {graph_input_defs[0], graph_input_defs[1]}, {&and_arg});
    graph.AddNode("or_node", "BitwiseOr", "Opset17 BitwiseOr node", {&and_arg, graph_input_defs[2]}, {&or_arg});
    graph.AddNode("xor_node", "BitwiseXor", "Opset17 BitwiseXor node", {&or_arg, graph_input_defs[1]}, {&xor_arg});
    graph.AddNode("not_node", "BitwiseNot", "Opset17 BitwiseNot node", {&xor_arg}, {graph_output_defs[0]});
  }
};


class ShapeStringCastChainTester : public CompareOpTester {
 public:
  ShapeStringCastChainTester() : CompareOpTester("ShapeStringCastChain", 17) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto bool_2x3_type = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_BOOL, {2, 3});
    auto bool_3x2_type = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_BOOL, {3, 2});
    auto string_flat_type = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_STRING, {3});

    auto& less_arg = graph.GetOrCreateNodeArg("LessOut", &bool_2x3_type);
    auto& reshaped_bool_arg = graph.GetOrCreateNodeArg("ReshapedBool", &bool_3x2_type);
    auto& string_flat_arg = graph.GetOrCreateNodeArg("StringFlat", &string_flat_type);

    graph.AddNode("less_node", "Less", "Less before bool reshape",
                  {graph_input_defs[0], graph_input_defs[1]}, {&less_arg});
    graph.AddNode("reshape_bool_node", "Reshape", "Bool reshape",
                  {&less_arg, graph_input_defs[2]}, {&reshaped_bool_arg});
    graph.AddNode("reshape_string_node", "Reshape", "CPU string reshape",
                  {graph_input_defs[3], graph_input_defs[4]}, {&string_flat_arg});
    auto& cast_node = graph.AddNode("cast_string_node", "Cast", "String to int32 cast",
                                    {&string_flat_arg}, {graph_output_defs[1]});
    cast_node.AddAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType_INT32});
    graph.AddNode("identity_bool_node", "Identity", "Expose reshaped bool output",
                  {&reshaped_bool_arg}, {graph_output_defs[0]});
  }
};

class IdentityNShapeNChainTester : public CompareOpTester {
 public:
  IdentityNShapeNChainTester() : CompareOpTester("IdentityNShapeNChain", 17) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto float_tensor = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2, 2});
    auto bool_tensor = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_BOOL, {3});
    auto& identity0_arg = graph.GetOrCreateNodeArg("IdentityNOut0", &float_tensor);
    auto& identity1_arg = graph.GetOrCreateNodeArg("IdentityNOut1", &bool_tensor);

    graph.AddNode("identity_n_node", "IdentityN", "TF IdentityN compatibility",
                  {graph_input_defs[0], graph_input_defs[1]}, {&identity0_arg, &identity1_arg});
    auto& shape_n_node = graph.AddNode("shape_n_node", "ShapeN", "TF ShapeN compatibility",
                                       {&identity0_arg, &identity1_arg},
                                       {graph_output_defs[0], graph_output_defs[1]});
    shape_n_node.AddAttribute("out_type", int64_t{ONNX_NAMESPACE::TensorProto_DataType_INT32});
  }
};


class BroadcastGradientArgsConcatV2Tester : public CompareOpTester {
 public:
  BroadcastGradientArgsConcatV2Tester() : CompareOpTester("BroadcastGradientArgsConcatV2", 17) {}

  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    (void)add_attribute_funcs;

    auto int64_axis_tensor = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_INT64, {1});
    auto& r0_arg = graph.GetOrCreateNodeArg("BroadcastGradR0", &int64_axis_tensor);
    auto& r1_arg = graph.GetOrCreateNodeArg("BroadcastGradR1", &int64_axis_tensor);

    graph.AddNode("broadcast_gradient_args_node", "BroadcastGradientArgs", "TF BroadcastGradientArgs compatibility",
                  {graph_input_defs[0], graph_input_defs[1]}, {&r0_arg, &r1_arg});
    graph.AddNode("concatv2_node", "ConcatV2", "Concatenate reduction axes",
                  {&r0_arg, &r1_arg, graph_input_defs[2]}, {graph_output_defs[0]});
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

TEST(MusaComparisonElementwiseTest, GreaterEqualFloatBroadcastDynamicShape) {
  CompareOpTester test("GreaterEqual", 17);
  const std::vector<std::string> lhs_dim_params{"batch", "1"};
  const std::vector<std::string> rhs_dim_params{"1", "seq"};
  test.AddInput<float>("A", {2, 1}, {1.0f, 4.0f}, false, &lhs_dim_params);
  test.AddInput<float>("B", {1, 3}, {1.0f, 2.0f, 5.0f}, false, &rhs_dim_params);
  test.AddOutput<bool>("C", {2, 3}, {true, false, false, true, true, false});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, LessEqualFloat16NoCpuFallback) {
  CompareOpTester test("LessEqual", 17);
  test.AddInput<MLFloat16>("A", {4}, ToFloat16({1.0f, 4.0f, 2.0f, -1.0f}));
  test.AddInput<MLFloat16>("B", {4}, ToFloat16({1.0f, 3.0f, 3.0f, -2.0f}));
  test.AddOutput<bool>("C", {4}, {true, false, true, false});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, GreaterEqualInt32Scalar) {
  CompareOpTester test("GreaterEqual", 17);
  test.AddInput<int32_t>("A", {4}, {1, 4, 2, -1});
  test.AddInput<int32_t>("B", {}, {2});
  test.AddOutput<bool>("C", {4}, {false, true, true, false});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, LessEqualInt64Empty) {
  CompareOpTester test("LessEqual", 17);
  const std::vector<int64_t> empty;
  test.AddInput<int64_t>("A", {0}, empty);
  test.AddInput<int64_t>("B", {0}, empty);
  test.AddOutput<bool>("C", {0}, {});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, NotEqualBoolBroadcastDynamicShape) {
  CompareOpTester test("NotEqual", 17);
  const std::vector<std::string> lhs_dim_params{"batch", "1"};
  const std::vector<std::string> rhs_dim_params{"1", "seq"};
  test.AddInput<bool>("A", {2, 1}, {true, false}, false, &lhs_dim_params);
  test.AddInput<bool>("B", {1, 3}, {true, false, true}, false, &rhs_dim_params);
  test.AddOutput<bool>("C", {2, 3}, {false, true, false, true, false, true});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, NotEqualFloat16NoCpuFallback) {
  CompareOpTester test("NotEqual", 17);
  test.AddInput<MLFloat16>("A", {4}, ToFloat16({1.0f, 4.0f, 2.0f, -1.0f}));
  test.AddInput<MLFloat16>("B", {4}, ToFloat16({1.0f, 3.0f, 2.0f, -2.0f}));
  test.AddOutput<bool>("C", {4}, {false, true, false, true});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, NotEqualInt32Scalar) {
  CompareOpTester test("NotEqual", 17);
  test.AddInput<int32_t>("A", {4}, {1, 4, 2, -1});
  test.AddInput<int32_t>("B", {}, {2});
  test.AddOutput<bool>("C", {4}, {true, true, false, true});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, GreaterLessNotEqualCompatMultiOpGraph) {
  CompatComparisonChainTester test;
  test.AddInput<float>("A", {2, 3}, {1.0f, 4.0f, 2.0f, 8.0f, 5.0f, 0.0f});
  test.AddInput<float>("B", {2, 3}, {1.0f, 3.0f, 3.0f, 7.0f, 6.0f, -1.0f});
  test.AddInput<float>("X", {2, 3}, {2.0f, 4.0f, 2.0f, 8.0f, 5.0f, 0.0f});
  test.AddInput<float>("Y", {2, 3}, {3.0f, 3.0f, 2.0f, 9.0f, 6.0f, -1.0f});
  test.AddOutput<bool>("Z", {2, 3}, {false, true, true, false, true, true});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, AndBoolBroadcastDynamicShape) {
  CompareOpTester test("And", 7);
  const std::vector<std::string> lhs_dim_params{"batch", "1"};
  const std::vector<std::string> rhs_dim_params{"1", "seq"};
  test.AddInput<bool>("A", {2, 1}, {true, false}, false, &lhs_dim_params);
  test.AddInput<bool>("B", {1, 3}, {true, false, true}, false, &rhs_dim_params);
  test.AddOutput<bool>("C", {2, 3}, {true, false, true, false, false, false});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, OrBoolScalar) {
  CompareOpTester test("Or", 7);
  test.AddInput<bool>("A", {4}, {false, false, true, true});
  test.AddInput<bool>("B", {}, {true});
  test.AddOutput<bool>("C", {4}, {true, true, true, true});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, XorBoolEmpty) {
  CompareOpTester test("Xor", 7);
  test.AddInput<bool>("A", {0}, {});
  test.AddInput<bool>("B", {0}, {});
  test.AddOutput<bool>("C", {0}, {});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, LogicalChainNoCpuFallback) {
  LogicalChainTester test;
  test.AddInput<bool>("A", {2, 3}, {true, false, true, false, true, false});
  test.AddInput<bool>("B", {2, 3}, {true, true, false, false, true, true});
  test.AddInput<bool>("C", {2, 3}, {false, false, true, true, false, true});
  test.AddOutput<bool>("Y", {2, 3}, {false, true, false, false, false, false});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, PadV2Float2D) {
  CompareOpTester test("PadV2", 17);
  test.AddInput<float>("X", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  test.AddInput<int64_t>("Pads", {4}, {0, 1, 1, 0}, true);
  test.AddInput<float>("Value", {}, {9.0f}, true);
  test.AddOutput<float>("Y", {3, 3}, {9.0f, 1.0f, 2.0f, 9.0f, 3.0f, 4.0f, 9.0f, 9.0f, 9.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, PadV2Float16NoCpuFallback) {
  CompareOpTester test("PadV2", 17);
  test.AddInput<MLFloat16>("X", {2, 2}, ToFloat16({1.0f, 2.0f, 3.0f, 4.0f}));
  test.AddInput<int64_t>("Pads", {4}, {0, 1, 1, 0}, true);
  test.AddInput<MLFloat16>("Value", {}, ToFloat16({0.0f}), true);
  test.AddOutput<MLFloat16>("Y", {3, 3}, ToFloat16({0.0f, 1.0f, 2.0f, 0.0f, 3.0f, 4.0f, 0.0f, 0.0f, 0.0f}));
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, PadV2FloatEmpty) {
  CompareOpTester test("PadV2", 17);
  test.AddInput<float>("X", {0}, {});
  test.AddInput<int64_t>("Pads", {2}, {0, 0}, true);
  test.AddInput<float>("Value", {}, {0.0f}, true);
  test.AddOutput<float>("Y", {0}, {});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, PadV2AddChainNoCpuFallback) {
  PadV2AddChainTester test;
  test.AddInput<float>("X", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  test.AddInput<int64_t>("Pads", {4}, {0, 1, 1, 0}, true);
  test.AddInput<float>("Value", {}, {1.0f}, true);
  test.AddInput<float>("Bias", {3, 3}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
  test.AddOutput<float>("Y", {3, 3}, {2.0f, 2.0f, 3.0f, 2.0f, 4.0f, 5.0f, 2.0f, 2.0f, 2.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, ExpandDimsFloatDynamicShape) {
  CompareOpTester test("ExpandDims", 17);
  const std::vector<std::string> input_dim_params{"batch", "seq"};
  test.AddInput<float>("X", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}, false, &input_dim_params);
  test.AddInput<int64_t>("Dim", {}, {1}, true);
  test.AddOutput<float>("Y", {2, 1, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, ExpandDimsFloat16Scalar) {
  CompareOpTester test("ExpandDims", 17);
  test.AddInput<MLFloat16>("X", {}, ToFloat16({1.5f}));
  test.AddInput<int64_t>("Dim", {}, {0}, true);
  test.AddOutput<MLFloat16>("Y", {1}, ToFloat16({1.5f}));
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, ExpandDimsInt32Empty) {
  CompareOpTester test("ExpandDims", 17);
  test.AddInput<int32_t>("X", {0, 2}, {});
  test.AddInput<int64_t>("Dim", {}, {1}, true);
  test.AddOutput<int32_t>("Y", {0, 1, 2}, {});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, StopGradientInt64Scalar) {
  CompareOpTester test("StopGradient", 17);
  test.AddInput<int64_t>("X", {}, {42});
  test.AddOutput<int64_t>("Y", {}, {42});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, SnapshotBoolVector) {
  CompareOpTester test("Snapshot", 17);
  test.AddInput<bool>("X", {4}, {true, false, true, false});
  test.AddOutput<bool>("Y", {4}, {true, false, true, false});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, ExpandDimsIdentityChainNoCpuFallback) {
  ExpandDimsIdentityChainTester test;
  test.AddInput<float>("X", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  test.AddInput<int64_t>("Dim", {}, {1}, true);
  test.AddOutput<float>("Y", {2, 1, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, BroadcastToFloatBroadcastDynamicShape) {
  CompareOpTester test("BroadcastTo", 17);
  const std::vector<std::string> input_dim_params{"batch", "1"};
  test.AddInput<float>("X", {2, 1}, {1.0f, -2.0f}, false, &input_dim_params);
  test.AddInput<int64_t>("Shape", {2}, {2, 3}, true);
  test.AddOutput<float>("Y", {2, 3}, {1.0f, 1.0f, 1.0f, -2.0f, -2.0f, -2.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, BroadcastToFloat16Scalar) {
  CompareOpTester test("BroadcastTo", 17);
  test.AddInput<MLFloat16>("X", {}, ToFloat16({2.5f}));
  test.AddInput<int64_t>("Shape", {1}, {3}, true);
  test.AddOutput<MLFloat16>("Y", {3}, ToFloat16({2.5f, 2.5f, 2.5f}));
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, BroadcastToInt64Empty) {
  CompareOpTester test("BroadcastTo", 17);
  test.AddInput<int64_t>("X", {0}, {});
  test.AddInput<int64_t>("Shape", {1}, {0}, true);
  test.AddOutput<int64_t>("Y", {0}, {});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, GatherNdFloatSlices) {
  CompareOpTester test("GatherNd", 17);
  test.AddInput<float>("Params", {3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddInput<int64_t>("Indices", {2, 1}, {0, 2}, true);
  test.AddOutput<float>("Y", {2, 2}, {1.0f, 2.0f, 5.0f, 6.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, GatherNdInt32ScalarSlices) {
  CompareOpTester test("GatherNd", 17);
  test.AddInput<int32_t>("Params", {2, 2}, {1, 2, 3, 4});
  test.AddInput<int64_t>("Indices", {2, 2}, {0, 1, 1, 0}, true);
  test.AddOutput<int32_t>("Y", {2}, {2, 3});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, GatherNdBoolEmpty) {
  CompareOpTester test("GatherNd", 17);
  test.AddInput<bool>("Params", {0, 2}, {});
  test.AddInput<int64_t>("Indices", {0, 1}, {}, true);
  test.AddOutput<bool>("Y", {0, 2}, {});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, FillFloatRuntimeDims) {
  CompareOpTester test("Fill", 17);
  const std::vector<std::string> dims_dim_params{"seq"};
  test.AddInput<int64_t>("dims", {2}, {2, 3}, false, &dims_dim_params);
  test.AddInput<float>("value", {}, {1.5f}, true);
  test.AddOutput<float>("output", {2, 3}, {1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, FillFloat16ScalarOutput) {
  CompareOpTester test("Fill", 17);
  test.AddInput<int64_t>("dims", {0}, {}, true);
  test.AddInput<MLFloat16>("value", {}, ToFloat16({2.5f}), true);
  test.AddOutput<MLFloat16>("output", {}, ToFloat16({2.5f}));
  CompareWithMusaNoFallback(test, true, 1e-3, 1e-3);
}

TEST(MusaComparisonElementwiseTest, FillInt32ScalarDims) {
  CompareOpTester test("Fill", 17);
  test.AddInput<int32_t>("dims", {}, {4}, true);
  test.AddInput<int32_t>("value", {}, {-7}, true);
  test.AddOutput<int32_t>("output", {4}, {-7, -7, -7, -7});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, FillInt64Empty) {
  CompareOpTester test("Fill", 17);
  const std::vector<int64_t> empty;
  test.AddInput<int64_t>("dims", {2}, {0, 3}, true);
  test.AddInput<int64_t>("value", {}, {9}, true);
  test.AddOutput<int64_t>("output", {0, 3}, empty);
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, FillBoolNoCpuFallback) {
  CompareOpTester test("Fill", 17);
  test.AddInput<int64_t>("dims", {1}, {3}, true);
  test.AddInput<bool>("value", {}, {true}, true);
  test.AddOutput<bool>("output", {3}, {true, true, true});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, FillAddChainNoCpuFallback) {
  FillAddChainTester test;
  test.AddInput<int64_t>("dims", {2}, {2, 3}, true);
  test.AddInput<float>("value", {}, {2.0f}, true);
  test.AddInput<float>("extra", {}, {0.5f}, true);
  test.AddOutput<float>("output", {2, 3}, {2.5f, 2.5f, 2.5f, 2.5f, 2.5f, 2.5f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, GatherV2FloatAxis1DynamicShape) {
  CompareOpTester test("GatherV2", 17);
  const std::vector<std::string> params_dim_params{"batch", "seq"};
  test.AddInput<float>("params", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, false, &params_dim_params);
  test.AddInput<int64_t>("indices", {2}, {2, 0}, true);
  test.AddInput<int64_t>("axis", {}, {1}, true);
  test.AddOutput<float>("output", {2, 2}, {3.0f, 1.0f, 6.0f, 4.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, GatherV2Float16ScalarIndex) {
  CompareOpTester test("GatherV2", 17);
  test.AddInput<MLFloat16>("params", {3}, ToFloat16({1.0f, 2.0f, 3.0f}));
  test.AddInput<int64_t>("indices", {}, {2}, true);
  test.AddInput<int64_t>("axis", {}, {0}, true);
  test.AddOutput<MLFloat16>("output", {}, ToFloat16({3.0f}));
  CompareWithMusaNoFallback(test, true, 1e-3, 1e-3);
}

TEST(MusaComparisonElementwiseTest, GatherV2Int32NegativeAxis) {
  CompareOpTester test("GatherV2", 17);
  test.AddInput<int32_t>("params", {2, 3}, {1, 2, 3, 4, 5, 6});
  test.AddInput<int32_t>("indices", {2}, {1, 0}, true);
  test.AddInput<int32_t>("axis", {}, {-1}, true);
  test.AddOutput<int32_t>("output", {2, 2}, {2, 1, 5, 4});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, GatherV2Int64Empty) {
  CompareOpTester test("GatherV2", 17);
  const std::vector<int64_t> empty;
  test.AddInput<int64_t>("params", {0, 2}, empty);
  test.AddInput<int64_t>("indices", {0}, empty, true);
  test.AddInput<int64_t>("axis", {}, {0}, true);
  test.AddOutput<int64_t>("output", {0, 2}, empty);
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, GatherV2FloatBatchDims) {
  CompareOpTester test("GatherV2", 17);
  test.AddAttribute("batch_dims", int64_t{1});
  test.AddInput<float>("params", {2, 3}, {10.0f, 11.0f, 12.0f, 20.0f, 21.0f, 22.0f});
  test.AddInput<int32_t>("indices", {2, 2}, {2, 0, 1, 1}, true);
  test.AddInput<int64_t>("axis", {}, {1}, true);
  test.AddOutput<float>("output", {2, 2}, {12.0f, 10.0f, 21.0f, 21.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, GatherV2AddChainNoCpuFallback) {
  GatherV2AddChainTester test;
  test.AddInput<float>("params", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddInput<int64_t>("indices", {2}, {2, 0}, true);
  test.AddInput<int64_t>("axis", {}, {1}, true);
  test.AddInput<float>("extra", {}, {1.0f}, true);
  test.AddOutput<float>("output", {2, 2}, {4.0f, 2.0f, 7.0f, 5.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, IsNanFloatDynamicShape) {
  CompareOpTester test("IsNan", 17);
  const std::vector<std::string> dim_params{"seq"};
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  test.AddInput<float>("X", {4}, {0.0f, nan, inf, nan}, false, &dim_params);
  test.AddOutput<bool>("Y", {4}, {false, true, false, true});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, IsNanFloat16NoCpuFallback) {
  CompareOpTester test("IsNan", 17);
  const float nan = std::numeric_limits<float>::quiet_NaN();
  test.AddInput<MLFloat16>("X", {3}, {MLFloat16(0.0f), MLFloat16(nan), MLFloat16(1.0f)});
  test.AddOutput<bool>("Y", {3}, {false, true, false});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, LogicalAndBoolBroadcastDynamicShape) {
  CompareOpTester test("LogicalAnd", 17);
  const std::vector<std::string> lhs_dim_params{"batch", "1"};
  const std::vector<std::string> rhs_dim_params{"1", "seq"};
  test.AddInput<bool>("A", {2, 1}, {true, false}, false, &lhs_dim_params);
  test.AddInput<bool>("B", {1, 3}, {true, false, true}, false, &rhs_dim_params);
  test.AddOutput<bool>("C", {2, 3}, {true, false, true, false, false, false});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, LogicalOrBoolScalar) {
  CompareOpTester test("LogicalOr", 17);
  test.AddInput<bool>("A", {4}, {false, false, true, true});
  test.AddInput<bool>("B", {}, {true});
  test.AddOutput<bool>("C", {4}, {true, true, true, true});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, LogicalNotBoolEmpty) {
  CompareOpTester test("LogicalNot", 17);
  test.AddInput<bool>("X", {0}, {});
  test.AddOutput<bool>("Y", {0}, {});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, TfArrayLogicalAliasChainNoCpuFallback) {
  TfArrayLogicalAliasChainTester test;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  test.AddInput<float>("X", {2, 1}, {1.0f, nan});
  test.AddInput<int64_t>("Shape", {2}, {2, 2}, true);
  test.AddInput<bool>("Mask", {2, 2}, {true, false, true, false});
  test.AddInput<bool>("FallbackMask", {2, 2}, {false, false, false, true});
  test.AddOutput<bool>("Y", {2, 2}, {true, false, false, true});
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

TEST(MusaComparisonElementwiseTest, BitwiseAndOpset17Int32BroadcastDynamicShape) {
  CompareOpTester test("BitwiseAnd", 17);
  const std::vector<std::string> lhs_dim_params{"batch", "1"};
  const std::vector<std::string> rhs_dim_params{"1", "seq"};
  test.AddInput<int32_t>("A", {2, 1}, {7, 12}, false, &lhs_dim_params);
  test.AddInput<int32_t>("B", {1, 3}, {3, 10, 15}, false, &rhs_dim_params);
  test.AddOutput<int32_t>("C", {2, 3}, {3, 2, 7, 0, 8, 12});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, BitwiseAndOpset17Int64Scalar) {
  CompareOpTester test("BitwiseAnd", 17);
  test.AddInput<int64_t>("A", {4}, {1, 2, 3, 4});
  test.AddInput<int64_t>("B", {}, {3});
  test.AddOutput<int64_t>("C", {4}, {1, 2, 3, 0});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, BitwiseOrOpset17UInt16Empty) {
  CompareOpTester test("BitwiseOr", 17);
  const std::vector<uint16_t> empty;
  test.AddInput<uint16_t>("A", {0}, empty);
  test.AddInput<uint16_t>("B", {0}, empty);
  test.AddOutput<uint16_t>("C", {0}, empty);
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, BitwiseNotOpset17UInt8) {
  CompareOpTester test("BitwiseNot", 17);
  test.AddInput<uint8_t>("X", {4}, {0x00, 0x0f, 0xf0, 0xff});
  test.AddOutput<uint8_t>("Y", {4}, {0xff, 0xf0, 0x0f, 0x00});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, BitwiseOrInt32BroadcastDynamicShape) {
  CompareOpTester test("BitwiseOr", 18);
  const std::vector<std::string> lhs_dim_params{"batch", "1"};
  const std::vector<std::string> rhs_dim_params{"1", "seq"};
  test.AddInput<int32_t>("A", {2, 1}, {1, 4}, false, &lhs_dim_params);
  test.AddInput<int32_t>("B", {1, 3}, {2, 5, 8}, false, &rhs_dim_params);
  test.AddOutput<int32_t>("C", {2, 3}, {3, 5, 9, 6, 5, 12});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, BitwiseXorInt64Scalar) {
  CompareOpTester test("BitwiseXor", 18);
  test.AddInput<int64_t>("A", {4}, {1, 2, 3, 4});
  test.AddInput<int64_t>("B", {}, {3});
  test.AddOutput<int64_t>("C", {4}, {2, 1, 0, 7});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, BitwiseNotUInt8) {
  CompareOpTester test("BitwiseNot", 18);
  test.AddInput<uint8_t>("X", {4}, {0x00, 0x0f, 0xf0, 0xff});
  test.AddOutput<uint8_t>("Y", {4}, {0xff, 0xf0, 0x0f, 0x00});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, BitwiseOrUInt16Empty) {
  CompareOpTester test("BitwiseOr", 18);
  const std::vector<uint16_t> empty;
  test.AddInput<uint16_t>("A", {0}, empty);
  test.AddInput<uint16_t>("B", {0}, empty);
  test.AddOutput<uint16_t>("C", {0}, empty);
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, BitwiseExtraChainNoCpuFallback) {
  BitwiseExtraChainTester test;
  test.AddInput<int32_t>("A", {4}, {1, 2, 4, 8});
  test.AddInput<int32_t>("B", {4}, {3, 1, 12, 0});
  test.AddInput<int32_t>("C", {4}, {1, 3, 5, 7});
  test.AddOutput<int32_t>("Y", {4}, {-3, -1, -10, -16});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, BitwiseCompatChainOpset17NoCpuFallback) {
  BitwiseCompatChainTester test;
  test.AddInput<int32_t>("A", {4}, {7, 12, 5, 0});
  test.AddInput<int32_t>("B", {4}, {3, 10, 6, 15});
  test.AddInput<int32_t>("C", {4}, {8, 1, 2, 4});
  test.AddOutput<int32_t>("Y", {4}, {-9, -4, -1, -12});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, ReshapeBoolOpset17NoCpuFallback) {
  CompareOpTester test("Reshape", 17);
  test.AddInput<bool>("data", {2, 3}, {true, false, true, false, false, true});
  test.AddInput<int64_t>("shape", {2}, {3, 2});
  test.AddOutput<bool>("reshaped", {3, 2}, {true, false, true, false, false, true});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, ReshapeStringOpset17NoCpuFallback) {
  CompareOpTester test("Reshape", 17);
  test.AddInput<std::string>("data", {3, 1}, {"10", "20", "-3"});
  test.AddInput<int64_t>("shape", {1}, {3});
  test.AddOutput<std::string>("reshaped", {3}, {"10", "20", "-3"});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, CastStringToInt32Opset17NoCpuFallback) {
  CompareOpTester test("Cast", 17);
  test.AddAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType_INT32});
  test.AddInput<std::string>("input", {4}, {"10", "20", "-3", "0"});
  test.AddOutput<int32_t>("output", {4}, {10, 20, -3, 0});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, CastStringToInt32EmptyOpset17NoCpuFallback) {
  CompareOpTester test("Cast", 17);
  test.AddAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType_INT32});
  const std::vector<std::string> empty_input;
  const std::vector<int32_t> empty_output;
  test.AddInput<std::string>("input", {0}, empty_input);
  test.AddOutput<int32_t>("output", {0}, empty_output);
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, ShapeStringCastChainOpset17NoCpuFallback) {
  ShapeStringCastChainTester test;
  test.AddInput<float>("A", {2, 3}, {1.0f, 4.0f, 2.0f, 8.0f, 5.0f, 0.0f});
  test.AddInput<float>("B", {}, {3.0f});
  test.AddInput<int64_t>("BoolShape", {2}, {3, 2});
  test.AddInput<std::string>("S", {3, 1}, {"10", "20", "-3"});
  test.AddInput<int64_t>("StringShape", {1}, {3});
  test.AddOutput<bool>("BoolY", {3, 2}, {true, false, true, false, false, true});
  test.AddOutput<int32_t>("IntY", {3}, {10, 20, -3});
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

TEST(MusaComparisonElementwiseTest, MaximumFloatBroadcastDynamicShape) {
  CompareOpTester test("Maximum", 17);
  const std::vector<std::string> lhs_dim_params{"batch", "1"};
  const std::vector<std::string> rhs_dim_params{"1", "seq"};
  test.AddInput<float>("A", {2, 1}, {1.0f, 4.0f}, false, &lhs_dim_params);
  test.AddInput<float>("B", {1, 3}, {0.0f, 2.0f, 5.0f}, false, &rhs_dim_params);
  test.AddOutput<float>("C", {2, 3}, {1.0f, 2.0f, 5.0f, 4.0f, 4.0f, 5.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, MaximumFloat16NoCpuFallback) {
  CompareOpTester test("Maximum", 17);
  test.AddInput<MLFloat16>("A", {4}, ToFloat16({1.0f, 4.0f, 2.0f, -1.0f}));
  test.AddInput<MLFloat16>("B", {4}, ToFloat16({0.5f, 4.5f, 3.0f, -2.0f}));
  test.AddOutput<MLFloat16>("C", {4}, ToFloat16({1.0f, 4.5f, 3.0f, -1.0f}));
  CompareWithMusaNoFallback(test, true, 1e-3, 1e-3);
}

TEST(MusaComparisonElementwiseTest, MaximumInt32Scalar) {
  CompareOpTester test("Maximum", 17);
  test.AddInput<int32_t>("A", {4}, {1, 4, 2, -1});
  test.AddInput<int32_t>("B", {}, {2});
  test.AddOutput<int32_t>("C", {4}, {2, 4, 2, 2});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, MaximumInt64Empty) {
  CompareOpTester test("Maximum", 17);
  const std::vector<int64_t> empty;
  test.AddInput<int64_t>("A", {0}, empty);
  test.AddInput<int64_t>("B", {0}, empty);
  test.AddOutput<int64_t>("C", {0}, empty);
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, MinimumFloatBroadcastDynamicShape) {
  CompareOpTester test("Minimum", 17);
  const std::vector<std::string> lhs_dim_params{"batch", "1"};
  const std::vector<std::string> rhs_dim_params{"1", "seq"};
  test.AddInput<float>("A", {2, 1}, {1.0f, 4.0f}, false, &lhs_dim_params);
  test.AddInput<float>("B", {1, 3}, {0.0f, 2.0f, 5.0f}, false, &rhs_dim_params);
  test.AddOutput<float>("C", {2, 3}, {0.0f, 1.0f, 1.0f, 0.0f, 2.0f, 4.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, MinimumFloat16NoCpuFallback) {
  CompareOpTester test("Minimum", 17);
  test.AddInput<MLFloat16>("A", {4}, ToFloat16({1.0f, 4.0f, 2.0f, -1.0f}));
  test.AddInput<MLFloat16>("B", {4}, ToFloat16({0.5f, 4.5f, 3.0f, -2.0f}));
  test.AddOutput<MLFloat16>("C", {4}, ToFloat16({0.5f, 4.0f, 2.0f, -2.0f}));
  CompareWithMusaNoFallback(test, true, 1e-3, 1e-3);
}

TEST(MusaComparisonElementwiseTest, MinimumInt32Scalar) {
  CompareOpTester test("Minimum", 17);
  test.AddInput<int32_t>("A", {4}, {1, 4, 2, -1});
  test.AddInput<int32_t>("B", {}, {2});
  test.AddOutput<int32_t>("C", {4}, {1, 2, 2, -1});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, MinimumInt64Empty) {
  CompareOpTester test("Minimum", 17);
  const std::vector<int64_t> empty;
  test.AddInput<int64_t>("A", {0}, empty);
  test.AddInput<int64_t>("B", {0}, empty);
  test.AddOutput<int64_t>("C", {0}, empty);
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, MaximumMinimumAddMultiOpGraph) {
  MaximumMinimumAddTester test;
  test.AddInput<float>("A", {2, 3}, {1.0f, 4.0f, 2.0f, 8.0f, 5.0f, 0.0f});
  test.AddInput<float>("B", {2, 3}, {2.0f, 3.0f, 2.0f, 7.0f, 6.0f, -1.0f});
  test.AddInput<float>("C", {2, 3}, {1.5f, 2.0f, 3.0f, 9.0f, 4.0f, 1.0f});
  test.AddInput<float>("Bias", {}, {0.5f}, true);
  test.AddOutput<float>("Y", {2, 3}, {2.0f, 2.5f, 2.5f, 8.5f, 4.5f, 0.5f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, AddV2FloatBroadcastDynamicShape) {
  CompareOpTester test("AddV2", 17);
  const std::vector<std::string> lhs_dim_params{"batch", "1"};
  const std::vector<std::string> rhs_dim_params{"1", "seq"};
  test.AddInput<float>("A", {2, 1}, {1.0f, 4.0f}, false, &lhs_dim_params);
  test.AddInput<float>("B", {1, 3}, {0.0f, 2.0f, 5.0f}, false, &rhs_dim_params);
  test.AddOutput<float>("C", {2, 3}, {1.0f, 3.0f, 6.0f, 4.0f, 6.0f, 9.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, AddV2Float16Scalar) {
  CompareOpTester test("AddV2", 17);
  test.AddInput<MLFloat16>("A", {4}, ToFloat16({1.0f, 2.0f, 3.0f, 4.0f}));
  test.AddInput<MLFloat16>("B", {}, ToFloat16({0.5f}));
  test.AddOutput<MLFloat16>("C", {4}, ToFloat16({1.5f, 2.5f, 3.5f, 4.5f}));
  CompareWithMusaNoFallback(test, true, 1e-3, 1e-3);
}

TEST(MusaComparisonElementwiseTest, AddV2Int32Scalar) {
  CompareOpTester test("AddV2", 17);
  test.AddInput<int32_t>("A", {4}, {1, -2, 3, -4});
  test.AddInput<int32_t>("B", {}, {10});
  test.AddOutput<int32_t>("C", {4}, {11, 8, 13, 6});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, AddV2Int64Empty) {
  CompareOpTester test("AddV2", 17);
  const std::vector<int64_t> empty;
  test.AddInput<int64_t>("A", {0}, empty);
  test.AddInput<int64_t>("B", {0}, empty);
  test.AddOutput<int64_t>("C", {0}, empty);
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, BiasAddFloatNHWCBroadcastDynamicShape) {
  CompareOpTester test("BiasAdd", 17);
  const std::vector<std::string> value_dim_params{"batch", "1", "seq"};
  const std::vector<std::string> bias_dim_params{"seq"};
  test.AddInput<float>("value", {2, 1, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, false, &value_dim_params);
  test.AddInput<float>("bias", {3}, {10.0f, 20.0f, 30.0f}, false, &bias_dim_params);
  test.AddOutput<float>("output", {2, 1, 3}, {11.0f, 22.0f, 33.0f, 14.0f, 25.0f, 36.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, BiasAddFloat16ScalarChannel) {
  CompareOpTester test("BiasAdd", 17);
  test.AddInput<MLFloat16>("value", {2, 1}, ToFloat16({1.0f, 2.0f}));
  test.AddInput<MLFloat16>("bias", {1}, ToFloat16({0.5f}));
  test.AddOutput<MLFloat16>("output", {2, 1}, ToFloat16({1.5f, 2.5f}));
  CompareWithMusaNoFallback(test, true, 1e-3, 1e-3);
}

TEST(MusaComparisonElementwiseTest, BiasAddInt32NCHW) {
  CompareOpTester test("BiasAdd", 17);
  test.AddAttribute("data_format", "NCHW");
  test.AddInput<int32_t>("value", {1, 2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
  test.AddInput<int32_t>("bias", {2}, {10, 20});
  test.AddOutput<int32_t>("output", {1, 2, 2, 2}, {11, 12, 13, 14, 25, 26, 27, 28});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, BiasAddV1Int64Empty) {
  CompareOpTester test("BiasAddV1", 17);
  const std::vector<int64_t> empty;
  test.AddInput<int64_t>("value", {0, 3}, empty);
  test.AddInput<int64_t>("bias", {3}, {1, 2, 3});
  test.AddOutput<int64_t>("output", {0, 3}, empty);
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, BiasAddAddChainNoCpuFallback) {
  BiasAddAddChainTester test;
  test.AddInput<float>("value", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddInput<float>("bias", {3}, {10.0f, 20.0f, 30.0f});
  test.AddInput<float>("extra", {}, {1.0f}, true);
  test.AddOutput<float>("output", {2, 3}, {12.0f, 23.0f, 34.0f, 15.0f, 26.0f, 37.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, SubV2FloatBroadcastDynamicShape) {
  CompareOpTester test("SubV2", 17);
  const std::vector<std::string> lhs_dim_params{"batch", "1"};
  const std::vector<std::string> rhs_dim_params{"1", "seq"};
  test.AddInput<float>("A", {2, 1}, {1.0f, 4.0f}, false, &lhs_dim_params);
  test.AddInput<float>("B", {1, 3}, {0.0f, 2.0f, 5.0f}, false, &rhs_dim_params);
  test.AddOutput<float>("C", {2, 3}, {1.0f, -1.0f, -4.0f, 4.0f, 2.0f, -1.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, SubV2Float16Scalar) {
  CompareOpTester test("SubV2", 17);
  test.AddInput<MLFloat16>("A", {4}, ToFloat16({1.0f, 2.0f, 3.0f, 4.0f}));
  test.AddInput<MLFloat16>("B", {}, ToFloat16({0.5f}));
  test.AddOutput<MLFloat16>("C", {4}, ToFloat16({0.5f, 1.5f, 2.5f, 3.5f}));
  CompareWithMusaNoFallback(test, true, 1e-3, 1e-3);
}

TEST(MusaComparisonElementwiseTest, SubV2Int32Scalar) {
  CompareOpTester test("SubV2", 17);
  test.AddInput<int32_t>("A", {4}, {1, -2, 3, -4});
  test.AddInput<int32_t>("B", {}, {2});
  test.AddOutput<int32_t>("C", {4}, {-1, -4, 1, -6});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, SubV2Int64Empty) {
  CompareOpTester test("SubV2", 17);
  const std::vector<int64_t> empty;
  test.AddInput<int64_t>("A", {0}, empty);
  test.AddInput<int64_t>("B", {0}, empty);
  test.AddOutput<int64_t>("C", {0}, empty);
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, RealDivFloatBroadcastDynamicShape) {
  CompareOpTester test("RealDiv", 17);
  const std::vector<std::string> lhs_dim_params{"batch", "seq"};
  test.AddInput<float>("A", {2, 3}, {2.0f, 4.0f, 8.0f, 10.0f, 12.0f, 14.0f}, false, &lhs_dim_params);
  test.AddInput<float>("B", {1, 3}, {2.0f, 4.0f, 2.0f});
  test.AddOutput<float>("C", {2, 3}, {1.0f, 1.0f, 4.0f, 5.0f, 3.0f, 7.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, RealDivFloat16Scalar) {
  CompareOpTester test("RealDiv", 17);
  test.AddInput<MLFloat16>("A", {4}, ToFloat16({2.0f, 4.0f, 8.0f, 16.0f}));
  test.AddInput<MLFloat16>("B", {}, ToFloat16({2.0f}));
  test.AddOutput<MLFloat16>("C", {4}, ToFloat16({1.0f, 2.0f, 4.0f, 8.0f}));
  CompareWithMusaNoFallback(test, true, 1e-3, 1e-3);
}

TEST(MusaComparisonElementwiseTest, RealDivFloatEmpty) {
  CompareOpTester test("RealDiv", 17);
  const std::vector<float> empty;
  test.AddInput<float>("A", {0}, empty);
  test.AddInput<float>("B", {0}, empty);
  test.AddOutput<float>("C", {0}, empty);
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, AddNFloatThreeInputsDynamicShape) {
  CompareOpTester test("AddN", 17);
  const std::vector<std::string> dim_params{"batch", "seq"};
  test.AddInput<float>("A", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}, false, &dim_params);
  test.AddInput<float>("B", {2, 2}, {0.5f, 1.5f, 2.5f, 3.5f});
  test.AddInput<float>("C", {}, {1.0f});
  test.AddOutput<float>("Y", {2, 2}, {2.5f, 4.5f, 6.5f, 8.5f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, AddNFloat16NoCpuFallback) {
  CompareOpTester test("AddN", 17);
  test.AddInput<MLFloat16>("A", {4}, ToFloat16({1.0f, 2.0f, 3.0f, 4.0f}));
  test.AddInput<MLFloat16>("B", {4}, ToFloat16({0.5f, 1.5f, 2.5f, 3.5f}));
  test.AddInput<MLFloat16>("C", {}, ToFloat16({1.0f}));
  test.AddOutput<MLFloat16>("Y", {4}, ToFloat16({2.5f, 4.5f, 6.5f, 8.5f}));
  CompareWithMusaNoFallback(test, true, 1e-3, 1e-3);
}

TEST(MusaComparisonElementwiseTest, AddNInt32Scalar) {
  CompareOpTester test("AddN", 17);
  test.AddInput<int32_t>("A", {}, {1});
  test.AddInput<int32_t>("B", {}, {2});
  test.AddInput<int32_t>("C", {}, {3});
  test.AddOutput<int32_t>("Y", {}, {6});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, AddNInt64Empty) {
  CompareOpTester test("AddN", 17);
  const std::vector<int64_t> empty;
  test.AddInput<int64_t>("A", {0}, empty);
  test.AddInput<int64_t>("B", {0}, empty);
  test.AddInput<int64_t>("C", {0}, empty);
  test.AddOutput<int64_t>("Y", {0}, empty);
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, TfMathAliasMultiOpGraph) {
  TfMathAliasChainTester test;
  test.AddInput<float>("A", {2, 2}, {2.0f, 4.0f, 6.0f, 8.0f});
  test.AddInput<float>("B", {2, 2}, {1.0f, 1.0f, 1.0f, 1.0f});
  test.AddInput<float>("C", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  test.AddInput<float>("D", {}, {2.0f}, true);
  test.AddInput<float>("E", {2, 2}, {0.5f, 0.5f, 0.5f, 0.5f});
  test.AddInput<float>("F", {}, {1.0f}, true);
  test.AddOutput<float>("Y", {2, 2}, {2.5f, 3.0f, 3.5f, 4.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, SquareFloatDynamicShape) {
  CompareOpTester test("Square", 17);
  const std::vector<std::string> dim_params{"batch", "seq"};
  test.AddInput<float>("X", {2, 3}, {1.0f, -2.0f, 3.0f, -4.0f, 0.5f, -0.25f}, false, &dim_params);
  test.AddOutput<float>("Y", {2, 3}, {1.0f, 4.0f, 9.0f, 16.0f, 0.25f, 0.0625f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, SquareFloat16Scalar) {
  CompareOpTester test("Square", 17);
  test.AddInput<MLFloat16>("X", {}, ToFloat16({-3.0f}));
  test.AddOutput<MLFloat16>("Y", {}, ToFloat16({9.0f}));
  CompareWithMusaNoFallback(test, true, 1e-3, 1e-3);
}

TEST(MusaComparisonElementwiseTest, SquareInt32Scalar) {
  CompareOpTester test("Square", 17);
  test.AddInput<int32_t>("X", {}, {-7});
  test.AddOutput<int32_t>("Y", {}, {49});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, SquareInt64Empty) {
  CompareOpTester test("Square", 17);
  const std::vector<int64_t> empty;
  test.AddInput<int64_t>("X", {0}, empty);
  test.AddOutput<int64_t>("Y", {0}, empty);
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, RsqrtFloatDynamicShape) {
  CompareOpTester test("Rsqrt", 17);
  const std::vector<std::string> dim_params{"batch", "seq"};
  test.AddInput<float>("X", {2, 2}, {1.0f, 4.0f, 16.0f, 25.0f}, false, &dim_params);
  test.AddOutput<float>("Y", {2, 2}, {1.0f, 0.5f, 0.25f, 0.2f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, RsqrtFloat16Scalar) {
  CompareOpTester test("Rsqrt", 17);
  test.AddInput<MLFloat16>("X", {}, ToFloat16({4.0f}));
  test.AddOutput<MLFloat16>("Y", {}, ToFloat16({0.5f}));
  CompareWithMusaNoFallback(test, true, 1e-3, 1e-3);
}

TEST(MusaComparisonElementwiseTest, RsqrtFloatEmpty) {
  CompareOpTester test("Rsqrt", 17);
  const std::vector<float> empty;
  test.AddInput<float>("X", {0}, empty);
  test.AddOutput<float>("Y", {0}, empty);
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, SelectFloatBroadcastDynamicShape) {
  CompareOpTester test("Select", 17);
  const std::vector<std::string> cond_dim_params{"batch", "1"};
  const std::vector<std::string> x_dim_params{"1", "seq"};
  test.AddInput<bool>("Cond", {2, 1}, {true, false}, false, &cond_dim_params);
  test.AddInput<float>("X", {1, 3}, {1.0f, 2.0f, 3.0f}, false, &x_dim_params);
  test.AddInput<float>("Y", {}, {9.0f});
  test.AddOutput<float>("Z", {2, 3}, {1.0f, 2.0f, 3.0f, 9.0f, 9.0f, 9.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, SelectV2Float16NoCpuFallback) {
  CompareOpTester test("SelectV2", 17);
  test.AddInput<bool>("Cond", {2, 1}, {true, false});
  test.AddInput<MLFloat16>("X", {2, 3}, ToFloat16({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
  test.AddInput<MLFloat16>("Y", {}, ToFloat16({9.0f}));
  test.AddOutput<MLFloat16>("Z", {2, 3}, ToFloat16({1.0f, 2.0f, 3.0f, 9.0f, 9.0f, 9.0f}));
  CompareWithMusaNoFallback(test, true, 1e-3, 1e-3);
}

TEST(MusaComparisonElementwiseTest, SelectInt32Scalar) {
  CompareOpTester test("Select", 17);
  test.AddInput<bool>("Cond", {1, 3}, {true, false, true});
  test.AddInput<int32_t>("X", {}, {7});
  test.AddInput<int32_t>("Y", {2, 3}, {1, 2, 3, 4, 5, 6});
  test.AddOutput<int32_t>("Z", {2, 3}, {7, 2, 7, 7, 5, 7});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, SelectV2Int64Empty) {
  CompareOpTester test("SelectV2", 17);
  const std::vector<int64_t> value_empty;
  test.AddInput<bool>("Cond", {0}, {});
  test.AddInput<int64_t>("X", {0}, value_empty);
  test.AddInput<int64_t>("Y", {0}, value_empty);
  test.AddOutput<int64_t>("Z", {0}, value_empty);
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, SquareRsqrtSelectMultiOpGraph) {
  SquareRsqrtSelectTester test;
  test.AddInput<float>("X", {4}, {1.0f, 4.0f, 9.0f, 16.0f});
  test.AddInput<float>("Threshold", {}, {5.0f}, true);
  test.AddOutput<float>("Y", {4}, {1.0f, 16.0f, 81.0f, 256.0f});
  CompareWithMusaNoFallback(test, false, 1e-4, 1e-4);
}

TEST(MusaComparisonElementwiseTest, DivNoNanFloatBroadcastDynamicShape) {
  CompareOpTester test("DivNoNan", 17);
  const std::vector<std::string> a_dim_params{"batch", "seq"};
  test.AddInput<float>("A", {2, 3}, {2.0f, 3.0f, 6.0f, -4.0f, 5.0f, 0.0f}, false, &a_dim_params);
  test.AddInput<float>("B", {1, 3}, {2.0f, 0.0f, 3.0f});
  test.AddOutput<float>("C", {2, 3}, {1.0f, 0.0f, 2.0f, -2.0f, 0.0f, 0.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, DivNoNanFloat16NoCpuFallback) {
  CompareOpTester test("DivNoNan", 17);
  test.AddInput<MLFloat16>("A", {4}, ToFloat16({1.0f, 2.0f, -3.0f, 4.0f}));
  test.AddInput<MLFloat16>("B", {4}, ToFloat16({1.0f, 0.0f, -1.0f, 2.0f}));
  test.AddOutput<MLFloat16>("C", {4}, ToFloat16({1.0f, 0.0f, 3.0f, 2.0f}));
  CompareWithMusaNoFallback(test, true, 1e-3, 1e-3);
}

TEST(MusaComparisonElementwiseTest, DivNoNanDoubleScalarNoCpuFallback) {
  CompareOpTester test("DivNoNan", 17);
  test.AddInput<double>("A", {}, {3.0});
  test.AddInput<double>("B", {}, {0.0});
  test.AddOutput<double>("C", {}, {0.0});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, DivNoNanFloatEmpty) {
  CompareOpTester test("DivNoNan", 17);
  test.AddInput<float>("A", {0, 3}, std::vector<float>{});
  test.AddInput<float>("B", {1, 3}, {1.0f, 0.0f, 3.0f});
  test.AddOutput<float>("C", {0, 3}, std::vector<float>{});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, SquaredDifferenceFloatBroadcastDynamicShape) {
  CompareOpTester test("SquaredDifference", 17);
  const std::vector<std::string> a_dim_params{"batch", "seq"};
  test.AddInput<float>("A", {2, 3}, {2.0f, 3.0f, 6.0f, -4.0f, 5.0f, 0.0f}, false, &a_dim_params);
  test.AddInput<float>("B", {1, 3}, {1.0f, 3.0f, 2.0f});
  test.AddOutput<float>("C", {2, 3}, {1.0f, 0.0f, 16.0f, 25.0f, 4.0f, 4.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, SquaredDifferenceFloat16NoCpuFallback) {
  CompareOpTester test("SquaredDifference", 17);
  test.AddInput<MLFloat16>("A", {4}, ToFloat16({1.0f, 2.0f, 4.0f, -1.0f}));
  test.AddInput<MLFloat16>("B", {4}, ToFloat16({0.5f, 2.0f, 1.0f, -3.0f}));
  test.AddOutput<MLFloat16>("C", {4}, ToFloat16({0.25f, 0.0f, 9.0f, 4.0f}));
  CompareWithMusaNoFallback(test, true, 1e-3, 1e-3);
}

TEST(MusaComparisonElementwiseTest, SquaredDifferenceInt32Scalar) {
  CompareOpTester test("SquaredDifference", 17);
  test.AddInput<int32_t>("A", {3}, {1, 5, -3});
  test.AddInput<int32_t>("B", {}, {2});
  test.AddOutput<int32_t>("C", {3}, {1, 9, 25});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, SquaredDifferenceInt64Empty) {
  CompareOpTester test("SquaredDifference", 17);
  test.AddInput<int64_t>("A", {2, 0}, std::vector<int64_t>{});
  test.AddInput<int64_t>("B", {1, 0}, std::vector<int64_t>{});
  test.AddOutput<int64_t>("C", {2, 0}, std::vector<int64_t>{});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, DivNoNanSquaredDifferenceMultiOpGraph) {
  DivNoNanSquaredDifferenceTester test;
  test.AddInput<float>("A", {2, 2}, {2.0f, 0.0f, 6.0f, -4.0f});
  test.AddInput<float>("B", {2, 2}, {2.0f, 0.0f, 3.0f, 2.0f});
  test.AddInput<float>("C", {2, 2}, {1.0f, 0.0f, 1.0f, -2.0f});
  test.AddOutput<float>("Y", {2, 2}, {0.0f, 0.0f, 1.0f, 0.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, FloorDivFloatBroadcastDynamicShape) {
  CompareOpTester test("FloorDiv", 17);
  const std::vector<std::string> a_dim_params{"batch", "seq"};
  test.AddInput<float>("A", {2, 3}, {5.0f, -5.0f, 7.0f, -7.0f, 9.0f, -9.0f}, false, &a_dim_params);
  test.AddInput<float>("B", {1, 3}, {2.0f, -3.0f, 4.0f});
  test.AddOutput<float>("C", {2, 3}, {2.0f, 1.0f, 1.0f, -4.0f, -3.0f, -3.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, FloorDivFloat16NoCpuFallback) {
  CompareOpTester test("FloorDiv", 17);
  test.AddInput<MLFloat16>("A", {4}, ToFloat16({5.0f, -5.0f, 7.0f, -7.0f}));
  test.AddInput<MLFloat16>("B", {4}, ToFloat16({2.0f, 2.0f, -3.0f, -3.0f}));
  test.AddOutput<MLFloat16>("C", {4}, ToFloat16({2.0f, -3.0f, -3.0f, 2.0f}));
  CompareWithMusaNoFallback(test, true, 1e-3, 1e-3);
}

TEST(MusaComparisonElementwiseTest, FloorDivInt32Scalar) {
  CompareOpTester test("FloorDiv", 17);
  test.AddInput<int32_t>("A", {4}, {5, -5, 7, -7});
  test.AddInput<int32_t>("B", {}, {2});
  test.AddOutput<int32_t>("C", {4}, {2, -3, 3, -4});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, FloorDivInt64Empty) {
  CompareOpTester test("FloorDiv", 17);
  test.AddInput<int64_t>("A", {2, 0}, std::vector<int64_t>{});
  test.AddInput<int64_t>("B", {1, 0}, std::vector<int64_t>{});
  test.AddOutput<int64_t>("C", {2, 0}, std::vector<int64_t>{});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, FloorModFloatBroadcastDynamicShape) {
  CompareOpTester test("FloorMod", 17);
  const std::vector<std::string> a_dim_params{"batch", "seq"};
  test.AddInput<float>("A", {2, 3}, {5.0f, -5.0f, 7.0f, -7.0f, 9.0f, -9.0f}, false, &a_dim_params);
  test.AddInput<float>("B", {1, 3}, {2.0f, -3.0f, 4.0f});
  test.AddOutput<float>("C", {2, 3}, {1.0f, -2.0f, 3.0f, 1.0f, 0.0f, 3.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, FloorModFloat16NoCpuFallback) {
  CompareOpTester test("FloorMod", 17);
  test.AddInput<MLFloat16>("A", {4}, ToFloat16({5.0f, -5.0f, 7.0f, -7.0f}));
  test.AddInput<MLFloat16>("B", {4}, ToFloat16({2.0f, 2.0f, -3.0f, -3.0f}));
  test.AddOutput<MLFloat16>("C", {4}, ToFloat16({1.0f, 1.0f, -2.0f, -1.0f}));
  CompareWithMusaNoFallback(test, true, 1e-3, 1e-3);
}

TEST(MusaComparisonElementwiseTest, FloorModInt64Scalar) {
  CompareOpTester test("FloorMod", 17);
  test.AddInput<int64_t>("A", {4}, {5, -5, 7, -7});
  test.AddInput<int64_t>("B", {}, {-3});
  test.AddOutput<int64_t>("C", {4}, {-1, -2, -2, -1});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, FloorModInt32Empty) {
  CompareOpTester test("FloorMod", 17);
  test.AddInput<int32_t>("A", {2, 0}, std::vector<int32_t>{});
  test.AddInput<int32_t>("B", {1, 0}, std::vector<int32_t>{});
  test.AddOutput<int32_t>("C", {2, 0}, std::vector<int32_t>{});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, FloorDivModMultiOpGraph) {
  FloorDivModTester test;
  test.AddInput<float>("A", {4}, {5.0f, -5.0f, 7.0f, -7.0f});
  test.AddInput<float>("B", {4}, {2.0f, 2.0f, -3.0f, -3.0f});
  test.AddOutput<float>("Y", {4}, {3.0f, -2.0f, -5.0f, 1.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, IdentityNFloatAndInt32NoCpuFallback) {
  CompareOpTester test("IdentityN", 17);
  test.AddInput<float>("X0", {2, 2}, {1.0f, -2.0f, 3.0f, 4.0f});
  test.AddInput<int32_t>("X1", {3}, {7, -8, 9});
  test.AddOutput<float>("Y0", {2, 2}, {1.0f, -2.0f, 3.0f, 4.0f});
  test.AddOutput<int32_t>("Y1", {3}, {7, -8, 9});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, IdentityNFloat16BoolAndEmptyNoCpuFallback) {
  CompareOpTester test("IdentityN", 17);
  test.AddInput<MLFloat16>("X0", {2}, ToFloat16({1.5f, -2.5f}));
  test.AddInput<bool>("X1", {}, {true});
  test.AddInput<float>("X2", {0}, std::vector<float>{});
  test.AddOutput<MLFloat16>("Y0", {2}, ToFloat16({1.5f, -2.5f}));
  test.AddOutput<bool>("Y1", {}, {true});
  test.AddOutput<float>("Y2", {0}, std::vector<float>{});
  CompareWithMusaNoFallback(test, true, 1e-3, 1e-3);
}

TEST(MusaComparisonElementwiseTest, ShapeNInt32MixedDynamicAndScalarNoCpuFallback) {
  CompareOpTester test("ShapeN", 17);
  test.AddAttribute("out_type", int64_t{ONNX_NAMESPACE::TensorProto_DataType_INT32});
  const std::vector<std::string> dims{"batch", "seq"};
  test.AddInput<float>("X0", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, false, &dims);
  test.AddInput<int64_t>("X1", {}, {42});
  test.AddOutput<int32_t>("Y0", {2}, {2, 3});
  test.AddOutput<int32_t>("Y1", {0}, std::vector<int32_t>{});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, ShapeNInt64Float16AndEmptyBoolNoCpuFallback) {
  CompareOpTester test("ShapeN", 17);
  test.AddAttribute("out_type", int64_t{ONNX_NAMESPACE::TensorProto_DataType_INT64});
  test.AddInput<MLFloat16>("X0", {0, 4}, std::vector<MLFloat16>{});
  test.AddInput<bool>("X1", {1}, {true});
  test.AddOutput<int64_t>("Y0", {2}, {0, 4});
  test.AddOutput<int64_t>("Y1", {1}, {1});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, ShapeNFloatOutputNoCpuFallback) {
  CompareOpTester test("ShapeN", 17);
  test.AddAttribute("out_type", int64_t{ONNX_NAMESPACE::TensorProto_DataType_FLOAT});
  test.AddInput<int32_t>("X", {2}, {3, 4});
  test.AddOutput<float>("Y", {1}, {2.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, IdentityNShapeNChainNoCpuFallback) {
  IdentityNShapeNChainTester test;
  test.AddInput<float>("X0", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  test.AddInput<bool>("X1", {3}, {true, false, true});
  test.AddOutput<int32_t>("Y0", {2}, {2, 2});
  test.AddOutput<int32_t>("Y1", {1}, {3});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, ConcatV2FloatAxis1DynamicShape) {
  CompareOpTester test("ConcatV2", 17);
  const std::vector<std::string> dims_a{"batch", "1"};
  const std::vector<std::string> dims_b{"batch", "seq"};
  test.AddInput<float>("A", {2, 1}, {1.0f, 4.0f}, false, &dims_a);
  test.AddInput<float>("B", {2, 2}, {2.0f, 3.0f, 5.0f, 6.0f}, false, &dims_b);
  test.AddInput<int32_t>("Axis", {}, {1});
  test.AddOutput<float>("Y", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, ConcatV2Float16Axis0NoCpuFallback) {
  CompareOpTester test("ConcatV2", 17);
  test.AddInput<MLFloat16>("A", {1, 2}, ToFloat16({1.0f, 2.0f}));
  test.AddInput<MLFloat16>("B", {1, 2}, ToFloat16({3.0f, 4.0f}));
  test.AddInput<int64_t>("Axis", {}, {0});
  test.AddOutput<MLFloat16>("Y", {2, 2}, ToFloat16({1.0f, 2.0f, 3.0f, 4.0f}));
  CompareWithMusaNoFallback(test, true, 1e-3, 1e-3);
}

TEST(MusaComparisonElementwiseTest, ConcatV2Int32WithEmptyInput) {
  CompareOpTester test("ConcatV2", 17);
  test.AddInput<int32_t>("A", {0, 2}, std::vector<int32_t>{});
  test.AddInput<int32_t>("B", {1, 2}, {7, 8});
  test.AddInput<int32_t>("Axis", {}, {0});
  test.AddOutput<int32_t>("Y", {1, 2}, {7, 8});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, ConcatV2BoolNoCpuFallback) {
  CompareOpTester test("ConcatV2", 17);
  test.AddInput<bool>("A", {2}, {true, false});
  test.AddInput<bool>("B", {1}, {true});
  test.AddInput<int64_t>("Axis", {}, {0});
  test.AddOutput<bool>("Y", {3}, {true, false, true});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, SplitVFloatWithInferDynamicShape) {
  CompareOpTester test("SplitV", 17);
  const std::vector<std::string> dims{"batch", "seq"};
  test.AddInput<float>("X", {2, 4}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, false, &dims);
  test.AddInput<int32_t>("Sizes", {2}, {1, -1});
  test.AddInput<int32_t>("Axis", {}, {1});
  test.AddOutput<float>("Y0", {2, 1}, {1.0f, 5.0f});
  test.AddOutput<float>("Y1", {2, 3}, {2.0f, 3.0f, 4.0f, 6.0f, 7.0f, 8.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, SplitVFloat16EmptyNoCpuFallback) {
  CompareOpTester test("SplitV", 17);
  test.AddInput<MLFloat16>("X", {2, 0}, std::vector<MLFloat16>{});
  test.AddInput<int64_t>("Sizes", {2}, {0, 0});
  test.AddInput<int64_t>("Axis", {}, {1});
  test.AddOutput<MLFloat16>("Y0", {2, 0}, std::vector<MLFloat16>{});
  test.AddOutput<MLFloat16>("Y1", {2, 0}, std::vector<MLFloat16>{});
  CompareWithMusaNoFallback(test, true, 1e-3, 1e-3);
}

TEST(MusaComparisonElementwiseTest, SplitVInt32NegativeAxis) {
  CompareOpTester test("SplitV", 17);
  test.AddInput<int32_t>("X", {2, 3}, {1, 2, 3, 4, 5, 6});
  test.AddInput<int32_t>("Sizes", {2}, {1, 2});
  test.AddInput<int32_t>("Axis", {}, {-1});
  test.AddOutput<int32_t>("Y0", {2, 1}, {1, 4});
  test.AddOutput<int32_t>("Y1", {2, 2}, {2, 3, 5, 6});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, SplitVInt64AndBoolNoCpuFallback) {
  CompareOpTester int64_test("SplitV", 17);
  int64_test.AddInput<int64_t>("X", {4}, {10, 20, 30, 40});
  int64_test.AddInput<int64_t>("Sizes", {2}, {2, 2});
  int64_test.AddInput<int64_t>("Axis", {}, {0});
  int64_test.AddOutput<int64_t>("Y0", {2}, {10, 20});
  int64_test.AddOutput<int64_t>("Y1", {2}, {30, 40});
  CompareWithMusaNoFallback(int64_test);

  CompareOpTester bool_test("SplitV", 17);
  bool_test.AddInput<bool>("X", {3}, {true, false, true});
  bool_test.AddInput<int32_t>("Sizes", {2}, {1, 2});
  bool_test.AddInput<int32_t>("Axis", {}, {0});
  bool_test.AddOutput<bool>("Y0", {1}, {true});
  bool_test.AddOutput<bool>("Y1", {2}, {false, true});
  CompareWithMusaNoFallback(bool_test);
}

TEST(MusaComparisonElementwiseTest, ConcatV2SplitVMultiOpGraph) {
  ConcatV2SplitVChainTester test;
  test.AddInput<float>("A", {2, 1}, {1.0f, 5.0f});
  test.AddInput<float>("B", {2, 3}, {2.0f, 3.0f, 4.0f, 6.0f, 7.0f, 8.0f});
  test.AddInput<int64_t>("ConcatAxis", {}, {1});
  test.AddInput<int64_t>("SplitSizes", {2}, {1, 3});
  test.AddInput<int64_t>("SplitAxis", {}, {1});
  test.AddOutput<float>("Y", {2, 4}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  CompareWithMusaNoFallback(test);
}


TEST(MusaComparisonElementwiseTest, BroadcastGradientArgsInt32NoCpuFallback) {
  CompareOpTester test("BroadcastGradientArgs", 17);
  test.AddInput<int32_t>("S0", {3}, {2, 1, 3});
  test.AddInput<int32_t>("S1", {2}, {4, 3});
  test.AddOutput<int32_t>("R0", {1}, {1});
  test.AddOutput<int32_t>("R1", {1}, {0});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, BroadcastGradientArgsInt64ScalarShapeNoCpuFallback) {
  CompareOpTester test("BroadcastGradientArgs", 17);
  test.AddInput<int64_t>("S0", {0}, std::vector<int64_t>{});
  test.AddInput<int64_t>("S1", {1}, {5});
  test.AddOutput<int64_t>("R0", {1}, {0});
  test.AddOutput<int64_t>("R1", {0}, std::vector<int64_t>{});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, ConcatOffsetInt32ThreeInputsNoCpuFallback) {
  CompareOpTester test("ConcatOffset", 17);
  test.AddInput<int32_t>("ConcatDim", {}, {1});
  test.AddInput<int32_t>("Shape0", {2}, {2, 3});
  test.AddInput<int32_t>("Shape1", {2}, {2, 4});
  test.AddInput<int32_t>("Shape2", {2}, {2, 1});
  test.AddOutput<int32_t>("Offset0", {2}, {0, 0});
  test.AddOutput<int32_t>("Offset1", {2}, {0, 3});
  test.AddOutput<int32_t>("Offset2", {2}, {0, 7});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, ConcatOffsetInt64NegativeAxisNoCpuFallback) {
  CompareOpTester test("ConcatOffset", 17);
  test.AddInput<int64_t>("ConcatDim", {}, {-1});
  test.AddInput<int64_t>("Shape0", {3}, {2, 3, 4});
  test.AddInput<int64_t>("Shape1", {3}, {2, 3, 5});
  test.AddOutput<int64_t>("Offset0", {3}, {0, 0, 0});
  test.AddOutput<int64_t>("Offset1", {3}, {0, 0, 4});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, BroadcastGradientArgsConcatV2MultiOpGraph) {
  BroadcastGradientArgsConcatV2Tester test;
  test.AddInput<int64_t>("S0", {3}, {2, 1, 3});
  test.AddInput<int64_t>("S1", {2}, {4, 3});
  test.AddInput<int64_t>("Axis", {}, {0});
  test.AddOutput<int64_t>("Y", {2}, {1, 0});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, ReverseV2FloatAxis1DynamicShape) {
  CompareOpTester test("ReverseV2", 17);
  const std::vector<std::string> dims{"batch", "seq"};
  test.AddInput<float>("X", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, false, &dims);
  test.AddInput<int32_t>("Axis", {1}, {1});
  test.AddOutput<float>("Y", {2, 3}, {3.0f, 2.0f, 1.0f, 6.0f, 5.0f, 4.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, ReverseV2Float16MultiAxisNoCpuFallback) {
  CompareOpTester test("ReverseV2", 17);
  test.AddInput<MLFloat16>("X", {2, 2}, ToFloat16({1.0f, 2.0f, 3.0f, 4.0f}));
  test.AddInput<int64_t>("Axis", {2}, {0, -1});
  test.AddOutput<MLFloat16>("Y", {2, 2}, ToFloat16({4.0f, 3.0f, 2.0f, 1.0f}));
  CompareWithMusaNoFallback(test, true, 1e-3, 1e-3);
}

TEST(MusaComparisonElementwiseTest, ReverseV2Int32ScalarNoAxis) {
  CompareOpTester test("ReverseV2", 17);
  test.AddInput<int32_t>("X", {}, {7});
  test.AddInput<int32_t>("Axis", {0}, std::vector<int32_t>{});
  test.AddOutput<int32_t>("Y", {}, {7});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, ReverseV2BoolNoCpuFallback) {
  CompareOpTester test("ReverseV2", 17);
  test.AddInput<bool>("X", {2, 2}, {true, false, false, true});
  test.AddInput<int64_t>("Axis", {1}, {0});
  test.AddOutput<bool>("Y", {2, 2}, {false, true, true, false});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, InvertPermutationInt32NoCpuFallback) {
  CompareOpTester test("InvertPermutation", 17);
  test.AddInput<int32_t>("X", {3}, {2, 0, 1});
  test.AddOutput<int32_t>("Y", {3}, {1, 2, 0});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, InvertPermutationInt64EmptyNoCpuFallback) {
  CompareOpTester test("InvertPermutation", 17);
  test.AddInput<int64_t>("X", {0}, std::vector<int64_t>{});
  test.AddOutput<int64_t>("Y", {0}, std::vector<int64_t>{});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, ReverseV2InvertPermutationMultiOpGraph) {
  ReverseV2InvertPermutationChainTester test;
  test.AddInput<float>("X", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddInput<int64_t>("Perm", {2}, {1, 0});
  test.AddOutput<float>("Y", {2, 3}, {6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, ZerosLikeFloatDynamicShape) {
  CompareOpTester test("ZerosLike", 17);
  const std::vector<std::string> dim_params{"batch", "seq"};
  test.AddInput<float>("X", {2, 3}, {1.0f, -2.0f, 3.0f, 4.0f, 0.5f, -6.0f}, false, &dim_params);
  test.AddOutput<float>("Y", {2, 3}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, ZerosLikeFloat16NoCpuFallback) {
  CompareOpTester test("ZerosLike", 17);
  test.AddInput<MLFloat16>("X", {4}, ToFloat16({1.0f, -2.0f, 3.0f, 4.0f}));
  test.AddOutput<MLFloat16>("Y", {4}, ToFloat16({0.0f, 0.0f, 0.0f, 0.0f}));
  CompareWithMusaNoFallback(test, true, 1e-3, 1e-3);
}

TEST(MusaComparisonElementwiseTest, ZerosLikeInt32Scalar) {
  CompareOpTester test("ZerosLike", 17);
  test.AddInput<int32_t>("X", {}, {7});
  test.AddOutput<int32_t>("Y", {}, {0});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, ZerosLikeInt64Empty) {
  CompareOpTester test("ZerosLike", 17);
  test.AddInput<int64_t>("X", {2, 0}, std::vector<int64_t>{});
  test.AddOutput<int64_t>("Y", {2, 0}, std::vector<int64_t>{});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, ZerosLikeBool) {
  CompareOpTester test("ZerosLike", 17);
  test.AddInput<bool>("X", {4}, {true, false, true, true});
  test.AddOutput<bool>("Y", {4}, {false, false, false, false});
  CompareWithMusaNoFallback(test);
}

TEST(MusaComparisonElementwiseTest, ZerosLikeAddMultiOpGraph) {
  ZerosLikeAddTester test;
  test.AddInput<float>("X", {2, 3}, {1.0f, -2.0f, 3.0f, 4.0f, 0.5f, -6.0f});
  test.AddInput<float>("Bias", {2, 3}, {0.25f, 2.0f, -4.0f, 8.0f, 1.0f, -3.0f});
  test.AddOutput<float>("Y", {2, 3}, {0.25f, 2.0f, -4.0f, 8.0f, 1.0f, -3.0f});
  CompareWithMusaNoFallback(test);
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
