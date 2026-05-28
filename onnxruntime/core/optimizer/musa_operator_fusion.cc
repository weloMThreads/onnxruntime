// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/musa_operator_fusion.h"

#include "core/common/inlined_containers.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;

namespace onnxruntime {

namespace {

bool IsSupportedOptypeVersionAndDomain(const Node& node, const std::string& op_type,
                                       std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion> versions,
                                       std::string_view domain = kOnnxDomain) {
  return node.OpType() == op_type && graph_utils::MatchesOpSinceVersion(node, versions) &&
         graph_utils::MatchesOpSetDomain(node, domain);
}

bool IsSupportedRelu(const Node& node) {
  return IsSupportedOptypeVersionAndDomain(node, "Relu", {6, 13, 14}, kOnnxDomain);
}

bool IsSupportedConcat(const Node& node) {
  return IsSupportedOptypeVersionAndDomain(node, "Concat", {1, 4, 11, 13}, kOnnxDomain);
}

bool IsSupportedDataType(const NodeArg* node_arg) {
  if (node_arg == nullptr || node_arg->TypeAsProto() == nullptr) {
    return false;
  }

  const auto elem_type = node_arg->TypeAsProto()->tensor_type().elem_type();
  return elem_type == TensorProto_DataType_FLOAT || elem_type == TensorProto_DataType_FLOAT16;
}

bool HasBiasAddShape(const TensorShapeProto& bias_shape, const TensorShapeProto_Dimension& dim_n) {
  auto dim_has_value_1 = [](const TensorShapeProto_Dimension& dim) {
    return dim.has_dim_value() && dim.dim_value() == 1;
  };

  return (bias_shape.dim_size() == 1 && bias_shape.dim(0) == dim_n) ||
         (bias_shape.dim_size() == 2 && dim_has_value_1(bias_shape.dim(0)) && bias_shape.dim(1) == dim_n);
}

bool NormalizeAxis(int64_t axis, size_t rank, int64_t& normalized_axis) {
  const auto rank_i64 = static_cast<int64_t>(rank);
  if (rank == 0 || axis < -rank_i64 || axis >= rank_i64) {
    return false;
  }

  normalized_axis = axis < 0 ? axis + rank_i64 : axis;
  return true;
}

bool GetStaticShapeValues(const NodeArg* node_arg, TensorShapeVector& dims) {
  if (node_arg == nullptr || node_arg->Shape() == nullptr) {
    return false;
  }

  const auto* shape = node_arg->Shape();
  dims.clear();
  dims.reserve(static_cast<size_t>(shape->dim_size()));
  for (int i = 0; i < shape->dim_size(); ++i) {
    const auto& dim = shape->dim(i);
    if (!utils::HasDimValue(dim)) {
      return false;
    }
    dims.push_back(dim.dim_value());
  }

  return true;
}

bool TryFuseLinearPattern(Graph& graph, Node& node, bool& modified) {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "MatMul", {1, 9, 13}) ||
      node.GetExecutionProviderType() != kMusaExecutionProvider ||
      node.GetOutputEdgesCount() != 1 || graph.NodeProducesGraphOutput(node)) {
    return false;
  }

  auto next_node_itr = node.OutputNodesBegin();
  if (next_node_itr == node.OutputNodesEnd()) {
    return false;
  }

  Node& matmul_node = node;
  Node& add_node = *graph.GetNode(next_node_itr->Index());
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(add_node, "Add", {7, 13, 14}) ||
      add_node.GetExecutionProviderType() != matmul_node.GetExecutionProviderType()) {
    return false;
  }

  Node* relu_node = nullptr;
  if (add_node.GetOutputEdgesCount() == 1 && !graph.NodeProducesGraphOutput(add_node)) {
    auto relu_itr = add_node.OutputNodesBegin();
    if (relu_itr != add_node.OutputNodesEnd()) {
      auto* maybe_relu = graph.GetNode(relu_itr->Index());
      if (maybe_relu != nullptr && IsSupportedRelu(*maybe_relu) &&
          maybe_relu->GetExecutionProviderType() == add_node.GetExecutionProviderType()) {
        relu_node = maybe_relu;
      }
    }
  }

  auto matmul_input_defs = matmul_node.MutableInputDefs();
  auto add_input_defs = add_node.MutableInputDefs();
  if (matmul_input_defs.size() != 2 || add_input_defs.size() != 2) {
    return false;
  }

  auto* matmul_type = matmul_input_defs[0]->Type();
  auto* add_type = add_input_defs[0]->Type();
  if (matmul_type == nullptr || add_type == nullptr || *matmul_type != *add_type) {
    return false;
  }

  if (!IsSupportedDataType(matmul_input_defs[0])) {
    return false;
  }

  auto* matmul_a_shape = matmul_input_defs[0]->Shape();
  auto* matmul_b_shape = matmul_input_defs[1]->Shape();
  if (matmul_a_shape == nullptr || matmul_b_shape == nullptr || matmul_b_shape->dim_size() != 2) {
    return false;
  }

  bool need_reshape = matmul_a_shape->dim_size() != 2;
  const auto& dim_n = matmul_b_shape->dim(1);
  InlinedVector<int64_t> output_shape_values;
  int64_t m = 0;
  int64_t k = 0;
  int64_t n = 0;

  if (need_reshape) {
    auto a_shape = utils::GetTensorShapeFromTensorShapeProto(*matmul_a_shape);
    if (a_shape.Size() == -1) {
      return false;
    }

    const auto& dim_k = matmul_b_shape->dim(0);
    if (!utils::HasDimValue(dim_k) || !utils::HasDimValue(dim_n)) {
      return false;
    }

    output_shape_values = a_shape.AsShapeVector();
    m = a_shape.SizeToDimension(a_shape.NumDimensions() - 1);
    k = dim_k.dim_value();
    n = dim_n.dim_value();
  }

  const auto& matmul_output = *matmul_node.OutputDefs()[0];
  const auto matmul_output_name = matmul_output.Name();
  const int bias_idx = matmul_output_name == add_input_defs[0]->Name() ? 1 : 0;
  auto* bias_arg = add_input_defs[bias_idx];
  if (bias_arg == nullptr || bias_arg->Shape() == nullptr || !HasBiasAddShape(*bias_arg->Shape(), dim_n)) {
    return false;
  }

  std::vector<NodeArg*> fused_input_defs{matmul_input_defs[0], matmul_input_defs[1], bias_arg};
  auto fused_output_defs = relu_node ? relu_node->MutableOutputDefs() : add_node.MutableOutputDefs();
  Node* input_node = nullptr;
  Node* output_node = nullptr;

  if (need_reshape) {
    auto add_reshape = [&](const InlinedVector<int64_t>& shape, bool is_input) -> Node* {
      const std::string name = is_input ? "musa_fused_gemm_input" : "musa_fused_gemm_output";
      TensorProto shape_initializer_proto;
      shape_initializer_proto.set_name(graph.GenerateNodeName(name + "_shape"));
      shape_initializer_proto.add_dims(static_cast<int64_t>(shape.size()));
      shape_initializer_proto.set_data_type(TensorProto_DataType_INT64);
      utils::SetRawDataInTensorProto(shape_initializer_proto, shape.data(), shape.size() * sizeof(int64_t));
      NodeArg* shape_arg = &graph_utils::AddInitializerWithOrtValue(graph, shape_initializer_proto);

      TypeProto reshape_output_type;
      const auto element_type = static_cast<TensorProto_DataType>(
          fused_input_defs[0]->TypeAsProto()->tensor_type().elem_type());
      reshape_output_type.mutable_tensor_type()->set_elem_type(element_type);
      reshape_output_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(m);
      reshape_output_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(is_input ? k : n);

      NodeArg* reshape_output_arg =
          &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(name + "_reshape_arg"), &reshape_output_type);
      Node& reshape_node = graph.AddNode(graph.GenerateNodeName(name + "_reshape"), "Reshape",
                                         "Reshape for " + name,
                                         {is_input ? fused_input_defs[0] : reshape_output_arg, shape_arg},
                                         {is_input ? reshape_output_arg : fused_output_defs[0]}, matmul_node);
      reshape_node.SetExecutionProviderType(matmul_node.GetExecutionProviderType());
      return &reshape_node;
    };

    input_node = add_reshape({m, k}, true);
    fused_input_defs[0] = input_node->MutableOutputDefs()[0];
    output_shape_values.back() = n;
    output_node = add_reshape(output_shape_values, false);
    fused_output_defs[0] = output_node->MutableInputDefs()[0];
  }

  Node& fused_node = graph.AddNode(graph.GenerateNodeName(matmul_node.Name() + "/MusaOperatorFusion"),
                                   "FusedGemm",
                                   relu_node ? "fused MatMul, BiasAdd and Relu for MUSA"
                                             : "fused MatMul and BiasAdd for MUSA",
                                   fused_input_defs, fused_output_defs, nullptr, kMSDomain);
  fused_node.SetExecutionProviderType(matmul_node.GetExecutionProviderType());
  if (relu_node != nullptr) {
    fused_node.AddAttribute("activation", std::string{"Relu"});
  }

  if (need_reshape) {
    graph.AddEdge(input_node->Index(), fused_node.Index(), 0, 0);
    graph.AddEdge(fused_node.Index(), output_node->Index(), 0, 0);
  } else {
    input_node = &fused_node;
    output_node = &fused_node;
  }

  auto matmul_input_edges = graph_utils::GraphEdge::GetNodeInputEdges(matmul_node);
  for (const auto& edge : matmul_input_edges) {
    if (edge.dst_arg_index == 0) {
      graph.AddEdge(edge.src_node, input_node->Index(), edge.src_arg_index, 0);
    } else if (edge.dst_arg_index == 1) {
      graph.AddEdge(edge.src_node, fused_node.Index(), edge.src_arg_index, 1);
    }
  }
  graph_utils::GraphEdge::RemoveGraphEdges(graph, matmul_input_edges);

  auto add_input_edges = graph_utils::GraphEdge::GetNodeInputEdges(add_node);
  for (const auto& edge : add_input_edges) {
    if (edge.dst_arg_index == bias_idx) {
      graph.AddEdge(edge.src_node, fused_node.Index(), edge.src_arg_index, 2);
      break;
    }
  }
  graph_utils::GraphEdge::RemoveGraphEdges(graph, add_input_edges);

  graph_utils::RemoveNodeOutputEdges(graph, matmul_node);
  if (relu_node != nullptr) {
    graph_utils::RemoveNodeOutputEdges(graph, add_node);
    graph_utils::ReplaceDownstreamNodeInput(graph, *relu_node, 0, *output_node, 0);
    graph.RemoveNode(relu_node->Index());
  } else {
    graph_utils::ReplaceDownstreamNodeInput(graph, add_node, 0, *output_node, 0);
  }

  graph.RemoveNode(matmul_node.Index());
  graph.RemoveNode(add_node.Index());
  modified = true;
  return true;
}

bool CanFuseConcatMatMul(const Node& concat_node, const Node& matmul_node, int concat_input_idx,
                         int64_t& normalized_axis) {
  const auto concat_input_defs = concat_node.InputDefs();
  const auto matmul_input_defs = matmul_node.InputDefs();
  if (concat_input_defs.size() < 2 || matmul_input_defs.size() != 2) {
    return false;
  }

  auto* other_input_arg = matmul_input_defs[1 - concat_input_idx];
  if (!IsSupportedDataType(concat_input_defs[0]) || !IsSupportedDataType(other_input_arg)) {
    return false;
  }

  auto* concat_type = concat_input_defs[0]->Type();
  auto* other_type = other_input_arg->Type();
  if (concat_type == nullptr || other_type == nullptr || *concat_type != *other_type) {
    return false;
  }

  TensorShapeVector concat_dims;
  if (!GetStaticShapeValues(concat_input_defs[0], concat_dims) || concat_dims.size() < 2) {
    return false;
  }

  const auto* axis_attr = graph_utils::GetNodeAttribute(concat_node, "axis");
  if (axis_attr == nullptr || !NormalizeAxis(axis_attr->i(), concat_dims.size(), normalized_axis)) {
    return false;
  }

  const size_t axis_index = static_cast<size_t>(normalized_axis);
  concat_dims[axis_index] = 0;
  for (auto* input_arg : concat_input_defs) {
    TensorShapeVector dims;
    if (!GetStaticShapeValues(input_arg, dims) || dims.size() != concat_dims.size()) {
      return false;
    }
    for (size_t i = 0; i < dims.size(); ++i) {
      if (i == axis_index) {
        continue;
      }
      if (dims[i] != concat_dims[i]) {
        return false;
      }
    }
    if (dims[axis_index] <= 0) {
      return false;
    }
    concat_dims[axis_index] += dims[axis_index];
  }

  TensorShapeVector other_dims;
  if (!GetStaticShapeValues(other_input_arg, other_dims) || other_dims.size() != concat_dims.size()) {
    return false;
  }

  const auto& lhs_dims = concat_input_idx == 0 ? concat_dims : other_dims;
  const auto& rhs_dims = concat_input_idx == 0 ? other_dims : concat_dims;
  const size_t rank = lhs_dims.size();
  for (size_t i = 0; i + 2 < rank; ++i) {
    if (lhs_dims[i] <= 0 || rhs_dims[i] <= 0 || lhs_dims[i] != rhs_dims[i]) {
      return false;
    }
  }

  return lhs_dims[rank - 2] > 0 && lhs_dims[rank - 1] > 0 &&
         rhs_dims[rank - 2] > 0 && rhs_dims[rank - 1] > 0 &&
         lhs_dims[rank - 1] == rhs_dims[rank - 2];
}

bool TryFuseConcatMatMul(Graph& graph, Node& matmul_node, bool& modified) {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(matmul_node, "MatMul", {1, 9, 13}) ||
      matmul_node.GetExecutionProviderType() != kMusaExecutionProvider) {
    return false;
  }

  Node* concat_node = nullptr;
  int concat_input_idx = -1;
  for (int i = 0; i < 2; ++i) {
    const Node* input_node = graph_utils::GetInputNode(matmul_node, i);
    if (input_node != nullptr && IsSupportedConcat(*input_node) &&
        input_node->GetExecutionProviderType() == matmul_node.GetExecutionProviderType()) {
      concat_node = graph.GetNode(input_node->Index());
      concat_input_idx = i;
      break;
    }
  }

  if (concat_node == nullptr || concat_input_idx < 0 || concat_node->GetOutputEdgesCount() != 1 ||
      graph.NodeProducesGraphOutput(*concat_node)) {
    return false;
  }

  int64_t axis = 0;
  if (!CanFuseConcatMatMul(*concat_node, matmul_node, concat_input_idx, axis)) {
    return false;
  }

  auto concat_input_defs = concat_node->MutableInputDefs();
  auto matmul_input_defs = matmul_node.MutableInputDefs();
  std::vector<NodeArg*> fused_input_defs;
  fused_input_defs.reserve(concat_input_defs.size() + 1);
  fused_input_defs.insert(fused_input_defs.end(), concat_input_defs.begin(), concat_input_defs.end());
  fused_input_defs.push_back(matmul_input_defs[1 - concat_input_idx]);

  Node& fused_node = graph.AddNode(graph.GenerateNodeName(matmul_node.Name() + "/MusaConcatMatMul"),
                                   "MusaConcatMatMul",
                                   "fused Concat and MatMul for MUSA",
                                   fused_input_defs,
                                   matmul_node.MutableOutputDefs(),
                                   nullptr,
                                   kMSDomain);
  fused_node.SetExecutionProviderType(matmul_node.GetExecutionProviderType());
  fused_node.AddAttribute("axis", axis);
  fused_node.AddAttribute("concat_input_idx", static_cast<int64_t>(concat_input_idx));

  auto concat_input_edges = graph_utils::GraphEdge::GetNodeInputEdges(*concat_node);
  for (const auto& edge : concat_input_edges) {
    graph.AddEdge(edge.src_node, fused_node.Index(), edge.src_arg_index, edge.dst_arg_index);
  }

  const int other_input_slot = static_cast<int>(concat_input_defs.size());
  auto matmul_input_edges = graph_utils::GraphEdge::GetNodeInputEdges(matmul_node);
  for (const auto& edge : matmul_input_edges) {
    if (edge.dst_arg_index == 1 - concat_input_idx) {
      graph.AddEdge(edge.src_node, fused_node.Index(), edge.src_arg_index, other_input_slot);
    }
  }

  graph_utils::GraphEdge::RemoveGraphEdges(graph, concat_input_edges);
  graph_utils::GraphEdge::RemoveGraphEdges(graph, matmul_input_edges);
  graph_utils::RemoveNodeOutputEdges(graph, *concat_node);
  graph_utils::ReplaceDownstreamNodeInput(graph, matmul_node, 0, fused_node, 0);

  graph.RemoveNode(matmul_node.Index());
  graph.RemoveNode(concat_node->Index());
  modified = true;
  return true;
}

}  // namespace

Status MusaOperatorFusionTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                                const logging::Logger& logger) const {
  {
    GraphViewer graph_viewer(graph);
    const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

    for (auto node_index : node_topology_list) {
      auto* node_ptr = graph.GetNode(node_index);
      if (!node_ptr) {
        continue;
      }

      auto& node = *node_ptr;
      ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));
      TryFuseLinearPattern(graph, node, modified);
    }
  }

  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (!node_ptr) {
      continue;
    }

    TryFuseConcatMatMul(graph, *node_ptr, modified);
  }

  return Status::OK();
}

}  // namespace onnxruntime
