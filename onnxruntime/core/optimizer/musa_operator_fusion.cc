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

}  // namespace

Status MusaOperatorFusionTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                                const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (!node_ptr) {
      continue;
    }

    auto& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "MatMul", {1, 9, 13}) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) ||
        node.GetOutputEdgesCount() != 1 || graph.NodeProducesGraphOutput(node)) {
      continue;
    }

    auto next_node_itr = node.OutputNodesBegin();
    if (next_node_itr == node.OutputNodesEnd()) {
      continue;
    }

    Node& matmul_node = node;
    Node& add_node = *graph.GetNode(next_node_itr->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(add_node, "Add", {7, 13, 14}) ||
        add_node.GetExecutionProviderType() != matmul_node.GetExecutionProviderType()) {
      continue;
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
      continue;
    }

    auto* matmul_type = matmul_input_defs[0]->Type();
    auto* add_type = add_input_defs[0]->Type();
    if (matmul_type == nullptr || add_type == nullptr || *matmul_type != *add_type) {
      continue;
    }

    if (!IsSupportedDataType(matmul_input_defs[0])) {
      continue;
    }

    auto* matmul_a_shape = matmul_input_defs[0]->Shape();
    auto* matmul_b_shape = matmul_input_defs[1]->Shape();
    if (matmul_a_shape == nullptr || matmul_b_shape == nullptr || matmul_b_shape->dim_size() != 2) {
      continue;
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
        continue;
      }

      const auto& dim_k = matmul_b_shape->dim(0);
      if (!utils::HasDimValue(dim_k) || !utils::HasDimValue(dim_n)) {
        continue;
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
      continue;
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
  }

  return Status::OK();
}

}  // namespace onnxruntime
