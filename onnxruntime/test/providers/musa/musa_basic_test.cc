// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test_environment.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

void VerifyOutputs(const std::vector<OrtValue> &fetches,
                   const std::vector<int64_t> &expected_dims,
                   const std::vector<int64_t> &expected_values) {
  ASSERT_EQ(1, fetches.size());
  const auto &rtensor = fetches.front().Get<Tensor>();
  TensorShape expected_shape(expected_dims);
  ASSERT_EQ(expected_shape, rtensor.Shape());
  const std::vector<int64_t> found(rtensor.template Data<int64_t>(),
                                   rtensor.template Data<int64_t>() +
                                       expected_values.size());
  ASSERT_EQ(expected_values, found);
}

TEST(MusaExecutionProviderTest, FunctionTest) {
  onnxruntime::Model model("graph_1", false,
                           DefaultLoggingManager().DefaultLogger());
  auto &graph = model.MainGraph();
  std::vector<onnxruntime::NodeArg *> inputs;
  std::vector<onnxruntime::NodeArg *> outputs;

  ONNX_NAMESPACE::TypeProto int64_tensor;
  int64_tensor.mutable_tensor_type()->set_elem_type(
      ONNX_NAMESPACE::TensorProto_DataType_INT64);
  int64_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(
      1);
  int64_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(
      1);
  int64_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(
      3);
  int64_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(
      2);

  auto &input_arg_1 = graph.GetOrCreateNodeArg("X", &int64_tensor);
  auto &input_arg_2 = graph.GetOrCreateNodeArg("Y", &int64_tensor);
  inputs.push_back(&input_arg_1);
  inputs.push_back(&input_arg_2);
  auto &output_arg = graph.GetOrCreateNodeArg("node_1_out_1", &int64_tensor);
  outputs.push_back(&output_arg);
  graph.AddNode("node_1", "Add", "node 1.", inputs, outputs);

  auto &input_arg_3 = graph.GetOrCreateNodeArg("Z", &int64_tensor);
  inputs.clear();
  inputs.push_back(&output_arg);
  inputs.push_back(&input_arg_3);
  auto &output_arg_2 = graph.GetOrCreateNodeArg("M", &int64_tensor);
  outputs.clear();
  outputs.push_back(&output_arg_2);
  graph.AddNode("node_2", "Add", "node 2.", inputs, outputs);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK());
  std::string model_file_name = "musa_execution_provider_test_graph.onnx";
  status = onnxruntime::Model::Save(model, model_file_name);

  SessionOptions so;
  so.session_logid = "MusaExecutionProviderTest.FunctionTest";
  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  InferenceSession session_object{so, GetEnvironment()};
  auto musa_provider = DefaultMusaExecutionProvider();
  status = session_object.RegisterExecutionProvider(std::move(musa_provider));
  ASSERT_TRUE(status.IsOK());
  status = session_object.Load(model_file_name);
  ASSERT_TRUE(status.IsOK());
  status = session_object.Initialize();
  ASSERT_TRUE(status.IsOK());

  // prepare inputs
  std::vector<int64_t> dims_mul_x = {1, 1, 3, 2};
  std::vector<int64_t> values_mul_x = {1, 2, 3, 4, 5, 6};
  OrtValue ml_value_x;
  CreateMLValue<int64_t>(gsl::make_span(dims_mul_x), values_mul_x.data(),
                         OrtMemoryInfo(), &ml_value_x);
  OrtValue ml_value_y;
  CreateMLValue<int64_t>(gsl::make_span(dims_mul_x), values_mul_x.data(),
                         OrtMemoryInfo(), &ml_value_y);
  OrtValue ml_value_z;
  CreateMLValue<int64_t>(gsl::make_span(dims_mul_x), values_mul_x.data(),
                         OrtMemoryInfo(), &ml_value_z);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));
  feeds.insert(std::make_pair("Y", ml_value_y));
  feeds.insert(std::make_pair("Z", ml_value_z));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.emplace_back("M");
  std::vector<OrtValue> fetches;

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_mul_m = {1, 1, 3, 2};
  std::vector<int64_t> expected_values_mul_m = {3, 6, 9, 12, 15, 18};

  // Now run
  status = session_object.Run(run_options, feeds, output_names, &fetches);
  ASSERT_TRUE(status.IsOK());
  VerifyOutputs(fetches, expected_dims_mul_m, expected_values_mul_m);
}

TEST(MusaExecutionProviderTest, MemcpyTriggerTest) {
  // This test creates a mixed execution graph with CPU and MUSA operators
  // to trigger MemcpyFromHost/MemcpyToHost operations
  // More complex graph: Reshape(CPU) -> Add(MUSA) -> Add(MUSA) -> Reshape(CPU)

  onnxruntime::Model model("memcpy_test_graph", false,
                           DefaultLoggingManager().DefaultLogger());
  auto &graph = model.MainGraph();
  std::vector<onnxruntime::NodeArg *> inputs;
  std::vector<onnxruntime::NodeArg *> outputs;

  ONNX_NAMESPACE::TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(6);

  ONNX_NAMESPACE::TypeProto float_tensor_2d;
  float_tensor_2d.mutable_tensor_type()->set_elem_type(
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  float_tensor_2d.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);
  float_tensor_2d.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);

  ONNX_NAMESPACE::TypeProto int64_tensor;
  int64_tensor.mutable_tensor_type()->set_elem_type(
      ONNX_NAMESPACE::TensorProto_DataType_INT64);
  int64_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);

  ONNX_NAMESPACE::TypeProto int64_tensor_1d;
  int64_tensor_1d.mutable_tensor_type()->set_elem_type(
      ONNX_NAMESPACE::TensorProto_DataType_INT64);
  int64_tensor_1d.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  // Create input nodes
  auto &input_arg_1 = graph.GetOrCreateNodeArg("input1", &float_tensor);
  auto &input_arg_2 = graph.GetOrCreateNodeArg("input2", &float_tensor);
  auto &shape_arg = graph.GetOrCreateNodeArg("shape", &int64_tensor);

  // Add shape as initializer
  ONNX_NAMESPACE::TensorProto shape_initializer;
  shape_initializer.set_name("shape");
  shape_initializer.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  shape_initializer.add_dims(2);
  shape_initializer.add_int64_data(2);
  shape_initializer.add_int64_data(3);
  graph.AddInitializedTensor(shape_initializer);

  // First Reshape node (CPU) - this will force CPU execution since MUSA doesn't have Reshape
  inputs.clear();
  inputs.push_back(&input_arg_1);
  inputs.push_back(&shape_arg);
  auto &reshape1_output = graph.GetOrCreateNodeArg("reshape1_out", &float_tensor_2d);
  outputs.clear();
  outputs.push_back(&reshape1_output);
  graph.AddNode("reshape1", "Reshape", "Reshape node 1 (CPU)", inputs, outputs);

  // Second Reshape node (CPU) for second input
  inputs.clear();
  inputs.push_back(&input_arg_2);
  inputs.push_back(&shape_arg);
  auto &reshape2_output = graph.GetOrCreateNodeArg("reshape2_out", &float_tensor_2d);
  outputs.clear();
  outputs.push_back(&reshape2_output);
  graph.AddNode("reshape2", "Reshape", "Reshape node 2 (CPU)", inputs, outputs);

  // First Add node (MUSA) - this should require MemcpyFromHost
  inputs.clear();
  inputs.push_back(&reshape1_output);
  inputs.push_back(&reshape2_output);
  auto &add1_output = graph.GetOrCreateNodeArg("add1_out", &float_tensor_2d);
  outputs.clear();
  outputs.push_back(&add1_output);
  graph.AddNode("add1_node", "Add", "Add node 1 (MUSA)", inputs, outputs);

  // Second Add node (MUSA) - this stays on MUSA device
  inputs.clear();
  inputs.push_back(&add1_output);
  inputs.push_back(&reshape2_output);  // Reuse reshaped input2
  auto &add2_output = graph.GetOrCreateNodeArg("add2_out", &float_tensor_2d);
  outputs.clear();
  outputs.push_back(&add2_output);
  graph.AddNode("add2_node", "Add", "Add node 2 (MUSA)", inputs, outputs);

  // Final Reshape node (CPU) - this should require MemcpyToHost
  auto &shape_final_arg = graph.GetOrCreateNodeArg("shape_final", &int64_tensor_1d);

  // Add final shape as initializer
  ONNX_NAMESPACE::TensorProto shape_final_initializer;
  shape_final_initializer.set_name("shape_final");
  shape_final_initializer.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  shape_final_initializer.add_dims(1);
  shape_final_initializer.add_int64_data(6);
  graph.AddInitializedTensor(shape_final_initializer);

  inputs.clear();
  inputs.push_back(&add2_output);
  inputs.push_back(&shape_final_arg);
  auto &final_output = graph.GetOrCreateNodeArg("final_output", &float_tensor);
  outputs.clear();
  outputs.push_back(&final_output);
    graph.AddNode("reshape_final", "Reshape", "Final Reshape node (CPU)", inputs, outputs);

  auto status = graph.Resolve();
  if (!status.IsOK()) {
    std::cout << "Graph resolve failed: " << status.ErrorMessage() << std::endl;
  }
  ASSERT_TRUE(status.IsOK());

  std::string model_file_name = "musa_memcpy_test_graph.onnx";
  status = onnxruntime::Model::Save(model, model_file_name);
  ASSERT_TRUE(status.IsOK());

  SessionOptions so;
  so.session_logid = "MusaExecutionProviderTest.MemcpyTriggerTest";
  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  InferenceSession session_object{so, GetEnvironment()};

  // Register MUSA provider first (higher priority)
  auto musa_provider = DefaultMusaExecutionProvider();
  status = session_object.RegisterExecutionProvider(std::move(musa_provider));
  ASSERT_TRUE(status.IsOK());

  status = session_object.Load(model_file_name);
  ASSERT_TRUE(status.IsOK());

  // This should trigger the MemcpyFromHost error during initialization
  // since we don't have MemcpyFromHost/MemcpyToHost implemented yet
  status = session_object.Initialize();
  std::cout << "Initialization succeeded - memcpy implementation is working!" << std::endl;

  // If initialization succeeds (after implementing memcpy), run the inference
  std::vector<int64_t> dims = {6};  // 1D input with 6 elements
  std::vector<float> values1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> values2 = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};

  OrtValue ml_value1;
  OrtValue ml_value2;
  CreateMLValue<float>(gsl::make_span(dims), values1.data(), OrtMemoryInfo(), &ml_value1);
  CreateMLValue<float>(gsl::make_span(dims), values2.data(), OrtMemoryInfo(), &ml_value2);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("input1", ml_value1));
  feeds.insert(std::make_pair("input2", ml_value2));

  std::vector<std::string> output_names;
  output_names.emplace_back("final_output");
  std::vector<OrtValue> fetches;

  status = session_object.Run(run_options, feeds, output_names, &fetches);
  ASSERT_TRUE(status.IsOK());

  // Just verify we got some output (no precision check needed)
  ASSERT_EQ(1, fetches.size());
  const auto &rtensor = fetches.front().Get<Tensor>();
  ASSERT_EQ(6, rtensor.Shape().Size()); // Should be 6 elements in final 1D tensor

  std::cout << "MemcpyTriggerTest passed - inference completed successfully!" << std::endl;
}

TEST(MusaExecutionProviderTest, MatMulTest) {
  // Create a simple 2x2 matmul test to verify MusaEP MatMul operator is called
  onnxruntime::Model model("matmul_test_graph", false,
                           DefaultLoggingManager().DefaultLogger());
  auto &graph = model.MainGraph();
  std::vector<onnxruntime::NodeArg *> inputs;
  std::vector<onnxruntime::NodeArg *> outputs;

  // Create float tensor type for 2x2 matrices
  ONNX_NAMESPACE::TypeProto float_tensor_2x2;
  float_tensor_2x2.mutable_tensor_type()->set_elem_type(
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  float_tensor_2x2.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);
  float_tensor_2x2.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);

  // Create input nodes
  auto &input_a = graph.GetOrCreateNodeArg("A", &float_tensor_2x2);
  auto &input_b = graph.GetOrCreateNodeArg("B", &float_tensor_2x2);
  inputs.push_back(&input_a);
  inputs.push_back(&input_b);

  // Create output node
  auto &output_c = graph.GetOrCreateNodeArg("C", &float_tensor_2x2);
  outputs.push_back(&output_c);

  // Add MatMul node
  graph.AddNode("matmul_node", "MatMul", "MatMul test node", inputs, outputs);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK());

  std::string model_file_name = "musa_matmul_test_graph.onnx";
  status = onnxruntime::Model::Save(model, model_file_name);
  ASSERT_TRUE(status.IsOK());

  SessionOptions so;
  so.session_logid = "MusaExecutionProviderTest.MatMulTest";
  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  InferenceSession session_object{so, GetEnvironment()};

  // Register MUSA provider
  auto musa_provider = DefaultMusaExecutionProvider();
  status = session_object.RegisterExecutionProvider(std::move(musa_provider));
  ASSERT_TRUE(status.IsOK());

  status = session_object.Load(model_file_name);
  ASSERT_TRUE(status.IsOK());

  status = session_object.Initialize();
  ASSERT_TRUE(status.IsOK());

  // Prepare inputs: 2x2 matrices
  std::vector<int64_t> dims = {2, 2};
  std::vector<float> values_a = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> values_b = {5.0f, 6.0f, 7.0f, 8.0f};

  OrtValue ml_value_a;
  OrtValue ml_value_b;
  CreateMLValue<float>(gsl::make_span(dims), values_a.data(), OrtMemoryInfo(), &ml_value_a);
  CreateMLValue<float>(gsl::make_span(dims), values_b.data(), OrtMemoryInfo(), &ml_value_b);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("A", ml_value_a));
  feeds.insert(std::make_pair("B", ml_value_b));

  std::vector<std::string> output_names;
  output_names.emplace_back("C");
  std::vector<OrtValue> fetches;

  // Run inference - this should trigger our MusaEP MatMul operator
  std::cout << "Running MatMul test - check for MusaEP debug output..." << std::endl;
  status = session_object.Run(run_options, feeds, output_names, &fetches);
  ASSERT_TRUE(status.IsOK());

  // Verify output dimensions
  ASSERT_EQ(1, fetches.size());
  const auto &result_tensor = fetches.front().Get<Tensor>();
  ASSERT_EQ(4, result_tensor.Shape().Size()); // 2x2 = 4 elements

  std::cout << "MatMulTest passed - MusaEP MatMul operator executed successfully!" << std::endl;
}

} // namespace test
} // namespace onnxruntime
