// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include <unordered_map>

#include "core/providers/musa/musa_provider_options.h"
#include "core/providers/musa/musa_execution_provider_info.h"
#include "core/graph/constants.h"
#include "core/graph/onnx_protobuf.h"
#include "core/graph/model.h"
#include "core/session/environment.h"
#include "core/session/inference_session.h"
#include "gtest/gtest.h"
#include "test/framework/test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/test_environment.h"

namespace onnxruntime {
namespace test {

namespace {

void VerifyFloatOutput(const std::vector<OrtValue>& fetches,
                       const std::vector<int64_t>& expected_dims,
                       const std::vector<float>& expected_values) {
  ASSERT_EQ(fetches.size(), 1U);

  const auto& tensor = fetches.front().Get<Tensor>();
  ASSERT_EQ(tensor.Shape(), TensorShape(expected_dims));

  const float* actual = tensor.Data<float>();
  ASSERT_NE(actual, nullptr);
  for (size_t i = 0; i < expected_values.size(); ++i) {
    EXPECT_FLOAT_EQ(actual[i], expected_values[i]) << "Mismatch at index " << i;
  }
}

int CountNodesWithDomainAndType(const Graph& graph, const std::string& domain, const std::string& op_type) {
  int count = 0;
  for (const auto& node : graph.Nodes()) {
    if (node.Domain() == domain && node.OpType() == op_type) {
      ++count;
    }
  }

  return count;
}

void AddGridSampleNode(Graph& graph,
                       const std::string& domain,
                       const ONNX_NAMESPACE::TypeProto& x_tensor,
                       const ONNX_NAMESPACE::TypeProto& grid_tensor,
                       const ONNX_NAMESPACE::TypeProto& y_tensor) {
  auto& x_arg = graph.GetOrCreateNodeArg("X", &x_tensor);
  auto& grid_arg = graph.GetOrCreateNodeArg("Grid", &grid_tensor);
  auto& y_arg = graph.GetOrCreateNodeArg("Y", &y_tensor);

  Node& node = graph.AddNode("grid_sample_node",
                             "GridSample",
                             "GridSample NHWC transform test",
                             {&x_arg, &grid_arg},
                             {&y_arg},
                             nullptr,
                             domain);
  node.AddAttribute("mode", "bilinear");
  node.AddAttribute("padding_mode", "border");
  node.AddAttribute("align_corners", static_cast<int64_t>(0));
}

void RunGridSampleNhwcTransformTest(const std::string& domain,
                                    const std::string& model_name,
                                    const std::string& optimized_model_name) {
  std::unordered_map<std::string, int> domain_to_version{{kOnnxDomain, 16}};
  if (domain == kMSDomain) {
    domain_to_version[kMSDomain] = 1;
  }

  Model model(model_name, false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), domain_to_version,
              {}, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  ONNX_NAMESPACE::TypeProto x_tensor;
  x_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  for (const int64_t dim : std::vector<int64_t>{1, 1, 2, 2}) {
    x_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
  }

  ONNX_NAMESPACE::TypeProto grid_tensor;
  grid_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  for (const int64_t dim : std::vector<int64_t>{1, 2, 3, 2}) {
    grid_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
  }

  ONNX_NAMESPACE::TypeProto y_tensor;
  y_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  for (const int64_t dim : std::vector<int64_t>{1, 1, 2, 3}) {
    y_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
  }

  AddGridSampleNode(graph, domain, x_tensor, grid_tensor, y_tensor);
  ASSERT_STATUS_OK(graph.Resolve());
  ASSERT_STATUS_OK(Model::Save(model, ToPathString(model_name)));

  SessionOptions session_options;
  session_options.session_logid = model_name;
  session_options.graph_optimization_level = TransformerLevel::MaxLevel;
  session_options.optimized_model_filepath = ToPathString(optimized_model_name);

  OrtMUSAProviderOptions provider_options{};
  provider_options.prefer_nhwc = true;
  auto musa_provider = MusaExecutionProviderWithOptions(&provider_options);
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  InferenceSession session_object{session_options, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(std::move(musa_provider)));
  ASSERT_STATUS_OK(session_object.Load(model_name));
  ASSERT_STATUS_OK(session_object.Initialize());

  std::shared_ptr<Model> optimized_model;
  ASSERT_STATUS_OK(Model::Load(ToPathString(optimized_model_name), optimized_model, nullptr,
                               DefaultLoggingManager().DefaultLogger()));

  const Graph& optimized_graph = optimized_model->MainGraph();
  EXPECT_EQ(CountNodesWithDomainAndType(optimized_graph, kMSInternalNHWCDomain, "GridSample"), 1);
  EXPECT_EQ(CountNodesWithDomainAndType(optimized_graph, kOnnxDomain, "Transpose"), 2);
}

}  // namespace

TEST(MusaGraphTest, OptionPassthrough) {
  OrtMUSAProviderOptions provider_options{};
  ASSERT_EQ(provider_options.enable_musa_graph, 0);

  provider_options.enable_musa_graph = 1;
  ASSERT_EQ(provider_options.enable_musa_graph, 1);
}

TEST(MusaGraphTest, ParseEnableMusaGraphProviderOptions) {
  {
    const ProviderOptions options{{"enable_musa_graph", "true"}};
    EXPECT_TRUE(MusaExecutionProviderInfo::FromProviderOptions(options).enable_musa_graph);
  }
  {
    const ProviderOptions options{{"enable_musa_graph", "false"}};
    EXPECT_FALSE(MusaExecutionProviderInfo::FromProviderOptions(options).enable_musa_graph);
  }
  {
    const ProviderOptions options{{"enable_musa_graph", "1"}};
    EXPECT_TRUE(MusaExecutionProviderInfo::FromProviderOptions(options).enable_musa_graph);
  }
  {
    const ProviderOptions options{{"enable_musa_graph", "0"}};
    EXPECT_FALSE(MusaExecutionProviderInfo::FromProviderOptions(options).enable_musa_graph);
  }
}

TEST(MusaGraphTest, BasicCaptureReplay) {
  OrtMUSAProviderOptions provider_options{};
  provider_options.enable_musa_graph = 1;

  auto musa_provider = MusaExecutionProviderWithOptions(&provider_options);
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  Model model("musa_graph_basic_capture_replay", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  ONNX_NAMESPACE::TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);

  auto& x_arg = graph.GetOrCreateNodeArg("X", &float_tensor);
  auto& y_arg = graph.GetOrCreateNodeArg("Y", &float_tensor);
  auto& add_out = graph.GetOrCreateNodeArg("AddOut", &float_tensor);
  auto& z_arg = graph.GetOrCreateNodeArg("Z", &float_tensor);

  std::vector<NodeArg*> add_inputs{&x_arg, &y_arg};
  std::vector<NodeArg*> add_outputs{&add_out};
  std::vector<NodeArg*> relu_inputs{&add_out};
  std::vector<NodeArg*> relu_outputs{&z_arg};

  graph.AddNode("add_node", "Add", "Add node", add_inputs, add_outputs);
  graph.AddNode("relu_node", "Relu", "Relu node", relu_inputs, relu_outputs);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  const std::string model_file_name = "musa_graph_basic_capture_replay_test.onnx";
  status = Model::Save(model, model_file_name);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  SessionOptions session_options;
  session_options.session_logid = "MusaGraphTest.BasicCaptureReplay";
  RunOptions run_options;
  run_options.run_tag = session_options.session_logid;

  InferenceSession session_object{session_options, GetEnvironment()};
  status = session_object.RegisterExecutionProvider(std::move(musa_provider));
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  status = session_object.Load(model_file_name);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  status = session_object.Initialize();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  const std::vector<int64_t> dims = {2, 3};
  std::vector<std::string> output_names{"Z"};

  auto run_and_verify = [&](std::vector<float> x_values,
                            std::vector<float> y_values,
                            const std::vector<float>& expected_values) {
    OrtValue x_value;
    OrtValue y_value;
    CreateMLValue<float>(gsl::make_span(dims), x_values.data(), OrtMemoryInfo(), &x_value);
    CreateMLValue<float>(gsl::make_span(dims), y_values.data(), OrtMemoryInfo(), &y_value);

    NameMLValMap feeds;
    feeds.insert(std::make_pair("X", x_value));
    feeds.insert(std::make_pair("Y", y_value));

    std::vector<OrtValue> fetches;
    auto run_status = session_object.Run(run_options, feeds, output_names, &fetches);
    ASSERT_TRUE(run_status.IsOK()) << run_status.ErrorMessage();
    VerifyFloatOutput(fetches, dims, expected_values);
  };

  run_and_verify({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
                 {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
                 {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f});

  run_and_verify({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
                 {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
                 {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f});

  run_and_verify({2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f},
                 {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f},
                 {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f});
}

TEST(MusaGraphTest, GridSampleOnnxDomainTransformsToNhwc) {
  RunGridSampleNhwcTransformTest(kOnnxDomain,
                                 "musa_graph_gridsample_onnx_test.onnx",
                                 "musa_graph_gridsample_onnx_optimized.onnx");
}

TEST(MusaGraphTest, GridSampleContribDomainTransformsToNhwc) {
  RunGridSampleNhwcTransformTest(kMSDomain,
                                 "musa_graph_gridsample_contrib_test.onnx",
                                 "musa_graph_gridsample_contrib_optimized.onnx");
}

}  // namespace test
}  // namespace onnxruntime
