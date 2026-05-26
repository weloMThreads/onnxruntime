// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "core/session/inference_session.h"
#include "core/session/environment.h"
#include "core/graph/model.h"
#include "test/framework/test_utils.h"
#include "gtest/gtest.h"
#include <chrono>
#include <random>
#include <iomanip>

namespace onnxruntime {
namespace test {

class MusaGemmInferenceTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Test will skip if MUSA provider is not available
  }
};

// Helper function to create random input data
std::vector<float> CreateRandomInputData(int64_t size, float min_val = -1.0f, float max_val = 1.0f) {
  std::vector<float> data(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(min_val, max_val);

  for (int64_t i = 0; i < size; ++i) {
    data[i] = dis(gen);
  }

  return data;
}

// Helper function to create weight data (initialized with small random values)
std::vector<float> CreateWeightData(int64_t size) {
  return CreateRandomInputData(size, -0.1f, 0.1f);
}

TEST_F(MusaGemmInferenceTest, GemmInferencePerformanceTest) {
  // Check if MUSA provider is available
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  // Model dimensions
  const int64_t batch_size = 8;
  const int64_t input_features = 5120;
  const int64_t output_features = 256;
  const int num_runs = 10;

  // Create ONNX model in memory
  onnxruntime::Model model("gemm_inference_test", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  // Create type for float tensor
  ONNX_NAMESPACE::TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  // Input tensor type [batch_size, input_features]
  ONNX_NAMESPACE::TypeProto input_tensor_type = float_tensor;
  input_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(batch_size);
  input_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(input_features);

  // Weight tensor type [input_features, output_features]
  ONNX_NAMESPACE::TypeProto weight_tensor_type = float_tensor;
  weight_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(input_features);
  weight_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(output_features);

  // Output tensor type [batch_size, output_features]
  ONNX_NAMESPACE::TypeProto output_tensor_type = float_tensor;
  output_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(batch_size);
  output_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(output_features);

  // Create input and weight node args
  auto& input_arg = graph.GetOrCreateNodeArg("feats", &input_tensor_type);
  auto& weight_arg = graph.GetOrCreateNodeArg("weight", &weight_tensor_type);
  auto& output_arg = graph.GetOrCreateNodeArg("embs", &output_tensor_type);

  // Create weight initializer
  ONNX_NAMESPACE::TensorProto weight_tensor;
  weight_tensor.set_name("weight");
  weight_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  weight_tensor.add_dims(input_features);
  weight_tensor.add_dims(output_features);

  // Create random weight data
  auto weight_data = CreateWeightData(input_features * output_features);
  weight_tensor.mutable_raw_data()->assign(
    reinterpret_cast<const char*>(weight_data.data()),
    weight_data.size() * sizeof(float)
  );

  graph.AddInitializedTensor(weight_tensor);

  // Add Gemm node
  std::vector<onnxruntime::NodeArg*> gemm_inputs;
  std::vector<onnxruntime::NodeArg*> gemm_outputs;
  gemm_inputs.push_back(&input_arg);
  gemm_inputs.push_back(&weight_arg);
  gemm_outputs.push_back(&output_arg);

  auto& gemm_node = graph.AddNode("gemm_node", "Gemm", "Gemm operation", gemm_inputs, gemm_outputs);

  // Set Gemm attributes
  gemm_node.AddAttribute("transA", (int64_t)0);
  gemm_node.AddAttribute("transB", (int64_t)0);
  gemm_node.AddAttribute("alpha", 1.0f);
  gemm_node.AddAttribute("beta", 0.0f);

  // Resolve graph
  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << "Graph resolve failed: " << status.ErrorMessage();

  // Save model to file
  std::string model_file_name = "musa_gemm_inference_test.onnx";
  status = onnxruntime::Model::Save(model, model_file_name);
  ASSERT_TRUE(status.IsOK()) << "Model save failed: " << status.ErrorMessage();

  // Create session
  SessionOptions so;
  so.session_logid = "MusaGemmInferenceTest.GemmInferencePerformanceTest";
  InferenceSession session_object{so, GetEnvironment()};

  // Register MUSA provider
  status = session_object.RegisterExecutionProvider(std::move(musa_provider));
  ASSERT_TRUE(status.IsOK()) << "Register MUSA provider failed: " << status.ErrorMessage();

  // Load and initialize session
  status = session_object.Load(model_file_name);
  ASSERT_TRUE(status.IsOK()) << "Load model failed: " << status.ErrorMessage();

  status = session_object.Initialize();
  ASSERT_TRUE(status.IsOK()) << "Initialize session failed: " << status.ErrorMessage();

  // Create random input data
  auto input_data = CreateRandomInputData(batch_size * input_features);

  // Create OrtValue for input
  std::vector<int64_t> input_dims = {batch_size, input_features};
  OrtValue input_ml_value;
  CreateMLValue<float>(gsl::make_span(input_dims), input_data.data(), OrtMemoryInfo(), &input_ml_value);

  // Prepare feeds
  NameMLValMap feeds;
  feeds.insert(std::make_pair("feats", input_ml_value));

  // Prepare output names
  std::vector<std::string> output_names;
  output_names.push_back("embs");

  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  // Warm-up run
  std::vector<OrtValue> fetches;
  status = session_object.Run(run_options, feeds, output_names, &fetches);
  ASSERT_TRUE(status.IsOK()) << "Warm-up run failed: " << status.ErrorMessage();

  // Performance measurement
  std::vector<double> inference_times;
  inference_times.reserve(num_runs);

  std::cout << "\n=== MUSA Gemm Inference Performance Test ===\n";
  std::cout << "Model: Input[" << batch_size << ", " << input_features << "] -> Output["
            << batch_size << ", " << output_features << "]\n";
  std::cout << "Number of runs: " << num_runs << "\n\n";

  for (int i = 0; i < num_runs; ++i) {
    // Clear previous results
    fetches.clear();

    // Create random input data
    auto input_data = CreateRandomInputData(batch_size * input_features);

    // Create OrtValue for input
    std::vector<int64_t> input_dims = {batch_size, input_features};
    OrtValue input_ml_value;
    CreateMLValue<float>(gsl::make_span(input_dims), input_data.data(), OrtMemoryInfo(), &input_ml_value);

    // Prepare feeds
    NameMLValMap feeds;
    feeds.insert(std::make_pair("feats", input_ml_value));


    // Measure inference time
    auto start_time = std::chrono::high_resolution_clock::now();

    status = session_object.Run(run_options, feeds, output_names, &fetches);

    auto end_time = std::chrono::high_resolution_clock::now();

    ASSERT_TRUE(status.IsOK()) << "Inference run " << i + 1 << " failed: " << status.ErrorMessage();
    ASSERT_EQ(fetches.size(), 1) << "Expected 1 output, got " << fetches.size();

    // Calculate inference time in milliseconds
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    double time_ms = duration.count() / 1000.0;
    inference_times.push_back(time_ms);

    std::cout << "Run " << std::setw(2) << (i + 1) << ": "
              << std::fixed << std::setprecision(3) << time_ms << " ms\n";

    // Verify output shape
    const auto& output_tensor = fetches[0].Get<Tensor>();
    const auto& output_shape = output_tensor.Shape();
    ASSERT_EQ(output_shape.NumDimensions(), 2) << "Output should be 2D";
    ASSERT_EQ(output_shape[0], batch_size) << "Output batch size mismatch";
    ASSERT_EQ(output_shape[1], output_features) << "Output feature size mismatch";
  }

  // Calculate statistics
  double total_time = 0.0;
  double min_time = inference_times[0];
  double max_time = inference_times[0];

  for (double time : inference_times) {
    total_time += time;
    min_time = std::min(min_time, time);
    max_time = std::max(max_time, time);
  }

  double avg_time = total_time / num_runs;

  // Calculate standard deviation
  double variance = 0.0;
  for (double time : inference_times) {
    variance += (time - avg_time) * (time - avg_time);
  }
  variance /= num_runs;
  double std_dev = std::sqrt(variance);

  // Print summary
  std::cout << "\n=== Performance Summary ===\n";
  std::cout << "Average time: " << std::fixed << std::setprecision(3) << avg_time << " ms\n";
  std::cout << "Min time:     " << std::fixed << std::setprecision(3) << min_time << " ms\n";
  std::cout << "Max time:     " << std::fixed << std::setprecision(3) << max_time << " ms\n";
  std::cout << "Std dev:      " << std::fixed << std::setprecision(3) << std_dev << " ms\n";
  std::cout << "Total time:   " << std::fixed << std::setprecision(3) << total_time << " ms\n";

  // Calculate throughput
  int64_t total_operations = static_cast<int64_t>(batch_size) * input_features * output_features * 2; // multiply-add
  double avg_throughput = total_operations / (avg_time / 1000.0) / 1e9; // GFLOPS
  std::cout << "Throughput:   " << std::fixed << std::setprecision(2) << avg_throughput << " GFLOPS\n";

  std::cout << "\nTest completed successfully!\n";

  // Clean up model file
  std::remove(model_file_name.c_str());
}

} // namespace test
} // namespace onnxruntime
