// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <atomic>
#include <cmath>
#include <mutex>
#include <thread>
#include <vector>

#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"
#include "test/framework/test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Test to reproduce MusaEP concurrent execution issue
// Issue: When multiple threads share the same session with different input shapes,
// random shape mismatch errors occur due to missing PerThreadContext mechanism.
//
// The original issue manifests as:
// - MUSA kernel outputs getting cross-contaminated between threads
// - CPU operators (like Reshape) detecting shape mismatches when they receive wrong tensors
// - Random errors in Reshape, MatMul, GatherElements etc.

class MusaEPMultiThreadTest : public ::testing::Test {
 protected:
  // Helper to create a mixed MUSA+CPU graph model
  // Graph: Input[batch, 6] -> Reshape(CPU) -> [batch, 2, 3] -> Add(MUSA) -> [batch, 2, 3] -> Reshape(CPU) -> [batch*6]
  // This ensures we have CPU operators that will validate shapes and catch any cross-contamination
  static std::string CreateMixedEPModel(int test_id) {
    onnxruntime::Model model("multithread_mixed_ep_test", false,
                             DefaultLoggingManager().DefaultLogger());
    auto& graph = model.MainGraph();

    // Input tensor: [batch, 6] - dynamic batch
    ONNX_NAMESPACE::TypeProto input_tensor;
    input_tensor.mutable_tensor_type()->set_elem_type(
        ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    input_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param("batch");
    input_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(6);

    // Intermediate tensor after first reshape: [batch, 2, 3]
    ONNX_NAMESPACE::TypeProto reshaped_tensor;
    reshaped_tensor.mutable_tensor_type()->set_elem_type(
        ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    reshaped_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param("batch");
    reshaped_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);
    reshaped_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);

    // Shape tensor for reshape
    ONNX_NAMESPACE::TypeProto shape_tensor;
    shape_tensor.mutable_tensor_type()->set_elem_type(
        ONNX_NAMESPACE::TensorProto_DataType_INT64);
    shape_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);

    // Output tensor: [batch * 6] flattened
    ONNX_NAMESPACE::TypeProto output_tensor;
    output_tensor.mutable_tensor_type()->set_elem_type(
        ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    output_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param("output_size");

    // Create nodes
    auto& input_arg = graph.GetOrCreateNodeArg("input", &input_tensor);
    auto& bias_arg = graph.GetOrCreateNodeArg("bias", &reshaped_tensor);

    // Shape for first reshape: [-1, 2, 3]
    auto& shape1_arg = graph.GetOrCreateNodeArg("shape1", &shape_tensor);
    ONNX_NAMESPACE::TensorProto shape1_init;
    shape1_init.set_name("shape1");
    shape1_init.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
    shape1_init.add_dims(3);
    shape1_init.add_int64_data(-1);  // infer batch
    shape1_init.add_int64_data(2);
    shape1_init.add_int64_data(3);
    graph.AddInitializedTensor(shape1_init);

    // Bias initializer: [1, 2, 3] broadcast to [batch, 2, 3]
    ONNX_NAMESPACE::TensorProto bias_init;
    bias_init.set_name("bias");
    bias_init.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    bias_init.add_dims(1);
    bias_init.add_dims(2);
    bias_init.add_dims(3);
    for (int i = 0; i < 6; i++) {
      bias_init.add_float_data(0.1f * (i + 1));  // 0.1, 0.2, 0.3, 0.4, 0.5, 0.6
    }
    graph.AddInitializedTensor(bias_init);

    // Shape for final reshape: [-1]
    ONNX_NAMESPACE::TypeProto shape2_type;
    shape2_type.mutable_tensor_type()->set_elem_type(
        ONNX_NAMESPACE::TensorProto_DataType_INT64);
    shape2_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    auto& shape2_arg = graph.GetOrCreateNodeArg("shape2", &shape2_type);
    ONNX_NAMESPACE::TensorProto shape2_init;
    shape2_init.set_name("shape2");
    shape2_init.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
    shape2_init.add_dims(1);
    shape2_init.add_int64_data(-1);  // flatten
    graph.AddInitializedTensor(shape2_init);

    // Node 1: Reshape (CPU) - Input[batch, 6] -> [batch, 2, 3]
    auto& reshape1_out = graph.GetOrCreateNodeArg("reshape1_out", &reshaped_tensor);
    graph.AddNode("reshape1", "Reshape", "Reshape to 3D (CPU)",
                  {&input_arg, &shape1_arg}, {&reshape1_out});

    // Node 2: Add (MUSA) - [batch, 2, 3] + bias -> [batch, 2, 3]
    auto& add_out = graph.GetOrCreateNodeArg("add_out", &reshaped_tensor);
    graph.AddNode("add_node", "Add", "Add bias (MUSA)",
                  {&reshape1_out, &bias_arg}, {&add_out});

    // Node 3: Reshape (CPU) - [batch, 2, 3] -> [batch * 6]
    // This will detect shape mismatch if wrong tensor is passed
    auto& output_arg = graph.GetOrCreateNodeArg("output", &output_tensor);
    graph.AddNode("reshape2", "Reshape", "Flatten output (CPU)",
                  {&add_out, &shape2_arg}, {&output_arg});

    auto status = graph.Resolve();
    ORT_ENFORCE(status.IsOK(), "Failed to resolve graph: ", status.ErrorMessage());

    // Use unique filename with test_id to avoid conflicts
    std::string model_file_name = "musa_multithread_test_" + std::to_string(test_id) + "_" +
                                  std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())) +
                                  ".onnx";
    status = onnxruntime::Model::Save(model, model_file_name);
    ORT_ENFORCE(status.IsOK(), "Failed to save model: ", status.ErrorMessage());

    return model_file_name;
  }

  // Worker function for each thread
  // Each thread uses different batch_size and input_value to detect cross-contamination
  static void RunInferenceWorker(InferenceSession& session,
                                 int batch_size,
                                 float input_value,  // Different value per thread to detect mixing
                                 int thread_id,
                                 int iterations,
                                 std::atomic<int>& error_count,
                                 std::vector<std::string>& error_messages,
                                 std::mutex& error_mutex) {
    RunOptions run_options;
    run_options.run_tag = "thread_" + std::to_string(thread_id);

    // Input shape: [batch_size, 6]
    std::vector<int64_t> input_dims = {batch_size, 6};
    int64_t input_size = batch_size * 6;
    std::vector<float> input_data(input_size, input_value);

    // Expected output: input + bias (broadcast)
    // bias = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    std::vector<float> expected_output;
    expected_output.reserve(input_size);
    for (int b = 0; b < batch_size; b++) {
      for (int i = 0; i < 6; i++) {
        expected_output.push_back(input_value + 0.1f * (i + 1));
      }
    }

    for (int iter = 0; iter < iterations; ++iter) {
      // Create input tensor
      OrtValue input_tensor;
      CreateMLValue<float>(gsl::make_span(input_dims), input_data.data(),
                           OrtMemoryInfo(), &input_tensor);

      NameMLValMap feeds;
      feeds.insert(std::make_pair("input", input_tensor));

      std::vector<std::string> output_names = {"output"};
      std::vector<OrtValue> fetches;

      auto status = session.Run(run_options, feeds, output_names, &fetches);

      if (!status.IsOK()) {
        error_count++;
        std::string error_msg = "Thread " + std::to_string(thread_id) +
                                " (batch=" + std::to_string(batch_size) +
                                ", value=" + std::to_string(input_value) +
                                ") Iteration " + std::to_string(iter) +
                                " Run failed: " + status.ErrorMessage();
        std::lock_guard<std::mutex> lock(error_mutex);
        error_messages.push_back(error_msg);
        continue;
      }

      // Verify output shape
      const auto& output_tensor = fetches[0].Get<Tensor>();
      int64_t expected_output_size = batch_size * 6;

      if (output_tensor.Shape().Size() != expected_output_size) {
        error_count++;
        std::string error_msg = "Thread " + std::to_string(thread_id) +
                                " Iteration " + std::to_string(iter) +
                                " Shape mismatch: expected size " +
                                std::to_string(expected_output_size) + ", got " +
                                std::to_string(output_tensor.Shape().Size());
        std::lock_guard<std::mutex> lock(error_mutex);
        error_messages.push_back(error_msg);
        continue;
      }

      // Verify output values to detect cross-contamination
      const float* output_data = output_tensor.Data<float>();
      bool value_mismatch = false;
      std::string mismatch_details;

      for (int64_t i = 0; i < expected_output_size; i++) {
        float diff = std::abs(output_data[i] - expected_output[i]);
        if (diff > 1e-5f) {
          value_mismatch = true;
          if (mismatch_details.empty()) {
            mismatch_details = "First mismatch at index " + std::to_string(i) +
                               ": expected " + std::to_string(expected_output[i]) +
                               ", got " + std::to_string(output_data[i]);
          }
        }
      }

      if (value_mismatch) {
        error_count++;
        std::string error_msg = "Thread " + std::to_string(thread_id) +
                                " (batch=" + std::to_string(batch_size) +
                                ", value=" + std::to_string(input_value) +
                                ") Iteration " + std::to_string(iter) +
                                " Value mismatch (possible cross-contamination): " +
                                mismatch_details;
        std::lock_guard<std::mutex> lock(error_mutex);
        error_messages.push_back(error_msg);
      }
    }
  }
};

// Test: Multiple threads with different batch sizes sharing the same session
TEST_F(MusaEPMultiThreadTest, ConcurrentInferenceWithDifferentBatchSizes) {
  // Skip if MUSA EP is not available
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider is not available";
  }

  // Create model with unique ID
  std::string model_file = CreateMixedEPModel(1);

  // Create session
  SessionOptions so;
  so.session_logid = "MusaEPMultiThreadTest.ConcurrentInference";

  InferenceSession session{so, GetEnvironment()};
  auto status = session.RegisterExecutionProvider(std::move(musa_provider));
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  status = session.Load(model_file);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  status = session.Initialize();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  // Test configuration
  const int num_threads = 2;
  const int iterations_per_thread = 50;

  // Different batch sizes and input values for different threads
  // Thread 0: batch=1, input_value=1.0
  // Thread 1: batch=4, input_value=10.0
  // The different values help detect cross-contamination
  std::vector<int> batch_sizes = {1, 4};
  std::vector<float> input_values = {1.0f, 10.0f};

  std::atomic<int> error_count{0};
  std::vector<std::string> error_messages;
  std::mutex error_mutex;
  std::vector<std::thread> threads;

  // Launch threads
  for (int i = 0; i < num_threads; i++) {
    threads.push_back(std::thread(
        RunInferenceWorker,
        std::ref(session),
        batch_sizes[i],
        input_values[i],
        i,
        iterations_per_thread,
        std::ref(error_count),
        std::ref(error_messages),
        std::ref(error_mutex)));
  }

  // Wait for all threads to complete
  for (auto& th : threads) {
    th.join();
  }

  // Report errors if any
  if (error_count > 0) {
    std::cerr << "\n=== MusaEP Concurrency Test Errors ===" << std::endl;
    std::cerr << "Total errors: " << error_count << " / "
              << (num_threads * iterations_per_thread) << std::endl;
    for (const auto& msg : error_messages) {
      std::cerr << "  " << msg << std::endl;
    }
    std::cerr << "======================================\n" << std::endl;
  }

  // Expect all runs to succeed
  EXPECT_EQ(error_count, 0) << "Concurrent inference with different batch sizes failed. "
                            << "This indicates PerThreadContext issue in MusaEP.";

  // Cleanup
  std::remove(model_file.c_str());
}

// Test: Higher concurrency stress test with more threads and varied configurations
TEST_F(MusaEPMultiThreadTest, HighConcurrencyStressTest) {
  // Skip if MUSA EP is not available
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider is not available";
  }

  // Create model with unique ID
  std::string model_file = CreateMixedEPModel(2);

  // Create session
  SessionOptions so;
  so.session_logid = "MusaEPMultiThreadTest.StressTest";

  InferenceSession session{so, GetEnvironment()};
  auto status = session.RegisterExecutionProvider(std::move(musa_provider));
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  status = session.Load(model_file);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  status = session.Initialize();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  // Higher concurrency configuration
  const int num_threads = 4;
  const int iterations_per_thread = 100;

  // Varied batch sizes and input values
  std::vector<int> batch_sizes = {1, 2, 4, 8};
  std::vector<float> input_values = {1.0f, 5.0f, 10.0f, 20.0f};

  std::atomic<int> error_count{0};
  std::vector<std::string> error_messages;
  std::mutex error_mutex;
  std::vector<std::thread> threads;

  // Launch threads
  for (int i = 0; i < num_threads; i++) {
    threads.push_back(std::thread(
        RunInferenceWorker,
        std::ref(session),
        batch_sizes[i],
        input_values[i],
        i,
        iterations_per_thread,
        std::ref(error_count),
        std::ref(error_messages),
        std::ref(error_mutex)));
  }

  // Wait for all threads to complete
  for (auto& th : threads) {
    th.join();
  }

  // Report errors if any
  if (error_count > 0) {
    std::cerr << "\n=== MusaEP High Concurrency Stress Test Errors ===" << std::endl;
    std::cerr << "Total errors: " << error_count << " / "
              << (num_threads * iterations_per_thread) << std::endl;
    // Only print first 10 errors to avoid flooding output
    int print_count = std::min(10, static_cast<int>(error_messages.size()));
    for (int i = 0; i < print_count; i++) {
      std::cerr << "  " << error_messages[i] << std::endl;
    }
    if (static_cast<int>(error_messages.size()) > 10) {
      std::cerr << "  ... and " << (error_messages.size() - 10) << " more errors" << std::endl;
    }
    std::cerr << "=================================================\n" << std::endl;
  }

  // Expect all runs to succeed
  EXPECT_EQ(error_count, 0) << "High concurrency stress test failed. "
                            << "This indicates PerThreadContext issue in MusaEP.";

  // Cleanup
  std::remove(model_file.c_str());
}

}  // namespace test
}  // namespace onnxruntime
