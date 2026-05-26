// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <dlfcn.h>
#include <musa_runtime.h>

#include "core/graph/constants.h"
#include "core/graph/model.h"
#include "core/graph/onnx_protobuf.h"
#include "core/providers/musa/musa_provider_options.h"
#include "core/session/environment.h"
#include "core/session/inference_session.h"
#include "test/framework/test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

namespace {

constexpr const char* kRequiresTwoDevices = "requires at least 2 MUSA devices";

struct MusaRuntimeApi {
  using GetDeviceCountFn = musaError_t (*)(int*);
  using SetDeviceFn = musaError_t (*)(int);
  using PointerGetAttributesFn = musaError_t (*)(musaPointerAttributes*, const void*);

  GetDeviceCountFn get_device_count{};
  SetDeviceFn set_device{};
  PointerGetAttributesFn pointer_get_attributes{};
};

void* ResolveMusaSymbol(const char* name) {
  void* symbol = dlsym(RTLD_DEFAULT, name);
  if (symbol != nullptr) {
    return symbol;
  }

  for (const char* library_name : {"libmusa.so", "libmusart.so"}) {
    void* handle = dlopen(library_name, RTLD_LAZY | RTLD_GLOBAL);
    if (handle != nullptr) {
      symbol = dlsym(handle, name);
      if (symbol != nullptr) {
        return symbol;
      }
    }
  }

  return nullptr;
}

const MusaRuntimeApi& GetMusaRuntimeApi() {
  static const MusaRuntimeApi api{
      reinterpret_cast<MusaRuntimeApi::GetDeviceCountFn>(ResolveMusaSymbol("musaGetDeviceCount")),
      reinterpret_cast<MusaRuntimeApi::SetDeviceFn>(ResolveMusaSymbol("musaSetDevice")),
      reinterpret_cast<MusaRuntimeApi::PointerGetAttributesFn>(ResolveMusaSymbol("musaPointerGetAttributes"))};
  ORT_ENFORCE(api.get_device_count != nullptr, "Unable to resolve musaGetDeviceCount");
  ORT_ENFORCE(api.set_device != nullptr, "Unable to resolve musaSetDevice");
  ORT_ENFORCE(api.pointer_get_attributes != nullptr, "Unable to resolve musaPointerGetAttributes");
  return api;
}

void ExpectMusaSuccess(musaError_t status, const char* expression) {
  EXPECT_EQ(status, musaSuccess) << expression << " failed with status "
                                 << static_cast<int>(status);
}

int GetDeviceCount() {
  int count = 0;
  auto status = GetMusaRuntimeApi().get_device_count(&count);
  if (status != musaSuccess || count <= 0) {
    return 0;
  }
  return count;
}

std::vector<int> PickDevices(int count) {
  std::vector<int> ids{0};
  if (count > 1) {
    ids.push_back(count - 1);
  }
  return ids;
}

void ExpectPointerOnDevice(void* ptr, int expected_device) {
  musaPointerAttributes attrs{};
  ASSERT_EQ(GetMusaRuntimeApi().pointer_get_attributes(&attrs, ptr), musaSuccess);
  EXPECT_EQ(attrs.type, musaMemoryTypeDevice);
  EXPECT_EQ(attrs.device, expected_device);
}

std::unique_ptr<IExecutionProvider> CreateMusaProvider(int device_id, bool prefer_nhwc = false) {
  OrtMUSAProviderOptions provider_options{};
  provider_options.device_id = device_id;
  provider_options.prefer_nhwc = prefer_nhwc;
  auto provider = MusaExecutionProviderWithOptions(&provider_options);
  ORT_ENFORCE(provider != nullptr, "MUSA execution provider is unavailable");
  return provider;
}

AllocatorPtr GetAllocatorForDevice(IExecutionProvider& provider, OrtDevice::DeviceType device_type) {
  auto allocators = provider.CreatePreferredAllocators();
  for (auto& allocator : allocators) {
    if (allocator && allocator->Info().device.Type() == device_type) {
      return allocator;
    }
  }
  ORT_THROW("Expected allocator for device type ", device_type);
}

std::string CreateAddModel(const std::string& model_file_name) {
  Model model("musa_multi_device_add_graph", false,
              DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  ONNX_NAMESPACE::TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(4);

  auto& input_a = graph.GetOrCreateNodeArg("A", &float_tensor);
  auto& input_b = graph.GetOrCreateNodeArg("B", &float_tensor);
  auto& output_c = graph.GetOrCreateNodeArg("C", &float_tensor);
  graph.AddNode("add_node", "Add", "Add node", {&input_a, &input_b}, {&output_c});

  auto status = graph.Resolve();
  ORT_ENFORCE(status.IsOK(), "Failed to resolve add graph: ", status.ErrorMessage());

  status = Model::Save(model, model_file_name);
  ORT_ENFORCE(status.IsOK(), "Failed to save add graph: ", status.ErrorMessage());
  return model_file_name;
}

std::string CreateConvModel(const std::string& model_file_name) {
  Model model("musa_multi_device_conv_graph", false,
              DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  ONNX_NAMESPACE::TypeProto input_tensor;
  input_tensor.mutable_tensor_type()->set_elem_type(
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  for (int64_t dim : {1, 5, 5, 1}) {
    input_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
  }

  ONNX_NAMESPACE::TypeProto weight_tensor_type;
  weight_tensor_type.mutable_tensor_type()->set_elem_type(
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  for (int64_t dim : {1, 1, 3, 3}) {
    weight_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
  }

  ONNX_NAMESPACE::TypeProto output_tensor;
  output_tensor.mutable_tensor_type()->set_elem_type(
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  for (int64_t dim : {1, 3, 3, 1}) {
    output_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
  }

  auto& input_x = graph.GetOrCreateNodeArg("X", &input_tensor);
  auto& weight_w = graph.GetOrCreateNodeArg("W", &weight_tensor_type);
  auto& output_y = graph.GetOrCreateNodeArg("Y", &output_tensor);

  ONNX_NAMESPACE::TensorProto weight_tensor;
  weight_tensor.set_name("W");
  weight_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  for (int64_t dim : {1, 1, 3, 3}) {
    weight_tensor.add_dims(dim);
  }
  for (int i = 0; i < 9; ++i) {
    weight_tensor.add_float_data(1.0f);
  }
  graph.AddInitializedTensor(weight_tensor);

  auto& conv_node = graph.AddNode("conv_node", "Conv", "Conv node", {&input_x, &weight_w}, {&output_y},
                                  nullptr, kMSInternalNHWCDomain);
  conv_node.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
  conv_node.AddAttribute("strides", std::vector<int64_t>{1, 1});
  conv_node.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});

  auto status = graph.Resolve();
  ORT_ENFORCE(status.IsOK(), "Failed to resolve conv graph: ", status.ErrorMessage());

  status = Model::Save(model, model_file_name);
  ORT_ENFORCE(status.IsOK(), "Failed to save conv graph: ", status.ErrorMessage());
  return model_file_name;
}

std::unique_ptr<InferenceSession> CreateAddSession(const std::string& model_file_name,
                                                   int device_id,
                                                   const std::string& log_id) {
  SessionOptions so;
  so.session_logid = log_id;

  auto session = std::make_unique<InferenceSession>(so, GetEnvironment());
  auto provider = CreateMusaProvider(device_id);

  auto status = session->RegisterExecutionProvider(std::move(provider));
  ORT_ENFORCE(status.IsOK(), "RegisterExecutionProvider failed: ", status.ErrorMessage());
  status = session->Load(model_file_name);
  ORT_ENFORCE(status.IsOK(), "Session Load failed: ", status.ErrorMessage());
  status = session->Initialize();
  ORT_ENFORCE(status.IsOK(), "Session Initialize failed: ", status.ErrorMessage());
  return session;
}

std::unique_ptr<InferenceSession> CreateConvNhwcSession(const std::string& model_file_name,
                                                        int device_id,
                                                        const std::string& log_id) {
  SessionOptions so;
  so.session_logid = log_id;

  auto session = std::make_unique<InferenceSession>(so, GetEnvironment());
  auto provider = CreateMusaProvider(device_id, true);

  auto status = session->RegisterExecutionProvider(std::move(provider));
  ORT_ENFORCE(status.IsOK(), "RegisterExecutionProvider failed: ", status.ErrorMessage());
  status = session->Load(model_file_name);
  ORT_ENFORCE(status.IsOK(), "Session Load failed: ", status.ErrorMessage());
  ExpectMusaSuccess(GetMusaRuntimeApi().set_device(0), "musaSetDevice(0)");
  status = session->Initialize();
  ORT_ENFORCE(status.IsOK(), "Session Initialize failed: ", status.ErrorMessage());
  return session;
}

void RunAddAndVerify(InferenceSession& session, const std::string& run_tag) {
  RunOptions run_options;
  run_options.run_tag = run_tag;

  std::vector<int64_t> dims = {4};
  std::vector<float> values_a = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> values_b = {10.0f, 20.0f, 30.0f, 40.0f};
  std::vector<float> expected = {11.0f, 22.0f, 33.0f, 44.0f};

  OrtValue input_a;
  OrtValue input_b;
  CreateMLValue<float>(gsl::make_span(dims), values_a.data(), OrtMemoryInfo(), &input_a);
  CreateMLValue<float>(gsl::make_span(dims), values_b.data(), OrtMemoryInfo(), &input_b);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("A", input_a));
  feeds.insert(std::make_pair("B", input_b));

  std::vector<std::string> output_names{"C"};
  std::vector<OrtValue> fetches;
  auto status = session.Run(run_options, feeds, output_names, &fetches);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  ASSERT_EQ(fetches.size(), 1u);

  const auto& output_tensor = fetches.front().Get<Tensor>();
  ASSERT_EQ(output_tensor.Shape(), TensorShape(dims));
  const float* output_data = output_tensor.Data<float>();
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(output_data[i], expected[i], 1e-5f);
  }
}

void RunConvAndVerify(InferenceSession& session, const std::string& run_tag) {
  RunOptions run_options;
  run_options.run_tag = run_tag;

  std::vector<int64_t> dims = {1, 5, 5, 1};
  std::vector<float> values(25, 1.0f);

  OrtValue input_x;
  CreateMLValue<float>(gsl::make_span(dims), values.data(), OrtMemoryInfo(), &input_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", input_x));

  std::vector<std::string> output_names{"Y"};
  std::vector<OrtValue> fetches;
  auto status = session.Run(run_options, feeds, output_names, &fetches);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  ASSERT_EQ(fetches.size(), 1u);

  const auto& output_tensor = fetches.front().Get<Tensor>();
  ASSERT_EQ(output_tensor.Shape(), TensorShape({1, 3, 3, 1}));
}

}  // namespace

TEST(MusaMultiDeviceTest, T5_AllocatorAllocatesOnRequestedDevice) {
  int count = GetDeviceCount();
  if (count <= 0) {
    GTEST_SKIP() << "No MUSA device available";
  }
  ASSERT_GT(count, 0);

  for (int target_device : PickDevices(count)) {
    ExpectMusaSuccess(GetMusaRuntimeApi().set_device(0), "musaSetDevice(0)");
    auto provider = CreateMusaProvider(target_device);
    auto allocator = GetAllocatorForDevice(*provider, OrtDevice::GPU);
    void* ptr = allocator->Alloc(4096);
    ASSERT_NE(ptr, nullptr);
    ExpectPointerOnDevice(ptr, target_device);
    allocator->Free(ptr);
  }
}

TEST(MusaMultiDeviceTest, T6_AllocatorSetsDeviceFromWorkerThread) {
  int count = GetDeviceCount();
  if (count <= 0) {
    GTEST_SKIP() << "No MUSA device available";
  }
  if (count < 2) {
    GTEST_SKIP() << kRequiresTwoDevices;
  }

  auto provider = CreateMusaProvider(count - 1);
  auto allocator = GetAllocatorForDevice(*provider, OrtDevice::GPU);
  ExpectMusaSuccess(GetMusaRuntimeApi().set_device(0), "musaSetDevice(0)");

  musaError_t pointer_status = musaSuccess;
  musaPointerAttributes attrs{};
  bool pointer_checked = false;
  std::exception_ptr thread_exception;

  std::thread worker([&]() {
    try {
      void* ptr = allocator->Alloc(4096);
      pointer_status = GetMusaRuntimeApi().pointer_get_attributes(&attrs, ptr);
      pointer_checked = true;
      allocator->Free(ptr);
    } catch (...) {
      thread_exception = std::current_exception();
    }
  });
  worker.join();

  if (thread_exception) {
    std::rethrow_exception(thread_exception);
  }

  ASSERT_TRUE(pointer_checked);
  ASSERT_EQ(pointer_status, musaSuccess);
  EXPECT_EQ(attrs.type, musaMemoryTypeDevice);
  EXPECT_EQ(attrs.device, count - 1);
}

TEST(MusaMultiDeviceTest, T7_RuntimeDeviceCount) {
  int count = 0;
  ExpectMusaSuccess(GetMusaRuntimeApi().get_device_count(&count), "musaGetDeviceCount");
  EXPECT_GE(count, 1);
}

TEST(MusaMultiDeviceTest, T8_PinnedAllocatorSanity) {
  int count = GetDeviceCount();
  if (count <= 0) {
    GTEST_SKIP() << "No MUSA device available";
  }
  ASSERT_GT(count, 0);

  auto provider = CreateMusaProvider(0);
  auto allocator = GetAllocatorForDevice(*provider, OrtDevice::CPU);
  void* ptr = allocator->Alloc(4096);
  ASSERT_NE(ptr, nullptr);

  musaPointerAttributes attrs{};
  auto status = GetMusaRuntimeApi().pointer_get_attributes(&attrs, ptr);
  if (status == musaSuccess) {
    EXPECT_TRUE(attrs.type == musaMemoryTypeHost ||
                attrs.type == musaMemoryTypeUnregistered);
  }

  allocator->Free(ptr);
}

TEST(MusaMultiDeviceTest, T9_InvalidDeviceIdRejected) {
  int count = GetDeviceCount();
  if (count <= 0) {
    GTEST_SKIP() << "No MUSA device available";
  }
  if (count < 2) {
    GTEST_SKIP() << kRequiresTwoDevices;
  }

  OrtMUSAProviderOptions provider_options{};
  provider_options.device_id = count;

  try {
    auto provider = MusaExecutionProviderWithOptions(&provider_options);
    FAIL() << "Expected invalid MUSA device ID " << count << " to be rejected, got provider "
           << provider.get();
  } catch (const std::exception& ex) {
    std::string message = ex.what();
    EXPECT_NE(message.find("Invalid MUSA device ID"), std::string::npos) << message;
    EXPECT_NE(message.find(std::to_string(count)), std::string::npos) << message;
  }
}

TEST(MusaMultiDeviceTest, T10_TwoSessionsDifferentDevicesNormalRun) {
  int count = GetDeviceCount();
  if (count <= 0) {
    GTEST_SKIP() << "No MUSA device available";
  }
  if (count < 2) {
    GTEST_SKIP() << kRequiresTwoDevices;
  }

  const std::string model_file_name = "musa_multi_device_add_test.onnx";
  CreateAddModel(model_file_name);

  auto session0 = CreateAddSession(model_file_name, 0, "MusaMultiDeviceTest.Device0");
  auto session_last = CreateAddSession(model_file_name, count - 1, "MusaMultiDeviceTest.DeviceLast");

  for (int i = 0; i < 3; ++i) {
    RunAddAndVerify(*session0, "device0_run_" + std::to_string(i));
    RunAddAndVerify(*session_last, "device_last_run_" + std::to_string(i));
  }
}

TEST(MusaMultiDeviceTest, T11_ConvNhwcPrepackOnNonZeroDevice) {
  int count = GetDeviceCount();
  if (count <= 0) {
    GTEST_SKIP() << "No MUSA device available";
  }
  if (count < 2) {
    GTEST_SKIP() << kRequiresTwoDevices;
  }

  const std::string model_file_name = "musa_multi_device_conv_nhwc_test.onnx";
  CreateConvModel(model_file_name);

  auto session = CreateConvNhwcSession(model_file_name, 1, "MusaMultiDeviceTest.ConvNhwcDevice1");
  RunConvAndVerify(*session, "device1_conv_nhwc_run");
}

}  // namespace test
}  // namespace onnxruntime
