// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <mudnn.h>
#include <musa_runtime.h>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {
namespace {

struct ConvTransposeScanConfig {
  const char* name;
  std::vector<int64_t> x_shape;
  std::vector<int64_t> w_shape;
  std::vector<int64_t> y_shape;
  std::vector<int> pads;
  std::vector<int> strides;
  std::vector<int> dilations;
  int groups;
};

struct AlgorithmCase {
  const char* name;
  ::musa::dnn::Convolution::AlgorithmBwdData value;
};

const std::vector<AlgorithmCase>& BwdDataAlgorithms() {
  static const std::vector<AlgorithmCase> algorithms = {
      {"IMPLICIT_GEMM", ::musa::dnn::Convolution::AlgorithmBwdData::IMPLICIT_GEMM},
      {"WINOGRAD_NONFUSED", ::musa::dnn::Convolution::AlgorithmBwdData::WINOGRAD_NONFUSED},
      {"GEMM", ::musa::dnn::Convolution::AlgorithmBwdData::GEMM},
      {"DIRECT", ::musa::dnn::Convolution::AlgorithmBwdData::DIRECT},
  };
  return algorithms;
}

const char* StatusName(::musa::dnn::Status status) {
  switch (status) {
    case ::musa::dnn::Status::SUCCESS:
      return "SUCCESS";
    case ::musa::dnn::Status::INVALID_PARAMETER:
      return "INVALID_PARAMETER";
    case ::musa::dnn::Status::NOT_INITIALIZED:
      return "NOT_INITIALIZED";
    case ::musa::dnn::Status::ALLOC_FAILED:
      return "ALLOC_FAILED";
    case ::musa::dnn::Status::NOT_SUPPORTED:
      return "NOT_SUPPORTED";
    case ::musa::dnn::Status::INTERNAL_ERROR:
      return "INTERNAL_ERROR";
    case ::musa::dnn::Status::ARCH_MISMATCH:
      return "ARCH_MISMATCH";
    case ::musa::dnn::Status::EXECUTION_FAILED:
      return "EXECUTION_FAILED";
    default:
      return "UNKNOWN";
  }
}

size_t ElementCount(const std::vector<int64_t>& shape) {
  size_t count = 1;
  for (auto dim : shape) {
    count *= static_cast<size_t>(dim);
  }
  return count;
}

bool IsOneDimensionalConfig(const ConvTransposeScanConfig& config) {
  return config.x_shape.size() == 3;
}

std::vector<int64_t> OnnxPads(const ConvTransposeScanConfig& config) {
  std::vector<int64_t> pads;
  pads.reserve(config.pads.size() * 2);
  for (int pad : config.pads) {
    pads.push_back(static_cast<int64_t>(pad));
  }
  for (int pad : config.pads) {
    pads.push_back(static_cast<int64_t>(pad));
  }
  return pads;
}

std::vector<int64_t> ToInt64Vector(const std::vector<int>& values) {
  std::vector<int64_t> result;
  result.reserve(values.size());
  for (int value : values) {
    result.push_back(static_cast<int64_t>(value));
  }
  return result;
}

template <typename T>
::musa::dnn::Tensor::Type MusaTensorType();

template <>
::musa::dnn::Tensor::Type MusaTensorType<float>() {
  return ::musa::dnn::Tensor::Type::FLOAT;
}

template <>
::musa::dnn::Tensor::Type MusaTensorType<uint16_t>() {
  return ::musa::dnn::Tensor::Type::HALF;
}

::musa::dnn::Tensor::Format TensorFormat(size_t rank) {
  if (rank == 3) {
    return ::musa::dnn::Tensor::Format::NCW;
  }
  if (rank == 4) {
    return ::musa::dnn::Tensor::Format::NCHW;
  }
  return ::musa::dnn::Tensor::Format::NCDHW;
}

template <typename T>
bool SetupTensor(::musa::dnn::Tensor& tensor, void* data, const std::vector<int64_t>& shape) {
  if (tensor.SetType(MusaTensorType<T>()) != ::musa::dnn::Status::SUCCESS) {
    ADD_FAILURE() << "SetType failed";
    return false;
  }
  if (tensor.SetAddr(data) != ::musa::dnn::Status::SUCCESS) {
    ADD_FAILURE() << "SetAddr failed";
    return false;
  }
  if (tensor.SetFormat(TensorFormat(shape.size())) != ::musa::dnn::Status::SUCCESS) {
    ADD_FAILURE() << "SetFormat failed";
    return false;
  }
  if (tensor.SetNdInfo(static_cast<int>(shape.size()), shape.data()) != ::musa::dnn::Status::SUCCESS) {
    ADD_FAILURE() << "SetNdInfo failed";
    return false;
  }
  return true;
}

template <typename T>
bool RunOne(const ConvTransposeScanConfig& config, const AlgorithmCase& algo, double& latency_us,
            ::musa::dnn::Status& out_status) {
  void* x = nullptr;
  void* w = nullptr;
  void* y = nullptr;
  const size_t x_bytes = ElementCount(config.x_shape) * sizeof(T);
  const size_t w_bytes = ElementCount(config.w_shape) * sizeof(T);
  const size_t y_bytes = ElementCount(config.y_shape) * sizeof(T);

  auto free_buffers = [&]() {
    if (x != nullptr) {
      EXPECT_EQ(musaFree(x), musaSuccess);
    }
    if (w != nullptr) {
      EXPECT_EQ(musaFree(w), musaSuccess);
    }
    if (y != nullptr) {
      EXPECT_EQ(musaFree(y), musaSuccess);
    }
  };

  if (musaMalloc(&x, x_bytes) != musaSuccess || musaMalloc(&w, w_bytes) != musaSuccess ||
      musaMalloc(&y, y_bytes) != musaSuccess) {
    ADD_FAILURE() << "musaMalloc failed";
    free_buffers();
    return false;
  }
  if (musaMemset(x, 0, x_bytes) != musaSuccess || musaMemset(w, 0, w_bytes) != musaSuccess ||
      musaMemset(y, 0, y_bytes) != musaSuccess) {
    ADD_FAILURE() << "musaMemset failed";
    free_buffers();
    return false;
  }

  bool ok = false;
  try {
    ::musa::dnn::Handle handle;
    ::musa::dnn::Tensor x_tensor;
    ::musa::dnn::Tensor w_tensor;
    ::musa::dnn::Tensor y_tensor;
    if (!SetupTensor<T>(x_tensor, x, config.x_shape) || !SetupTensor<T>(w_tensor, w, config.w_shape) ||
        !SetupTensor<T>(y_tensor, y, config.y_shape)) {
      free_buffers();
      return false;
    }

    ::musa::dnn::Convolution conv;
    if (conv.SetGroups(config.groups) != ::musa::dnn::Status::SUCCESS ||
        conv.SetNdInfo(static_cast<int>(config.pads.size()), config.pads.data(), config.strides.data(),
                       config.dilations.data()) != ::musa::dnn::Status::SUCCESS ||
        conv.SetComputeMode(::musa::dnn::Convolution::ComputeMode::TENSOR) != ::musa::dnn::Status::SUCCESS) {
      ADD_FAILURE() << "Convolution setup failed";
      free_buffers();
      return false;
    }

    std::vector<void*> workspace_buffers;
    auto memory_allocator = [&workspace_buffers](size_t size) -> ::musa::dnn::MemoryHandler {
      if (size == 0) {
        return ::musa::dnn::MemoryHandler(nullptr, [](void*) {});
      }
      void* ptr = nullptr;
      if (musaMalloc(&ptr, size) != musaSuccess) {
        return ::musa::dnn::MemoryHandler(nullptr, [](void*) {});
      }
      workspace_buffers.push_back(ptr);
      return ::musa::dnn::MemoryHandler(ptr, [](void*) {});
    };
    ::musa::dnn::MemoryMaintainer maintainer = memory_allocator;

    const auto start = std::chrono::steady_clock::now();
    out_status = conv.RunBwdData(handle, y_tensor, x_tensor, w_tensor, algo.value, maintainer);
    if (musaDeviceSynchronize() != musaSuccess) {
      ADD_FAILURE() << "musaDeviceSynchronize failed";
      out_status = ::musa::dnn::Status::EXECUTION_FAILED;
    }
    const auto end = std::chrono::steady_clock::now();
    latency_us = std::chrono::duration<double, std::micro>(end - start).count();
    ok = out_status == ::musa::dnn::Status::SUCCESS;

    for (void* ptr : workspace_buffers) {
      EXPECT_EQ(musaFree(ptr), musaSuccess);
    }
  } catch (...) {
    out_status = ::musa::dnn::Status::INTERNAL_ERROR;
    ok = false;
  }

  free_buffers();
  return ok;
}

const std::vector<ConvTransposeScanConfig>& ScanConfigs() {
  static const std::vector<ConvTransposeScanConfig> configs = {
      {"asr_tts_ct3", {1, 256, 30}, {256, 128, 10}, {1, 128, 150}, {3}, {5}, {1}, 1},
      {"asr_tts_ct21", {1, 128, 150}, {128, 64, 8}, {1, 64, 600}, {2}, {4}, {1}, 1},
      {"asr_tts_ct39", {1, 64, 600}, {64, 32, 8}, {1, 32, 2400}, {2}, {4}, {1}, 1},
      {"asr_tts_ct57", {1, 32, 2400}, {32, 16, 4}, {1, 16, 4800}, {1}, {2}, {1}, 1},
      {"rife_head_240x5", {1, 240, 16, 16}, {240, 5, 4, 4}, {1, 5, 32, 32}, {1, 1}, {2, 2}, {1, 1}, 1},
      {"rife_up_512x128", {1, 512, 8, 8}, {512, 128, 4, 4}, {1, 128, 16, 16}, {1, 1}, {2, 2}, {1, 1}, 1},
      {"hifigan_stride8_512x256", {1, 512, 16}, {512, 256, 16}, {1, 256, 128}, {4}, {8}, {1}, 1},
      {"hifigan_stride2_128x64", {1, 128, 64}, {128, 64, 4}, {1, 64, 128}, {1}, {2}, {1}, 1},
  };
  return configs;
}

template <typename T>
void ScanDType(const char* dtype_name) {
  for (const auto& config : ScanConfigs()) {
    int pass_count = 0;
    for (const auto& algo : BwdDataAlgorithms()) {
      double latency_us = -1.0;
      ::musa::dnn::Status status = ::musa::dnn::Status::INTERNAL_ERROR;
      const bool ok = RunOne<T>(config, algo, latency_us, status);
      if (ok) {
        ++pass_count;
      }
      std::cout << "ConvTransposeAlgoScan"
                << " config=" << config.name
                << " dtype=" << dtype_name
                << " algo=" << algo.name
                << " status=" << StatusName(status)
                << " status_code=" << static_cast<int>(status)
                << " latency_us=" << (ok ? latency_us : -1.0)
                << std::endl;
    }
    if (IsOneDimensionalConfig(config)) {
      EXPECT_EQ(pass_count, 0) << "Direct muDNN ConvTranspose1D unexpectedly passed for config=" << config.name
                               << " dtype=" << dtype_name;
    } else {
      EXPECT_GT(pass_count, 0) << "All ConvTranspose BwdData algorithms failed for config=" << config.name
                               << " dtype=" << dtype_name;
    }
  }
}

template <typename T>
std::vector<T> Zeros(const std::vector<int64_t>& shape) {
  return std::vector<T>(ElementCount(shape), T{});
}

template <>
std::vector<MLFloat16> Zeros<MLFloat16>(const std::vector<int64_t>& shape) {
  return std::vector<MLFloat16>(ElementCount(shape), MLFloat16{});
}

template <typename T>
void RunOpLevelConfig(const ConvTransposeScanConfig& config, const char* dtype_name) {
  OpTester test("ConvTranspose", 11);
  test.AddAttribute("kernel_shape", ToInt64Vector(config.w_shape.size() == 3
                                                      ? std::vector<int>{static_cast<int>(config.w_shape[2])}
                                                      : std::vector<int>{static_cast<int>(config.w_shape[2]),
                                                                         static_cast<int>(config.w_shape[3])}));
  test.AddAttribute("pads", OnnxPads(config));
  test.AddAttribute("strides", ToInt64Vector(config.strides));
  test.AddAttribute("dilations", ToInt64Vector(config.dilations));
  test.AddAttribute("group", static_cast<int64_t>(config.groups));
  test.AddAttribute("output_shape", config.y_shape);

  test.AddInput<T>("X", config.x_shape, Zeros<T>(config.x_shape));
  test.AddInput<T>("W", config.w_shape, Zeros<T>(config.w_shape), true);
  test.AddOutput<T>("Y", config.y_shape, Zeros<T>(config.y_shape), false, 0.0f, 0.0f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  execution_providers.push_back(std::move(musa_provider));
  SessionOptions session_options;
  ASSERT_TRUE(session_options.config_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1").IsOK());
  SCOPED_TRACE(std::string("config=") + config.name + " dtype=" + dtype_name);
  test.Run(session_options, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

template <typename T>
void RunOpLevelDType(const char* dtype_name) {
  for (const auto& config : ScanConfigs()) {
    RunOpLevelConfig<T>(config, dtype_name);
    std::cout << "ConvTransposeOpLevel config=" << config.name
              << " dtype=" << dtype_name
              << " status=PASS"
              << std::endl;
  }
}

}  // namespace

TEST(MusaConvTransposeAlgoScanTest, BwdDataFp32) {
  ASSERT_EQ(musaSetDevice(0), musaSuccess);
  ScanDType<float>("fp32");
}

TEST(MusaConvTransposeAlgoScanTest, BwdDataFp16) {
  ASSERT_EQ(musaSetDevice(0), musaSuccess);
  ScanDType<uint16_t>("fp16");
}

TEST(MusaConvTransposeOpLevelTest, AllConfigsFp32) {
  RunOpLevelDType<float>("fp32");
}

TEST(MusaConvTransposeOpLevelTest, AllConfigsFp16) {
  RunOpLevelDType<MLFloat16>("fp16");
}

}  // namespace test
}  // namespace onnxruntime
