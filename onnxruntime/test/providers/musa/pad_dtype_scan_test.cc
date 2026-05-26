// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <mudnn.h>
#include <musa_runtime.h>

#include "core/session/onnxruntime_session_options_config_keys.h"
#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {
namespace {

struct PadScanConfig {
  const char* name;
  std::vector<int64_t> input_shape;
  std::vector<int64_t> output_shape;
  std::vector<int64_t> onnx_pads;
  std::vector<int> musa_pads;
};

struct PadModeCase {
  const char* name;
  ::musa::dnn::Pad::Mode value;
};

const std::vector<PadScanConfig>& PadScanConfigs() {
  static const std::vector<PadScanConfig> configs = {
      {"pad_stage0", {1, 512, 96}, {1, 512, 111}, {0, 0, 0, 0, 0, 15}, {15, 0, 0, 0, 0, 0}},
      {"pad_stage1", {1, 768, 48}, {1, 768, 63}, {0, 0, 0, 0, 0, 15}, {15, 0, 0, 0, 0, 0}},
      {"pad_stage2", {1, 1024, 24}, {1, 1024, 31}, {0, 0, 0, 0, 0, 7}, {7, 0, 0, 0, 0, 0}},
      {"pad_stage3", {1, 1536, 12}, {1, 1536, 19}, {0, 0, 0, 0, 0, 7}, {7, 0, 0, 0, 0, 0}},
  };
  return configs;
}

const std::vector<PadModeCase>& PadModes() {
  static const std::vector<PadModeCase> modes = {
      {"constant", ::musa::dnn::Pad::Mode::CONSTANT},
      {"edge", ::musa::dnn::Pad::Mode::REPLICATE},
  };
  return modes;
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
  for (int64_t dim : shape) {
    count *= static_cast<size_t>(dim);
  }
  return count;
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

template <>
::musa::dnn::Tensor::Type MusaTensorType<double>() {
  return ::musa::dnn::Tensor::Type::DOUBLE;
}

struct BFloat16Storage {
  uint16_t value;
};

template <>
::musa::dnn::Tensor::Type MusaTensorType<BFloat16Storage>() {
  return ::musa::dnn::Tensor::Type::BFLOAT16;
}

template <typename T>
bool SetupTensor(::musa::dnn::Tensor& tensor, void* data, const std::vector<int64_t>& shape) {
  if (tensor.SetType(MusaTensorType<T>()) != ::musa::dnn::Status::SUCCESS) {
    ADD_FAILURE() << "SetType failed";
    return false;
  }
  if (tensor.SetFormat(::musa::dnn::Tensor::Format::NCHW) != ::musa::dnn::Status::SUCCESS) {
    ADD_FAILURE() << "SetFormat failed";
    return false;
  }
  if (tensor.SetNdInfo(static_cast<int>(shape.size()), shape.data()) != ::musa::dnn::Status::SUCCESS) {
    ADD_FAILURE() << "SetNdInfo failed";
    return false;
  }
  if (tensor.SetAddr(data) != ::musa::dnn::Status::SUCCESS) {
    ADD_FAILURE() << "SetAddr failed";
    return false;
  }
  return true;
}

template <typename T>
bool RunOne(const PadScanConfig& config, const PadModeCase& mode, double& latency_us,
            ::musa::dnn::Status& out_status) {
  void* input = nullptr;
  void* output = nullptr;
  const size_t input_bytes = ElementCount(config.input_shape) * sizeof(T);
  const size_t output_bytes = ElementCount(config.output_shape) * sizeof(T);

  auto free_buffers = [&]() {
    if (input != nullptr) {
      EXPECT_EQ(musaFree(input), musaSuccess);
    }
    if (output != nullptr) {
      EXPECT_EQ(musaFree(output), musaSuccess);
    }
  };

  if (musaMalloc(&input, input_bytes) != musaSuccess ||
      musaMalloc(&output, output_bytes) != musaSuccess) {
    ADD_FAILURE() << "musaMalloc failed";
    free_buffers();
    return false;
  }
  if (musaMemset(input, 0, input_bytes) != musaSuccess ||
      musaMemset(output, 0, output_bytes) != musaSuccess) {
    ADD_FAILURE() << "musaMemset failed";
    free_buffers();
    return false;
  }

  bool ok = false;
  try {
    ::musa::dnn::Handle handle;
    ::musa::dnn::Tensor input_tensor;
    ::musa::dnn::Tensor output_tensor;
    if (!SetupTensor<T>(input_tensor, input, config.input_shape) ||
        !SetupTensor<T>(output_tensor, output, config.output_shape)) {
      free_buffers();
      return false;
    }

    ::musa::dnn::Pad pad;
    out_status = pad.SetMode(mode.value);
    if (out_status == ::musa::dnn::Status::SUCCESS && mode.value == ::musa::dnn::Pad::Mode::CONSTANT) {
      out_status = pad.SetValue(0.0);
    }
    if (out_status == ::musa::dnn::Status::SUCCESS) {
      out_status = pad.SetPaddingInfo(static_cast<int>(config.musa_pads.size()), config.musa_pads.data());
    }

    const auto start = std::chrono::steady_clock::now();
    if (out_status == ::musa::dnn::Status::SUCCESS) {
      out_status = pad.Run(handle, output_tensor, input_tensor);
      if (musaDeviceSynchronize() != musaSuccess) {
        out_status = ::musa::dnn::Status::EXECUTION_FAILED;
      }
    }
    const auto end = std::chrono::steady_clock::now();
    latency_us = std::chrono::duration<double, std::micro>(end - start).count();
    ok = out_status == ::musa::dnn::Status::SUCCESS;
  } catch (...) {
    out_status = ::musa::dnn::Status::INTERNAL_ERROR;
    ok = false;
  }

  free_buffers();
  return ok;
}

template <typename T>
void ScanDType(const char* dtype_name) {
  for (const auto& config : PadScanConfigs()) {
    for (const auto& mode : PadModes()) {
      double latency_us = -1.0;
      ::musa::dnn::Status status = ::musa::dnn::Status::INTERNAL_ERROR;
      const bool ok = RunOne<T>(config, mode, latency_us, status);
      std::cout << "PadDtypeScan"
                << " config=" << config.name
                << " dtype=" << dtype_name
                << " mode=" << mode.name
                << " status=" << StatusName(status)
                << " status_code=" << static_cast<int>(status)
                << " latency_us=" << (ok ? latency_us : -1.0)
                << std::endl;
    }
  }
}

SessionOptions MakeNoFallbackSessionOptions() {
  SessionOptions so;
  ORT_THROW_IF_ERROR(so.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  return so;
}

const char* PadFp16OpLevelStatusLabel() {
#if defined(MUDNN_VERSION) && (MUDNN_VERSION < 3100)
  return "EXPECTED_NO_MUSA_KERNEL";
#else
  return "PASS";
#endif
}

template <typename T>
void RunPadOpLevelConfig(const PadScanConfig& config, const char* dtype_name, const char* mode) {
  OpTester test("Pad", 13);
  if (std::string(mode) != "constant") {
    test.AddAttribute("mode", std::string(mode));
  }

  const std::vector<T> input(ElementCount(config.input_shape), T{});
  const std::vector<T> expected(ElementCount(config.output_shape), T{});
  test.AddInput<T>("data", config.input_shape, input);
  test.AddInput<int64_t>("pads", {static_cast<int64_t>(config.onnx_pads.size())}, config.onnx_pads);
  test.AddInput<T>("value", {}, {T{}});
  test.AddOutput<T>("output", config.output_shape, expected);

  auto musa_provider = DefaultMusaExecutionProvider();
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(musa_provider));

#if defined(MUDNN_VERSION) && (MUDNN_VERSION < 3100)
  const auto expect_result = OpTester::ExpectResult::kExpectFailure;
  const std::string expected_failure = "fallback to CPU EP has been explicitly disabled";
#else
  const auto expect_result = OpTester::ExpectResult::kExpectSuccess;
  const std::string expected_failure;
#endif

  SCOPED_TRACE(std::string("config=") + config.name + " dtype=" + dtype_name + " mode=" + mode);
  // Old muDNN builds do not register Pad fp16, so disabling CPU fallback should fail at session init.
  test.Run(MakeNoFallbackSessionOptions(),
           expect_result,
           expected_failure,
           {},
           nullptr,
           &execution_providers);
}

template <typename T>
void RunPadOpLevelDType(const char* dtype_name, const char* mode) {
  for (const auto& config : PadScanConfigs()) {
    RunPadOpLevelConfig<T>(config, dtype_name, mode);
    if (!::testing::Test::HasFailure()) {
      std::cout << "PadOpLevel"
                << " config=" << config.name
                << " dtype=" << dtype_name
                << " mode=" << mode
                << " status=" << PadFp16OpLevelStatusLabel()
                << std::endl;
    }
  }
}

}  // namespace

TEST(MusaPadDtypeScanTest, Fp32) {
  ASSERT_EQ(musaSetDevice(0), musaSuccess);
  ScanDType<float>("fp32");
}

TEST(MusaPadDtypeScanTest, Fp16) {
  ASSERT_EQ(musaSetDevice(0), musaSuccess);
  ScanDType<uint16_t>("fp16");
}

TEST(MusaPadDtypeScanTest, Bf16) {
  ASSERT_EQ(musaSetDevice(0), musaSuccess);
  ScanDType<BFloat16Storage>("bf16");
}

TEST(MusaPadDtypeScanTest, Fp64) {
  ASSERT_EQ(musaSetDevice(0), musaSuccess);
  ScanDType<double>("fp64");
}

TEST(MusaPadOpLevelTest, AsrTtsFp16ConstantNoCpuFallback) {
  RunPadOpLevelDType<MLFloat16>("fp16", "constant");
}

TEST(MusaPadOpLevelTest, AsrTtsFp16EdgeNoCpuFallback) {
  RunPadOpLevelDType<MLFloat16>("fp16", "edge");
}

}  // namespace test
}  // namespace onnxruntime
