// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/providers/musa/musa_provider_options.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

namespace {

constexpr int64_t kN = 1;
constexpr int64_t kC = 3;
constexpr int64_t kH = 1088;
constexpr int64_t kW = 1920;
constexpr int64_t kResizeC = 5;
constexpr int64_t kResizeInH = 272;
constexpr int64_t kResizeInW = 480;
constexpr int64_t kResizeOutH = 1088;
constexpr int64_t kResizeOutW = 1920;
constexpr int kWarmupIters = 10;
constexpr int kTimedIters = 200;
constexpr const char* kDisableResizeFastPathEnv = "MUSA_RESIZE_DISABLE_FASTPATH";

std::vector<MLFloat16> MakeInputData(size_t numel) {
  std::vector<MLFloat16> data(numel);
  for (size_t i = 0; i < numel; ++i) {
    data[i] = MLFloat16(static_cast<float>(i % 257) / 257.0f);
  }
  return data;
}

std::vector<MLFloat16> MakeGridData() {
  std::vector<MLFloat16> grid(static_cast<size_t>(kN * kH * kW * 2));
  for (int64_t y = 0; y < kH; ++y) {
    for (int64_t x = 0; x < kW; ++x) {
      const size_t idx = static_cast<size_t>((y * kW + x) * 2);
      const float grid_x = (static_cast<float>(x) + 0.25f) / static_cast<float>(kW - 1) * 2.0f - 1.0f;
      const float grid_y = (static_cast<float>(y) + 0.5f) / static_cast<float>(kH - 1) * 2.0f - 1.0f;
      grid[idx] = MLFloat16(grid_x);
      grid[idx + 1] = MLFloat16(grid_y);
    }
  }
  return grid;
}

class ScopedEnvVar {
 public:
  ScopedEnvVar(const char* name, const char* value)
      : name_(name) {
    const char* original_value = std::getenv(name_);
    if (original_value != nullptr) {
      had_original_value_ = true;
      original_value_ = original_value;
    }

    if (value != nullptr) {
      setenv(name_, value, 1);
    } else {
      unsetenv(name_);
    }
  }

  ~ScopedEnvVar() {
    if (had_original_value_) {
      setenv(name_, original_value_.c_str(), 1);
    } else {
      unsetenv(name_);
    }
  }

 private:
  const char* name_;
  bool had_original_value_ = false;
  std::string original_value_;
};

void AddGridSampleData(OpTester& test,
                       const std::vector<int64_t>& x_dims,
                       const std::vector<MLFloat16>& x_vals,
                       const std::vector<int64_t>& grid_dims,
                       const std::vector<MLFloat16>& grid_vals,
                       const std::vector<int64_t>& y_dims,
                       const std::vector<MLFloat16>& y_vals) {
  test.AddInput<MLFloat16>("X", x_dims, x_vals);
  test.AddInput<MLFloat16>("Grid", grid_dims, grid_vals);
  test.AddOutput<MLFloat16>("Y", y_dims, y_vals);
}

void ConfigureGridSamplePerfTest(OpTester& test,
                                 const std::vector<int64_t>& x_dims,
                                 const std::vector<MLFloat16>& input_data,
                                 const std::vector<int64_t>& grid_dims,
                                 const std::vector<MLFloat16>& grid_data,
                                 const std::vector<int64_t>& y_dims,
                                 const std::vector<MLFloat16>& output_data) {
  AddGridSampleData(test, x_dims, input_data, grid_dims, grid_data, y_dims, output_data);
  test.AddAttribute("mode", std::string("bilinear"));
  test.AddAttribute("padding_mode", std::string("border"));
  test.AddAttribute("align_corners", static_cast<int64_t>(1));
}

void ConfigureResizePerfTest(OpTester& test,
                             const std::vector<int64_t>& x_dims,
                             const std::vector<MLFloat16>& input_data,
                             const std::vector<int64_t>& y_dims,
                             const std::vector<MLFloat16>& output_data) {
  std::vector<float> roi{};
  std::vector<float> scales{};
  test.AddInput<MLFloat16>("X", x_dims, input_data);
  test.AddInput<float>("roi", {0}, roi);
  test.AddInput<float>("", {0}, scales);
  test.AddInput<int64_t>("sizes", {4}, y_dims);
  test.AddOutput<MLFloat16>("Y", y_dims, output_data);
  test.AddAttribute("mode", std::string("linear"));
  test.SetCustomOutputVerifier([](const std::vector<OrtValue>&, const std::string&) {});
}

std::unique_ptr<IExecutionProvider> CreateMusaProvider(bool prefer_nhwc) {
  if (!prefer_nhwc) {
    return DefaultMusaExecutionProvider();
  }

  OrtMUSAProviderOptions provider_options{};
  provider_options.prefer_nhwc = true;
  return MusaExecutionProviderWithOptions(&provider_options);
}

void RunGridSamplePerfCase(const std::string& case_name,
                           std::string_view domain,
                           const std::vector<int64_t>& x_dims,
                           const std::vector<MLFloat16>& input_data,
                           const std::vector<int64_t>& grid_dims,
                           const std::vector<MLFloat16>& grid_data,
                           const std::vector<int64_t>& y_dims,
                           const std::vector<MLFloat16>& output_data,
                           bool prefer_nhwc) {
  auto make_execution_providers = [prefer_nhwc]() {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(CreateMusaProvider(prefer_nhwc));
    return execution_providers;
  };

  auto warmup_execution_providers = make_execution_providers();
  ASSERT_TRUE(warmup_execution_providers.front() != nullptr);

  OpTester warmup_test("GridSample", 16, domain, false);
  ConfigureGridSamplePerfTest(warmup_test, x_dims, input_data, grid_dims, grid_data, y_dims, output_data);
  warmup_test.SetNumRunCalls(kWarmupIters);
  warmup_test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &warmup_execution_providers);

  auto perf_execution_providers = make_execution_providers();
  ASSERT_TRUE(perf_execution_providers.front() != nullptr);

  OpTester perf_test("GridSample", 16, domain, false);
  ConfigureGridSamplePerfTest(perf_test, x_dims, input_data, grid_dims, grid_data, y_dims, output_data);
  perf_test.SetNumRunCalls(kTimedIters);
  const auto start_time = std::chrono::steady_clock::now();
  perf_test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &perf_execution_providers);
  const auto end_time = std::chrono::steady_clock::now();

  const double total_us = std::chrono::duration<double, std::micro>(end_time - start_time).count();
  const double per_call_us = total_us / static_cast<double>(kTimedIters);

  std::cout << "\n=== " << case_name << " ===\n";
  std::cout << "  Per call: " << std::fixed << std::setprecision(2) << per_call_us << " us\n";
  std::cout << "  x6 / run: " << std::fixed << std::setprecision(2) << (per_call_us * 6.0 / 1000.0) << " ms\n";
}

double RunResizePerfCase(const std::vector<int64_t>& x_dims,
                         const std::vector<MLFloat16>& input_data,
                         const std::vector<int64_t>& y_dims,
                         const std::vector<MLFloat16>& output_data,
                         bool disable_fast_path) {
  ScopedEnvVar disable_fast_path_env{kDisableResizeFastPathEnv, disable_fast_path ? "1" : nullptr};

  auto make_execution_providers = []() {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(CreateMusaProvider(false));
    return execution_providers;
  };

  auto warmup_execution_providers = make_execution_providers();
  ORT_ENFORCE(warmup_execution_providers.front() != nullptr);

  OpTester warmup_test("Resize", 18);
  ConfigureResizePerfTest(warmup_test, x_dims, input_data, y_dims, output_data);
  warmup_test.SetNumRunCalls(kWarmupIters);
  warmup_test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &warmup_execution_providers);

  auto perf_execution_providers = make_execution_providers();
  ORT_ENFORCE(perf_execution_providers.front() != nullptr);

  OpTester perf_test("Resize", 18);
  ConfigureResizePerfTest(perf_test, x_dims, input_data, y_dims, output_data);
  perf_test.SetNumRunCalls(kTimedIters);
  const auto start_time = std::chrono::steady_clock::now();
  perf_test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &perf_execution_providers);
  const auto end_time = std::chrono::steady_clock::now();

  const double total_us = std::chrono::duration<double, std::micro>(end_time - start_time).count();
  return total_us / static_cast<double>(kTimedIters);
}

void PrintResizePerfCase(const std::string& case_name, double slow_us, double fast_us) {
  const double delta_pct = slow_us > 0.0 ? (fast_us - slow_us) / slow_us * 100.0 : 0.0;
  std::cout << "\n=== " << case_name << " ===\n";
  std::cout << "  Slow path: " << std::fixed << std::setprecision(2) << slow_us << " us\n";
  std::cout << "  Fast path: " << std::fixed << std::setprecision(2) << fast_us << " us\n";
  std::cout << "  Delta: " << std::fixed << std::setprecision(2) << delta_pct << "%\n";
}

}  // namespace

class V016Microbench : public ::testing::Test {};

TEST_F(V016Microbench, GridSample_C3_Perf) {
  const size_t input_numel = static_cast<size_t>(kN * kC * kH * kW);
  const size_t output_numel = input_numel;
  const std::vector<int64_t> x_dims{kN, kC, kH, kW};
  const std::vector<int64_t> grid_dims{kN, kH, kW, 2};
  const std::vector<int64_t> y_dims{kN, kC, kH, kW};

  auto input_data = MakeInputData(input_numel);
  auto grid_data = MakeGridData();
  std::vector<MLFloat16> output_data(output_numel, MLFloat16(0.0f));

  if (!CreateMusaProvider(false)) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  RunGridSamplePerfCase(
      "GridSample C3 1088x1920 PERF",
      onnxruntime::kOnnxDomain,
      x_dims,
      input_data,
      grid_dims,
      grid_data,
      y_dims,
      output_data,
      false);
}

TEST_F(V016Microbench, GridSample_C3_NHWC_Perf) {
  const size_t input_numel = static_cast<size_t>(kN * kH * kW * kC);
  const size_t output_numel = input_numel;
  const std::vector<int64_t> x_dims{kN, kH, kW, kC};
  const std::vector<int64_t> grid_dims{kN, kH, kW, 2};
  const std::vector<int64_t> y_dims{kN, kH, kW, kC};

  auto input_data = MakeInputData(input_numel);
  auto grid_data = MakeGridData();
  std::vector<MLFloat16> output_data(output_numel, MLFloat16(0.0f));

  if (!CreateMusaProvider(true)) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  RunGridSamplePerfCase(
      "GridSample C3 1088x1920 NHWC PERF",
      onnxruntime::kMSInternalNHWCDomain,
      x_dims,
      input_data,
      grid_dims,
      grid_data,
      y_dims,
      output_data,
      true);
}

TEST_F(V016Microbench, Resize_C5_1088x1920_Perf) {
  const size_t input_numel = static_cast<size_t>(kN * kResizeC * kResizeInH * kResizeInW);
  const size_t output_numel = static_cast<size_t>(kN * kResizeC * kResizeOutH * kResizeOutW);
  const std::vector<int64_t> x_dims{kN, kResizeC, kResizeInH, kResizeInW};
  const std::vector<int64_t> y_dims{kN, kResizeC, kResizeOutH, kResizeOutW};

  auto input_data = MakeInputData(input_numel);
  std::vector<MLFloat16> output_data(output_numel, MLFloat16(0.0f));

  if (!CreateMusaProvider(false)) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  const double slow_us = RunResizePerfCase(x_dims, input_data, y_dims, output_data, true);
  const double fast_us = RunResizePerfCase(x_dims, input_data, y_dims, output_data, false);
  PrintResizePerfCase("Resize C5 272x480 -> 1088x1920 PERF", slow_us, fast_us);
}

}  // namespace test
}  // namespace onnxruntime
