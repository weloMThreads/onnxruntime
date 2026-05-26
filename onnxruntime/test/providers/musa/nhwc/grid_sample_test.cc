// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/constants.h"
#include "test/providers/musa/nhwc/nhwc_musa_helper.h"
#include "test/providers/provider_test_utils.h"

#include "gmock/gmock.h"

#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>

namespace onnxruntime {
namespace test {

namespace {

constexpr const char* kDisableFastPathEnv = "MUSA_GRIDSAMPLE_DISABLE_FASTPATH";
constexpr int64_t kN = 1;
constexpr int64_t kH = 1088;
constexpr int64_t kW = 1920;
constexpr int64_t kC = 3;

size_t NumElements(const std::vector<int64_t>& dims) {
  size_t numel = 1;
  for (int64_t dim : dims) {
    numel *= static_cast<size_t>(dim);
  }
  return numel;
}

std::vector<float> MakeInputData(size_t numel) {
  std::vector<float> data(numel);
  for (size_t i = 0; i < numel; ++i) {
    data[i] = static_cast<float>(static_cast<int>(i % 257) - 128) / 129.0f;
  }
  return data;
}

std::vector<float> MakeGridData() {
  std::vector<float> data(static_cast<size_t>(kN * kH * kW * 2));
  for (int64_t y = 0; y < kH; ++y) {
    for (int64_t x = 0; x < kW; ++x) {
      const size_t idx = static_cast<size_t>((y * kW + x) * 2);
      float grid_x = static_cast<float>(x) / static_cast<float>(kW - 1) * 2.4f - 1.2f;
      float grid_y = static_cast<float>(y) / static_cast<float>(kH - 1) * 2.4f - 1.2f;
      if (((x / 64) & 1) != 0) {
        grid_x = -grid_x;
      }
      if (((y / 64) & 1) != 0) {
        grid_y = -grid_y;
      }
      data[idx] = grid_x;
      data[idx + 1] = grid_y;
    }
  }
  return data;
}

template <typename T>
std::vector<T> ConvertData(const std::vector<float>& values) {
  if constexpr (std::is_same_v<T, MLFloat16>) {
    return FloatsToMLFloat16s(values);
  } else {
    return values;
  }
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

std::unique_ptr<IExecutionProvider> CreateNhwcMusaProvider() {
  OrtMUSAProviderOptions provider_options{};
  provider_options.prefer_nhwc = true;
  return MusaExecutionProviderWithOptions(&provider_options);
}

template <typename T>
OrtValue RunInternalNhwcGridSample(bool disable_fast_path) {
  ScopedEnvVar disable_fast_path_env{kDisableFastPathEnv, disable_fast_path ? "1" : nullptr};

  const std::vector<int64_t> x_dims{kN, kH, kW, kC};
  const std::vector<int64_t> grid_dims{kN, kH, kW, 2};
  const std::vector<int64_t> y_dims{kN, kH, kW, kC};

  auto input_data = ConvertData<T>(MakeInputData(NumElements(x_dims)));
  auto grid_data = ConvertData<T>(MakeGridData());
  auto output_data = ConvertData<T>(std::vector<float>(NumElements(y_dims), 0.0f));

  OpTester test("GridSample", 16, onnxruntime::kMSInternalNHWCDomain, false);
  test.AddInput<T>("X", x_dims, input_data);
  test.AddInput<T>("Grid", grid_dims, grid_data);
  test.AddOutput<T>("Y", y_dims, output_data);
  test.AddAttribute("mode", std::string("bilinear"));
  test.AddAttribute("padding_mode", std::string("border"));
  test.AddAttribute("align_corners", static_cast<int64_t>(1));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(CreateNhwcMusaProvider());
  ORT_ENFORCE(execution_providers.front() != nullptr);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  auto fetches = test.GetFetches();
  ORT_ENFORCE(fetches.size() == 1);
  return std::move(fetches[0]);
}

template <typename T>
void ExpectFastPathMatchesGeneric() {
  const OrtValue fast_output = RunInternalNhwcGridSample<T>(false);
  const OrtValue generic_output = RunInternalNhwcGridSample<T>(true);

  const auto& fast_tensor = fast_output.Get<Tensor>();
  const auto& generic_tensor = generic_output.Get<Tensor>();
  ASSERT_EQ(fast_tensor.Shape().GetDims(), generic_tensor.Shape().GetDims());

  if constexpr (std::is_same_v<T, MLFloat16>) {
    EXPECT_THAT(fast_tensor.DataAsSpan<MLFloat16>(),
                ::testing::Pointwise(::testing::FloatNear(5e-3f), generic_tensor.DataAsSpan<MLFloat16>()));
  } else {
    EXPECT_THAT(fast_tensor.DataAsSpan<float>(),
                ::testing::Pointwise(::testing::FloatNear(1e-6f), generic_tensor.DataAsSpan<float>()));
  }
}

}  // namespace

TYPED_TEST(MusaNhwcTypedTest, GridSampleFastPathConsistency) {
  if (!CreateNhwcMusaProvider()) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  ExpectFastPathMatchesGeneric<TypeParam>();
}

}  // namespace test
}  // namespace onnxruntime
