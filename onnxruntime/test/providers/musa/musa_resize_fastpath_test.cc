// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

#include "gtest/gtest.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

namespace onnxruntime {
namespace test {

namespace {

constexpr const char* kDisableFastPathEnv = "MUSA_RESIZE_DISABLE_FASTPATH";
constexpr int64_t kN = 1;
constexpr int64_t kC = 5;
constexpr int64_t kIdentityH = 1088;
constexpr int64_t kIdentityW = 1920;
constexpr int64_t kUpsampleInH = 272;
constexpr int64_t kUpsampleInW = 480;
constexpr int64_t kUpsampleOutH = 1088;
constexpr int64_t kUpsampleOutW = 1920;

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
    data[i] = static_cast<float>((static_cast<int>(i % 509) - 254)) / 255.0f;
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

std::unique_ptr<IExecutionProvider> CreateMusaProvider() {
  return DefaultMusaExecutionProvider();
}

template <typename T>
OrtValue RunResize(bool disable_fast_path,
                   const std::vector<int64_t>& x_dims,
                   const std::vector<int64_t>& y_dims) {
  ScopedEnvVar disable_fast_path_env{kDisableFastPathEnv, disable_fast_path ? "1" : nullptr};

  auto input_data = ConvertData<T>(MakeInputData(NumElements(x_dims)));
  auto output_data = ConvertData<T>(std::vector<float>(NumElements(y_dims), 0.0f));
  std::vector<float> roi{};
  std::vector<float> scales{};

  OpTester test("Resize", 18);
  test.AddAttribute("mode", std::string("linear"));
  test.AddInput<T>("X", x_dims, input_data);
  test.AddInput<float>("roi", {0}, roi);
  test.AddInput<float>("", {0}, scales);
  test.AddInput<int64_t>("sizes", {4}, y_dims);
  test.AddOutput<T>("Y", y_dims, output_data);
  test.SetCustomOutputVerifier([](const std::vector<OrtValue>&, const std::string&) {});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(CreateMusaProvider());
  ORT_ENFORCE(execution_providers.front() != nullptr);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  auto fetches = test.GetFetches();
  ORT_ENFORCE(fetches.size() == 1);
  return std::move(fetches[0]);
}

template <typename T>
void ExpectOutputsExactlyMatch(const OrtValue& lhs_value, const OrtValue& rhs_value) {
  const auto& lhs_tensor = lhs_value.Get<Tensor>();
  const auto& rhs_tensor = rhs_value.Get<Tensor>();
  ASSERT_EQ(lhs_tensor.Shape().GetDims(), rhs_tensor.Shape().GetDims());

  const size_t byte_size = static_cast<size_t>(lhs_tensor.Shape().Size()) * sizeof(T);
  EXPECT_EQ(std::memcmp(lhs_tensor.DataRaw(), rhs_tensor.DataRaw(), byte_size), 0);
}

template <typename T>
void ExpectIdentityFastPathMatchesGeneric() {
  const std::vector<int64_t> x_dims{kN, kC, kIdentityH, kIdentityW};
  const std::vector<int64_t> y_dims{kN, kC, kIdentityH, kIdentityW};

  const OrtValue fast_output = RunResize<T>(false, x_dims, y_dims);
  const OrtValue generic_output = RunResize<T>(true, x_dims, y_dims);
  ExpectOutputsExactlyMatch<T>(fast_output, generic_output);
}

template <typename T>
void ExpectVecFastPathMatchesGeneric() {
  const std::vector<int64_t> x_dims{kN, kC, kUpsampleInH, kUpsampleInW};
  const std::vector<int64_t> y_dims{kN, kC, kUpsampleOutH, kUpsampleOutW};

  const OrtValue fast_output = RunResize<T>(false, x_dims, y_dims);
  const OrtValue generic_output = RunResize<T>(true, x_dims, y_dims);
  ExpectOutputsExactlyMatch<T>(fast_output, generic_output);
}

template <typename T>
class MusaResizeFastPathTest : public ::testing::Test {};

using MusaResizeFastPathTypes = ::testing::Types<float, MLFloat16>;
TYPED_TEST_SUITE(MusaResizeFastPathTest, MusaResizeFastPathTypes);

}  // namespace

TYPED_TEST(MusaResizeFastPathTest, ResizeIdentityFastPathConsistency) {
  if (!CreateMusaProvider()) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  ExpectIdentityFastPathMatchesGeneric<TypeParam>();
}

TYPED_TEST(MusaResizeFastPathTest, ResizeVecC5FastPathConsistency) {
  if (!CreateMusaProvider()) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  ExpectVecFastPathMatchesGeneric<TypeParam>();
}

}  // namespace test
}  // namespace onnxruntime
