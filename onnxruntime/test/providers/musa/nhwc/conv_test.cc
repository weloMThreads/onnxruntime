// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/musa/musa_provider_options.h"
#include "core/graph/constants.h"
#include "test/providers/musa/nhwc/nhwc_musa_helper.h"

#include <type_traits>

namespace onnxruntime {
namespace test {

template <typename T>
struct ConvOp {
  std::vector<int64_t> input_dims;
  std::vector<int64_t> kernel_shape;
  int64_t channels;
  int64_t group = 1;
  bool bias = false;
  std::vector<int64_t> strides = {1, 1};
  std::vector<int64_t> padding = {0, 0, 0, 0};
  std::vector<int64_t> dilations = {1, 1};

  std::unique_ptr<CompareOpTester> get_test() {
    RandomValueGenerator random{optional<RandomValueGenerator::RandomSeedType>{42}};

    auto test = std::make_unique<CompareOpTester>("Conv", 11);  // internal NHWC domain starts at opset 11
    std::vector<T> input_data = random.Uniform<T>(input_dims, 0.0f, 1.0f);

    std::vector<int64_t> weight_dims{channels, input_dims[1] / group, kernel_shape[0], kernel_shape[1]};
    std::vector<T> weight_data = random.Uniform<T>(weight_dims, -0.4f, 0.4f);

    test->AddInput<T>("X", input_dims, input_data);
    test->AddInput<T>("W", weight_dims, weight_data, true);
    if (bias) {
      std::vector<int64_t> bias_dims{channels};
      std::vector<T> bias_data = random.Uniform<T>(bias_dims, 0.2f, 0.4f);
      test->AddInput<T>("B", bias_dims, bias_data, true);
    }
    test->AddAttribute("group", group);
    test->AddAttribute("kernel_shape", kernel_shape);
    test->AddAttribute("strides", strides);
    test->AddAttribute("dilations", dilations);
    test->AddAttribute("pads", padding);

    std::vector<int64_t> output_dims = {
        input_dims[0], channels,
        ComputeOutputShape(input_dims[2], strides[0], kernel_shape[0], dilations[0], padding[0], padding[1]),
        ComputeOutputShape(input_dims[3], strides[1], kernel_shape[1], dilations[1], padding[2], padding[3])};
    std::vector<T> output_data = FillZeros<T>(output_dims);

    test->AddOutput<T>("Y", output_dims, output_data);
    return test;
  }
};

TYPED_TEST(MusaNhwcTypedTest, ConvNhwcBias) {
  auto op = ConvOp<TypeParam>{};
  op.input_dims = {1, 16, 64, 64};
  op.kernel_shape = {3, 3};
  op.channels = 16;
  op.bias = true;

  MAKE_PROVIDERS_EPS_TYPE(TypeParam)
}

TYPED_TEST(MusaNhwcTypedTest, ConvNhwcGroupNoBias) {
  auto op = ConvOp<TypeParam>{};
  op.input_dims = {1, 16, 64, 64};
  op.kernel_shape = {3, 3};
  op.channels = 16;
  op.group = 4;

  MAKE_PROVIDERS_EPS_TYPE(TypeParam)
}

TYPED_TEST(MusaNhwcTypedTest, ConvNhwcPadding) {
  auto op = ConvOp<TypeParam>{};
  op.input_dims = {2, 4, 64, 64};
  op.kernel_shape = {3, 3};
  op.channels = 4;
  op.padding = {4, 4, 4, 4};

  MAKE_PROVIDERS_EPS_TYPE(TypeParam)
}

namespace {

template <typename T>
void AddGridSampleData(OpTester& test,
                       const std::vector<int64_t>& x_dims,
                       const std::vector<float>& x_vals,
                       const std::vector<int64_t>& grid_dims,
                       const std::vector<float>& grid_vals,
                       const std::vector<int64_t>& y_dims,
                       const std::vector<float>& y_vals) {
  if constexpr (std::is_same_v<T, MLFloat16>) {
    test.AddInput<MLFloat16>("X", x_dims, FloatsToMLFloat16s(x_vals));
    test.AddInput<MLFloat16>("Grid", grid_dims, FloatsToMLFloat16s(grid_vals));
    test.AddOutput<MLFloat16>("Y", y_dims, FloatsToMLFloat16s(y_vals));
  } else {
    test.AddInput<float>("X", x_dims, x_vals);
    test.AddInput<float>("Grid", grid_dims, grid_vals);
    test.AddOutput<float>("Y", y_dims, y_vals);
  }
}

template <typename T>
void RunGridSampleInternalNhwcTest() {
  OpTester test("GridSample", 16, onnxruntime::kMSInternalNHWCDomain);

  const std::vector<int64_t> x_dims{1, 2, 2, 1};
  const std::vector<int64_t> grid_dims{1, 2, 3, 2};
  const std::vector<int64_t> y_dims{1, 2, 3, 1};

  AddGridSampleData<T>(
      test,
      x_dims, {1.0f, 2.0f, 3.0f, 4.0f},
      grid_dims, {-1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f,
                  -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f},
      y_dims, {1.0f, 2.0f, 4.0f,
               3.0f, 1.0f, 4.0f});

  test.AddAttribute("mode", "nearest");
  test.AddAttribute("padding_mode", "zeros");
  test.AddAttribute("align_corners", static_cast<int64_t>(1));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  OrtMUSAProviderOptions provider_options{};
  provider_options.prefer_nhwc = true;
  auto musa_provider = MusaExecutionProviderWithOptions(&provider_options);
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  execution_providers.push_back(std::move(musa_provider));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

}  // namespace

TEST(MusaNhwcOpTest, GridSampleInternalNhwcFloat) {
  RunGridSampleInternalNhwcTest<float>();
}

TEST(MusaNhwcOpTest, GridSampleInternalNhwcFp16) {
  RunGridSampleInternalNhwcTest<MLFloat16>();
}

template <typename T>
void RunConvTransposeInternalNhwcBiasTest() {
  OpTester test("ConvTranspose", 11, onnxruntime::kMSInternalNHWCDomain);

  if constexpr (std::is_same_v<T, MLFloat16>) {
    test.AddInput<MLFloat16>("X", {1, 1, 1, 1}, ToFloat16({2.0f}));
    test.AddInput<MLFloat16>("W", {1, 1, 2, 2}, ToFloat16({1.0f, 1.0f, 1.0f, 1.0f}), true);
    test.AddInput<MLFloat16>("B", {1}, ToFloat16({3.0f}), true);
    test.AddOutput<MLFloat16>("Y", {1, 2, 2, 1}, ToFloat16({5.0f, 5.0f, 5.0f, 5.0f}));
  } else {
    test.AddInput<float>("X", {1, 1, 1, 1}, {2.0f});
    test.AddInput<float>("W", {1, 1, 2, 2}, {1.0f, 1.0f, 1.0f, 1.0f}, true);
    test.AddInput<float>("B", {1}, {3.0f}, true);
    test.AddOutput<float>("Y", {1, 2, 2, 1}, {5.0f, 5.0f, 5.0f, 5.0f});
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  OrtMUSAProviderOptions provider_options{};
  provider_options.prefer_nhwc = true;
  auto musa_provider = MusaExecutionProviderWithOptions(&provider_options);
  if (!musa_provider) {
    GTEST_SKIP() << "MUSA execution provider not available";
  }

  execution_providers.push_back(std::move(musa_provider));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(MusaNhwcOpTest, ConvTransposeInternalNhwcFloat) {
  RunConvTransposeInternalNhwcBiasTest<float>();
}

TEST(MusaNhwcOpTest, ConvTransposeInternalNhwcFp16) {
  RunConvTransposeInternalNhwcBiasTest<MLFloat16>();
}

}  // namespace test
}  // namespace onnxruntime
