// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include <utility>
#include <memory>

#include "gtest/gtest.h"

#include "core/providers/common.h"
#include "core/providers/musa/musa_provider_options.h"

//#include "test/common/cuda_op_test_utils.h"
#include "test/providers/compare_provider_test_utils.h"
#include "test/util/include/default_providers.h"

// extended musa provider args. compare NHWC implementation vs MUSA NCHW and CPU EP.
#define MAKE_PROVIDERS_EPS_EXT(eps, pad_to_nc1d)                                \
  {                                                                             \
    std::vector<std::shared_ptr<IExecutionProvider>> execution_providers;       \
    OrtMUSAProviderOptions nhwc{};                                              \
    nhwc.prefer_nhwc = true;                                                    \
    execution_providers.push_back(MusaExecutionProviderWithOptions(&nhwc));     \
                                                                                \
    double error_tolerance = eps;                                               \
    OrtMUSAProviderOptions nchw{};                                              \
    nchw.prefer_nhwc = false;                                                   \
    auto nchw_ep = MusaExecutionProviderWithOptions(&nchw);                     \
    auto test = op.get_test();                                                  \
    test->CompareEPs(std::move(nchw_ep), execution_providers, error_tolerance); \
    auto cpu_ep = DefaultCpuExecutionProvider();                                \
    test->CompareEPs(std::move(cpu_ep), execution_providers, error_tolerance);  \
  }

#define MAKE_PROVIDERS_EPS(eps) \
  MAKE_PROVIDERS_EPS_EXT(eps, false)

#define MAKE_PROVIDERS() MAKE_PROVIDERS_EPS(1e-3)

#define MAKE_PROVIDERS_EPS_TYPE_EXT(T, pad_to_nc1d) \
  if (std::is_same<T, MLFloat16>::value) {          \
    MAKE_PROVIDERS_EPS_EXT(2e-2, pad_to_nc1d)       \
  } else if (std::is_same<T, double>::value) {      \
    MAKE_PROVIDERS_EPS_EXT(2e-4, pad_to_nc1d)       \
  } else {                                          \
    MAKE_PROVIDERS_EPS_EXT(2e-3, pad_to_nc1d)       \
  }

#define MAKE_PROVIDERS_EPS_TYPE(T) \
  MAKE_PROVIDERS_EPS_TYPE_EXT(T, false)

namespace onnxruntime {
namespace test {

template <typename T>
class MusaNhwcTypedTest : public ::testing::Test {};

using MusaNhwcTestTypes = ::testing::Types<float, MLFloat16>;  // double,
TYPED_TEST_SUITE(MusaNhwcTypedTest, MusaNhwcTestTypes);
}  // namespace test
}  // namespace onnxruntime
