// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <iostream>
#include <vector>

#include <mudnn.h>
#include <musa_runtime.h>

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
namespace {

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

template <typename T>
::musa::dnn::Tensor::Type MusaTensorType();

template <>
::musa::dnn::Tensor::Type MusaTensorType<int32_t>() {
  return ::musa::dnn::Tensor::Type::INT32;
}

template <>
::musa::dnn::Tensor::Type MusaTensorType<int64_t>() {
  return ::musa::dnn::Tensor::Type::INT64;
}

template <>
::musa::dnn::Tensor::Type MusaTensorType<float>() {
  return ::musa::dnn::Tensor::Type::FLOAT;
}

template <>
::musa::dnn::Tensor::Type MusaTensorType<double>() {
  return ::musa::dnn::Tensor::Type::DOUBLE;
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
void ProbeCum(const char* dtype_name) {
  const std::vector<int64_t> shape{4, 32};
  const size_t count = static_cast<size_t>(shape[0] * shape[1]);
  void* input = nullptr;
  void* output = nullptr;
  ::musa::dnn::Status status = ::musa::dnn::Status::INTERNAL_ERROR;

  auto free_buffers = [&]() {
    if (input != nullptr) {
      EXPECT_EQ(musaFree(input), musaSuccess);
    }
    if (output != nullptr) {
      EXPECT_EQ(musaFree(output), musaSuccess);
    }
  };

  ASSERT_EQ(musaMalloc(&input, count * sizeof(T)), musaSuccess);
  ASSERT_EQ(musaMalloc(&output, count * sizeof(T)), musaSuccess);
  ASSERT_EQ(musaMemset(input, 1, count * sizeof(T)), musaSuccess);
  ASSERT_EQ(musaMemset(output, 0, count * sizeof(T)), musaSuccess);

  ::musa::dnn::Handle handle;
  ::musa::dnn::Tensor input_tensor;
  ::musa::dnn::Tensor output_tensor;
  ASSERT_TRUE(SetupTensor<T>(input_tensor, input, shape));
  ASSERT_TRUE(SetupTensor<T>(output_tensor, output, shape));

  try {
    ::musa::dnn::Cum cum;
    status = cum.SetDim(1);
    if (status == ::musa::dnn::Status::SUCCESS) {
      status = cum.SetMode(::musa::dnn::Cum::Mode::ADD);
    }
    if (status == ::musa::dnn::Status::SUCCESS) {
      status = cum.Run(handle, output_tensor, input_tensor, ::musa::dnn::MemoryMaintainer());
      if (musaDeviceSynchronize() != musaSuccess) {
        status = ::musa::dnn::Status::EXECUTION_FAILED;
      }
    }
  } catch (...) {
    status = ::musa::dnn::Status::INTERNAL_ERROR;
  }

  std::cout << "CumSumApiMicroprobe"
            << " api=Cum"
            << " dtype=" << dtype_name
            << " shape=4x32"
            << " dim=1"
            << " mode=ADD"
            << " exclusive_supported=false"
            << " reverse_supported=false"
            << " status=" << StatusName(status)
            << " status_code=" << static_cast<int>(status)
            << std::endl;

  free_buffers();
}

}  // namespace

TEST(MusaCumSumApiMicroprobeTest, CumInt32Axis1Add) {
  ASSERT_EQ(musaSetDevice(0), musaSuccess);
  ProbeCum<int32_t>("int32");
}

TEST(MusaCumSumApiMicroprobeTest, CumInt64Axis1Add) {
  ASSERT_EQ(musaSetDevice(0), musaSuccess);
  ProbeCum<int64_t>("int64");
}

TEST(MusaCumSumApiMicroprobeTest, CumFloatAxis1Add) {
  ASSERT_EQ(musaSetDevice(0), musaSuccess);
  ProbeCum<float>("float");
}

TEST(MusaCumSumApiMicroprobeTest, CumDoubleAxis1Add) {
  ASSERT_EQ(musaSetDevice(0), musaSuccess);
  ProbeCum<double>("double");
}

}  // namespace test
}  // namespace onnxruntime
