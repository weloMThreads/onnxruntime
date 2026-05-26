// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <musa_runtime.h>
#include <stdint.h>

// 常量定义
constexpr int32_t kMaxSliceDimensions = 8;

// 前向声明，避免在头文件中包含.mu文件的内容
struct SliceKernelParams {
    int32_t ndim;                              // 张量的维度数
    int64_t total_elements;                    // 输出张量的总元素数
    int64_t input_dims[kMaxSliceDimensions];   // 输入张量各维度大小（最多8维）
    int64_t output_dims[kMaxSliceDimensions];  // 输出张量各维度大小
    int64_t input_strides[kMaxSliceDimensions]; // 输入张量各维度步长
    int64_t starts[kMaxSliceDimensions];       // 各维度起始位置
    int64_t steps[kMaxSliceDimensions];        // 各维度步长（支持负值）
};

// GPU kernel启动函数声明
// 这些函数在slice_kernel.mu中实现

template<typename T>
void LaunchSliceKernel(
    const T* input_data,
    T* output_data, 
    const SliceKernelParams& params,
    musaStream_t stream);

// 显式声明支持的数据类型，确保链接正确
extern template void LaunchSliceKernel<float>(
    const float*, float*, const SliceKernelParams&, musaStream_t);
    
extern template void LaunchSliceKernel<double>(
    const double*, double*, const SliceKernelParams&, musaStream_t);
    
extern template void LaunchSliceKernel<int32_t>(
    const int32_t*, int32_t*, const SliceKernelParams&, musaStream_t);
    
extern template void LaunchSliceKernel<int64_t>(
    const int64_t*, int64_t*, const SliceKernelParams&, musaStream_t);

extern template void LaunchSliceKernel<int16_t>(
    const int16_t*, int16_t*, const SliceKernelParams&, musaStream_t);

extern template void LaunchSliceKernel<int8_t>(
    const int8_t*, int8_t*, const SliceKernelParams&, musaStream_t);

extern template void LaunchSliceKernel<uint8_t>(
    const uint8_t*, uint8_t*, const SliceKernelParams&, musaStream_t);