// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include <musa_runtime.h>
#include <cooperative_groups.h>
#include <stdint.h>
#include "slice_kernel.h"

// SliceKernelParams is defined in slice_kernel.h

// 统一的Slice GPU kernel，支持正负步长
template<typename T>
__global__ void SliceKernelUnified(
    const T* input_data,
    T* output_data,
    SliceKernelParams params) {
    
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_threads = gridDim.x * blockDim.x;
    
    // 每个线程处理多个元素，提高GPU利用率
    for (int64_t out_idx = tid; out_idx < params.total_elements; 
         out_idx += total_threads) {
        
        // 计算输出张量的多维坐标
        int64_t coords[kMaxSliceDimensions];
        int64_t temp = out_idx;
        
        // 从最高维开始逐层计算坐标
        for (int dim = params.ndim - 1; dim >= 0; dim--) {
            coords[dim] = temp % params.output_dims[dim];
            temp /= params.output_dims[dim];
        }
        
        // 计算输入索引（核心逻辑：支持负步长）
        int64_t input_idx = 0;
        for (int dim = 0; dim < params.ndim; dim++) {
            // 这里复现了递归拷贝的核心逻辑
            int64_t src_coord = params.starts[dim] + 
                               coords[dim] * params.steps[dim];
            
            // 边界保护（防止越界访问）
            if (src_coord < 0) src_coord = 0;
            if (src_coord >= params.input_dims[dim]) 
                src_coord = params.input_dims[dim] - 1;
            
            input_idx += src_coord * params.input_strides[dim];
        }
        
        // 执行数据拷贝
        output_data[out_idx] = input_data[input_idx];
    }
}

// 包装函数，供C++调用
template<typename T>
void LaunchSliceKernel(
    const T* input_data,
    T* output_data,
    const SliceKernelParams& params,
    musaStream_t stream) {
    
    if (params.total_elements == 0) return;
    
    // 计算最优的grid和block配置
    const int threads_per_block = 256;  // 经验值，适合大多数MUSA GPU
    const int max_blocks = 65535;       // MUSA GPU限制
    
    int64_t blocks = (params.total_elements + threads_per_block - 1) / threads_per_block;
    blocks = min(blocks, (int64_t)max_blocks);
    
    // 启动kernel
    SliceKernelUnified<T><<<(int)blocks, threads_per_block, 0, stream>>>(
        input_data, output_data, params);
}

// 显式模板实例化，支持ORT常用的数据类型
template void LaunchSliceKernel<float>(
    const float*, float*, const SliceKernelParams&, musaStream_t);
    
template void LaunchSliceKernel<double>(
    const double*, double*, const SliceKernelParams&, musaStream_t);
    
template void LaunchSliceKernel<int32_t>(
    const int32_t*, int32_t*, const SliceKernelParams&, musaStream_t);
    
template void LaunchSliceKernel<int64_t>(
    const int64_t*, int64_t*, const SliceKernelParams&, musaStream_t);

template void LaunchSliceKernel<int16_t>(
    const int16_t*, int16_t*, const SliceKernelParams&, musaStream_t);

template void LaunchSliceKernel<int8_t>(
    const int8_t*, int8_t*, const SliceKernelParams&, musaStream_t);

template void LaunchSliceKernel<uint8_t>(
    const uint8_t*, uint8_t*, const SliceKernelParams&, musaStream_t);

// MLFloat16 needs special handling - include ORT headers for the type
#include "core/common/common.h"  
#include "core/common/float16.h"

template void LaunchSliceKernel<onnxruntime::MLFloat16>(
    const onnxruntime::MLFloat16*, onnxruntime::MLFloat16*, const SliceKernelParams&, musaStream_t);