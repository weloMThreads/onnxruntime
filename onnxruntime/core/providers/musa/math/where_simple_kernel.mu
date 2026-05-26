// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Moore Threads. All rights reserved.
// Licensed under the MIT License.

#include <musa_runtime.h>
#include <musa_fp16.h>
#include <stdint.h>

// Simple Where GPU kernel for MUSA - minimal dependencies approach
// Based on successful Slice kernel pattern

// Improved kernel with proper broadcasting support
template<typename T>
__global__ void SimpleWhereKernel(
    const bool* cond_data,
    const T* x_data,
    const T* y_data,
    T* output_data,
    long long total_elements,
    // Shape information for broadcasting
    const long long* cond_shape,
    const long long* x_shape, 
    const long long* y_shape,
    const long long* output_shape,
    int rank) {
    
    long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long long total_threads = gridDim.x * blockDim.x;
    
    // Grid-stride loop
    for (long long idx = tid; idx < total_elements; idx += total_threads) {
        // Convert linear index to multi-dimensional coordinate
        long long temp_idx = idx;
        long long output_coord[8];  // Support up to 8D tensors
        long long cond_idx = 0, x_idx = 0, y_idx = 0;
        long long output_stride = 1;
        
        // Calculate output coordinates (right to left)
        for (int d = rank - 1; d >= 0; d--) {
            output_coord[d] = temp_idx % output_shape[d];
            temp_idx /= output_shape[d];
        }
        
        // Calculate input indices with broadcasting rules
        long long cond_stride = 1, x_stride = 1, y_stride = 1;
        for (int d = rank - 1; d >= 0; d--) {
            // Condition index
            long long cond_coord = (cond_shape[d] == 1) ? 0 : output_coord[d];
            cond_idx += cond_coord * cond_stride;
            cond_stride *= cond_shape[d];
            
            // X index  
            long long x_coord = (x_shape[d] == 1) ? 0 : output_coord[d];
            x_idx += x_coord * x_stride;
            x_stride *= x_shape[d];
            
            // Y index
            long long y_coord = (y_shape[d] == 1) ? 0 : output_coord[d];
            y_idx += y_coord * y_stride;
            y_stride *= y_shape[d];
        }
        
        // Where operation
        output_data[idx] = cond_data[cond_idx] ? x_data[x_idx] : y_data[y_idx];
    }
}

// Phase 2.1 Optimization: Adaptive kernel parameters based on workload
__device__ __host__ void GetOptimalKernelParams(long long total_elements, int* threads_per_block, int* num_blocks) {
    // Adaptive configuration based on workload size
    if (total_elements <= 1024) {
        // Small workload: use smaller blocks to reduce overhead
        *threads_per_block = 64;
    } else if (total_elements <= 32768) {
        // Medium workload: balanced configuration  
        *threads_per_block = 128;
    } else if (total_elements <= 1048576) {
        // Large workload: standard configuration
        *threads_per_block = 256;
    } else {
        // Very large workload: maximize occupancy
        *threads_per_block = 512;
    }
    
    // Calculate optimal number of blocks
    long long blocks = (total_elements + *threads_per_block - 1) / *threads_per_block;
    
    // Limit blocks based on workload size to avoid resource waste
    int max_blocks;
    if (total_elements <= 65536) {
        max_blocks = 256;    // Small to medium: limited blocks
    } else if (total_elements <= 1048576) {
        max_blocks = 2048;   // Large: moderate blocks
    } else {
        max_blocks = 8192;   // Very large: many blocks (but not max 65535)
    }
    
    *num_blocks = (int)((blocks > max_blocks) ? max_blocks : blocks);
}

extern "C" {

// C wrapper functions for different types
void launch_where_kernel_float(
    const bool* cond_data,
    const float* x_data,
    const float* y_data,
    float* output_data,
    long long total_elements,
    const long long* cond_shape,
    const long long* x_shape,
    const long long* y_shape,
    const long long* output_shape,
    int rank,
    void* stream) {
    
    // Phase 2.1: Use adaptive kernel parameters for optimal performance
    int threads_per_block, num_blocks;
    GetOptimalKernelParams(total_elements, &threads_per_block, &num_blocks);
    
    musaStream_t musa_stream = static_cast<musaStream_t>(stream);
    
    SimpleWhereKernel<float><<<num_blocks, threads_per_block, 0, musa_stream>>>(
        cond_data, x_data, y_data, output_data, total_elements,
        cond_shape, x_shape, y_shape, output_shape, rank);
}

void launch_where_kernel_double(
    const bool* cond_data,
    const double* x_data,
    const double* y_data,
    double* output_data,
    long long total_elements,
    const long long* cond_shape,
    const long long* x_shape,
    const long long* y_shape,
    const long long* output_shape,
    int rank,
    void* stream) {
    
    // Phase 2.1: Use adaptive kernel parameters for optimal performance  
    int threads_per_block, num_blocks;
    GetOptimalKernelParams(total_elements, &threads_per_block, &num_blocks);
    
    musaStream_t musa_stream = static_cast<musaStream_t>(stream);
    
    SimpleWhereKernel<double><<<num_blocks, threads_per_block, 0, musa_stream>>>(
        cond_data, x_data, y_data, output_data, total_elements,
        cond_shape, x_shape, y_shape, output_shape, rank);
}

void launch_where_kernel_int32(
    const bool* cond_data,
    const int* x_data,
    const int* y_data,
    int* output_data,
    long long total_elements,
    const long long* cond_shape,
    const long long* x_shape,
    const long long* y_shape,
    const long long* output_shape,
    int rank,
    void* stream) {
    
    // Phase 2.1: Use adaptive kernel parameters for optimal performance  
    int threads_per_block, num_blocks;
    GetOptimalKernelParams(total_elements, &threads_per_block, &num_blocks);
    
    musaStream_t musa_stream = static_cast<musaStream_t>(stream);
    
    SimpleWhereKernel<int><<<num_blocks, threads_per_block, 0, musa_stream>>>(
        cond_data, x_data, y_data, output_data, total_elements,
        cond_shape, x_shape, y_shape, output_shape, rank);
}

void launch_where_kernel_int64(
    const bool* cond_data,
    const long long* x_data,
    const long long* y_data,
    long long* output_data,
    long long total_elements,
    const long long* cond_shape,
    const long long* x_shape,
    const long long* y_shape,
    const long long* output_shape,
    int rank,
    void* stream) {
    
    // Phase 2.1: Use adaptive kernel parameters for optimal performance  
    int threads_per_block, num_blocks;
    GetOptimalKernelParams(total_elements, &threads_per_block, &num_blocks);
    
    musaStream_t musa_stream = static_cast<musaStream_t>(stream);
    
    SimpleWhereKernel<long long><<<num_blocks, threads_per_block, 0, musa_stream>>>(
        cond_data, x_data, y_data, output_data, total_elements,
        cond_shape, x_shape, y_shape, output_shape, rank);
}

void launch_where_kernel_half(
    const bool* cond_data,
    const void* x_data,
    const void* y_data,
    void* output_data,
    long long total_elements,
    const long long* cond_shape,
    const long long* x_shape,
    const long long* y_shape,
    const long long* output_shape,
    int rank,
    void* stream) {

    int threads_per_block, num_blocks;
    GetOptimalKernelParams(total_elements, &threads_per_block, &num_blocks);

    musaStream_t musa_stream = static_cast<musaStream_t>(stream);

    SimpleWhereKernel<half><<<num_blocks, threads_per_block, 0, musa_stream>>>(
        cond_data,
        reinterpret_cast<const half*>(x_data),
        reinterpret_cast<const half*>(y_data),
        reinterpret_cast<half*>(output_data),
        total_elements,
        cond_shape, x_shape, y_shape, output_shape, rank);
}

}  // extern "C"
