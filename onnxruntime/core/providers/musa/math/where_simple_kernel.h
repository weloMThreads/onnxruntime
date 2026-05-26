// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Simple Where GPU kernel - C interface for maximum compatibility
// Minimal dependencies, no ORT framework headers

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
    void* stream);

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
    void* stream);

void launch_where_kernel_int32(
    const bool* cond_data,
    const int* x_data,  // Use standard int for C compatibility
    const int* y_data,
    int* output_data,
    long long total_elements,
    const long long* cond_shape,
    const long long* x_shape,
    const long long* y_shape,
    const long long* output_shape,
    int rank,
    void* stream);

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
    void* stream);

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
    void* stream);

#ifdef __cplusplus
}
#endif
