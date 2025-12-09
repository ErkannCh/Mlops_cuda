#pragma once
#include <cuda_runtime.h>

void gpu_matrix_add(const float* d_A, const float* d_B, float* d_C, int rows, int cols);

void gpu_matrix_mul(const float* d_A, const float* d_B, float* d_C, int M, int K, int N);

void gpu_feedforward_layer(const float* d_input, const float* d_weight, const float* d_bias, float* d_output, int batch_size, int in_features, int out_features);

__global__ void feedforward_layer_kernel(const float* input, const float* weight, const float* bias, float* output, int batch, int in_f, int out_f);

__global__ void feedforward_layer_kernel_optimized(
    const float* __restrict__ input,
    const float* __restrict__ weight,  
    const float* __restrict__ bias,    
    float* __restrict__ output,        
    int batch, int in_f, int out_f);
 
void gpu_relu(float* d_data, int size);

void gpu_tanh(float* d_data, int size);

void checkCuda(cudaError_t result, const char* msg);

void gpu_conv2d_naive(const float* d_input, const float* d_weight, const float* d_bias, float* d_output, int N, int C_in, int H, int W, int C_out, int K, int H_out, int W_out);

void gpu_conv2d_relu_naive(const float* d_input, const float* d_weight, const float* d_bias, float* d_output, int N, int C_in, int H, int W, int C_out, int K, int H_out, int W_out);
