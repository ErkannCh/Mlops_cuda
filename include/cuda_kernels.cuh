#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>

void gpu_matrix_add_optimized(const float* d_A, const float* d_B, float* d_C,
                              int rows, int cols);

void launch_bias_add_and_tanh_kernel(
    float* d_output,
    const float* d_pre_act,
    const float* d_bias,
    int num_elements,
    int hidden_dim,
    cudaStream_t stream
);

__global__ void matrix_add_tail_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int start_idx, int size);

__global__ void matrix_add_vec4_kernel(const float4* __restrict__ A,
                                       const float4* __restrict__ B,
                                       float4* __restrict__ C,
                                       int vec4_size);

void gpu_fused_add_tanh(const float* d_lin_x, 
                        const float* d_lin_h, 
                        float* d_h_t, 
                        int batch_size, 
                        int hidden_dim);

void gpu_matrix_mul(const float* d_A, const float* d_B, float* d_C, int M, int K, int N);

void gpu_feedforward_layer(const float* d_input, const float* d_weight, const float* d_bias, float* d_output, int batch_size, int in_features, int out_features);

__global__ void bias_and_relu_kernel(float* output, const float* bias, int batch_size, int out_dim);

__global__ void feedforward_layer_kernel(const float* input, const float* weight, const float* bias, float* output, int batch, int in_f, int out_f);

void gpu_feedforward_layer_optimized_stream(
    const float* d_input, 
    const float* d_weight, 
    const float* d_bias, 
    float* d_output,
    int batch, 
    int in_f, 
    int out_f, 
    cudaStream_t stream);

__global__ void feedforward_layer_kernel_optimized(
    const float* __restrict__ input,
    const float* __restrict__ weight,  
    const float* __restrict__ bias,    
    float* __restrict__ output,        
    int batch, int in_f, int out_f);
 
void gpu_relu(float* d_data, int size);
void gpu_relu2(float* d_data, int size, cudaStream_t stream);

void gpu_tanh(float* d_data, int size);

void gpu_fused_add_bias_tanh(const float* d_lin_x, 
                             const float* d_lin_h, 
                             const float* d_b_h, 
                             float* d_h_t, 
                             int batch_size, 
                             int hidden_dim, 
                             cudaStream_t stream);

void checkCuda(cudaError_t result, const char* msg);

void checkCublas(cublasStatus_t status, const char* msg);

void gpu_conv2d_naive(const float* d_input, const float* d_weight, const float* d_bias, float* d_output, int N, int C_in, int H, int W, int C_out, int K, int H_out, int W_out);

void gpu_conv2d_relu_naive(const float* d_input, const float* d_weight, const float* d_bias, float* d_output, int N, int C_in, int H, int W, int C_out, int K, int H_out, int W_out);
