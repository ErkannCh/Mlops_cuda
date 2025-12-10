#include <iostream>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "cuda_kernels.cuh"

inline void checkCublas(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS Error: " << msg << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

inline void checkCuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error (" << msg << "): " 
                  << cudaGetErrorString(result) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

//
// ────────────────────────────────────────────────────────────────
// bias add and tanh
// ────────────────────────────────────────────────────────────────
//

__global__ void bias_add_and_tanh_kernel(
    float* d_output,
    const float* d_pre_act,
    const float* d_bias,
    int num_elements,
    int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_elements) {
        int bias_idx = idx % hidden_dim; 

        float activation_input = d_pre_act[idx] + d_bias[bias_idx];
        d_output[idx] = tanhf(activation_input);
    }
}

void launch_bias_add_and_tanh_kernel(
    float* d_output,
    const float* d_pre_act,
    const float* d_bias,
    int num_elements,
    int hidden_dim,
    cudaStream_t stream
) {
    const int BLOCK_SIZE = 256;
    int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    bias_add_and_tanh_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        d_output, d_pre_act, d_bias, num_elements, hidden_dim
    );
    
    checkCuda(cudaGetLastError(), "bias_add_and_tanh_kernel launch failed");
}

//
// ────────────────────────────────────────────────────────────────
// Matrix Add
// ────────────────────────────────────────────────────────────────
//

__global__ void matrix_add_tail_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int start_idx, int size)
{
    int idx = start_idx + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        C[idx] = A[idx] + B[idx];
}

__global__ void matrix_add_vec4_kernel(const float4* __restrict__ A,
                                       const float4* __restrict__ B,
                                       float4* __restrict__ C,
                                       int vec4_size) // size / 4
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec4_size) {
        float4 a = A[idx];
        float4 b = B[idx];
        C[idx] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
}

void gpu_matrix_add_optimized(const float* d_A, const float* d_B, float* d_C,
                              int rows, int cols)
{
    int size = rows * cols;
    constexpr int blockSize = 256;

    constexpr int vec4_elements = 4;
    int vec4_size = size / vec4_elements;

    if (vec4_size > 0) {
        int vec4_gridSize = (vec4_size + blockSize - 1) / blockSize;

        matrix_add_vec4_kernel<<<vec4_gridSize, blockSize>>>(
            (const float4*)d_A, 
            (const float4*)d_B, 
            (float4*)d_C, 
            vec4_size
        );
        checkCuda(cudaGetLastError(), "matrix_add_vec4_kernel launch");
    }

    int tail_size = size % vec4_elements;
    int start_idx = size - tail_size;    

    if (tail_size > 0) {
        int tail_gridSize = 1; 

        matrix_add_tail_kernel<<<tail_gridSize, blockSize>>>(
            d_A, 
            d_B, 
            d_C, 
            start_idx, 
            size
        );
        checkCuda(cudaGetLastError(), "matrix_add_tail_kernel launch");
    }
}

void gpu_matrix_mul_cublas(cublasHandle_t handle,
                           const float* d_A, 
                           const float* d_B, 
                           float* d_C,
                           int M, int K, int N)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    checkCublas(cublasSgemm(handle, 
                          CUBLAS_OP_N,
                          CUBLAS_OP_N,
                          N, M, K,     
                          &alpha,
                          d_B, K,      
                          d_A, M,      
                          &beta,
                          d_C, N), "CUBLAS Sgemm");
}


__global__ void fused_add_tanh_kernel(const float* __restrict__ A, 
                                      const float* __restrict__ B, 
                                      float* __restrict__ C,       
                                      int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        float sum = A[idx] + B[idx];

        C[idx] = tanhf(sum);
    }
}

void gpu_fused_add_tanh(const float* d_lin_x, 
                        const float* d_lin_h, 
                        float* d_h_t, 
                        int batch_size, 
                        int hidden_dim)
{
    int size = batch_size * hidden_dim;

    constexpr int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    // Lancement du kernel
    fused_add_tanh_kernel<<<gridSize, blockSize>>>(
        d_lin_x, 
        d_lin_h, 
        d_h_t, 
        size
    );

    // Vérification des erreurs CUDA après le lancement
    checkCuda(cudaGetLastError(), "fused_add_tanh_kernel launch");
}

__global__ void fused_add_bias_tanh_kernel(const float* __restrict__ d_lin_x,      
                                           const float* __restrict__ d_lin_h,      
                                           const float* __restrict__ d_b_h,   
                                           float* __restrict__ d_h_t,            
                                           int batch_size, 
                                           int hidden_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = batch_size * hidden_dim;

    if (idx < size)
    {
        int bias_idx = idx % hidden_dim; 

        float sum = d_lin_x[idx] + d_lin_h[idx] + d_b_h[bias_idx];

        d_h_t[idx] = tanhf(sum);
    }
}

void gpu_fused_add_bias_tanh(const float* d_lin_x, 
                             const float* d_lin_h, 
                             const float* d_b_h, 
                             float* d_h_t, 
                             int batch_size, 
                             int hidden_dim, 
                             cudaStream_t stream)
{
    int size = batch_size * hidden_dim;
    constexpr int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    fused_add_bias_tanh_kernel<<<gridSize, blockSize, 0, stream>>>(
        d_lin_x, 
        d_lin_h, 
        d_b_h, 
        d_h_t, 
        batch_size, 
        hidden_dim
    );
    checkCuda(cudaGetLastError(), "fused_add_bias_tanh_kernel launch"); 
}

//
// ────────────────────────────────────────────────────────────────
// Optimized Matrix Multiplication (light tiling)
// ────────────────────────────────────────────────────────────────
//

__global__ void matrix_mul_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int K, int N)
{
    __shared__ float tileA[16][16];
    __shared__ float tileB[16][16];

    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;

    float sum = 0.f;

    for (int t = 0; t < (K + 15) / 16; t++) {
        if (row < M && t * 16 + threadIdx.x < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + t * 16 + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0;

        if (col < N && t * 16 + threadIdx.y < K)
            tileB[threadIdx.y][threadIdx.x] = B[(t * 16 + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < 16; ++k)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

void gpu_matrix_mul(const float* d_A, const float* d_B, float* d_C,
                    int M, int K, int N)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((N + 15) / 16, (M + 15) / 16);

    matrix_mul_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);
    checkCuda(cudaGetLastError(), "matrix_mul_kernel launch");
}

//
// ────────────────────────────────────────────────────────────────
// Feedforward Layer (dense)
// ────────────────────────────────────────────────────────────────
//

__global__ void feedforward_layer_kernel(const float* __restrict__ input,
                                         const float* __restrict__ weight,
                                         const float* __restrict__ bias,
                                         float* __restrict__ output,
                                         int batch, int in_f, int out_f)
{
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch && j < out_f) {
        float sum = bias[j];

        #pragma unroll 4
        for (int i = 0; i < in_f; i++)
            sum += input[b * in_f + i] * weight[j * in_f + i];

        output[b * out_f + j] = sum;
    }
}

void gpu_feedforward_layer(const float* d_input, const float* d_weight,
                           const float* d_bias, float* d_output,
                           int batch, int in_f, int out_f)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((out_f + 15) / 16, (batch + 15) / 16);

    feedforward_layer_kernel<<<gridSize, blockSize>>>(
        d_input, d_weight, d_bias, d_output,
        batch, in_f, out_f);

    checkCuda(cudaGetLastError(), "feedforward_layer_kernel launch");
}

__global__ void feedforward_layer_kernel_optimized(
    const float* __restrict__ input,   
    const float* __restrict__ weight, 
    const float* __restrict__ bias,    
    float* __restrict__ output,        
    int batch, int in_f, int out_f)
{
    extern __shared__ float shared[];

    int b = blockIdx.y;
    int j = blockIdx.x;

    float sum = 0.0f;

    for(int i = threadIdx.x; i < in_f; i += blockDim.x)
        sum += input[b * in_f + i] * weight[j * in_f + i];

    shared[threadIdx.x] = sum;
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = 0.0f;
        for(int k = 0; k < blockDim.x; k++)
            total += shared[k];
        output[b * out_f + j] = total + bias[j];
    }
}

//
// ────────────────────────────────────────────────────────────────
// ReLU
// ────────────────────────────────────────────────────────────────
//

__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        data[idx] = fmaxf(0.0f, data[idx]);
}

void gpu_relu(float* d_data, int size) {
    constexpr int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    relu_kernel<<<gridSize, blockSize>>>(d_data, size);
    checkCuda(cudaGetLastError(), "relu_kernel launch");
}

//
// ────────────────────────────────────────────────────────────────
// Conv2D naive optimized
// ────────────────────────────────────────────────────────────────
//

__global__ void conv2d_naive_kernel(const float* __restrict__ input,
                                    const float* __restrict__ weight,
                                    const float* __restrict__ bias,
                                    float* __restrict__ output,
                                    int N, int C_in, int H, int W,
                                    int C_out, int K, int H_out, int W_out,
                                    bool apply_relu)
{
    int n  = blockIdx.z;
    int co = blockIdx.y * blockDim.y + threadIdx.y;
    int y  = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= N || co >= C_out || y >= H_out)
        return;

    for (int x = 0; x < W_out; x++) {

        float sum = (bias ? bias[co] : 0.0f);

        int out_idx = (((n * C_out + co) * H_out + y) * W_out + x);

        for (int ci = 0; ci < C_in; ci++) {
            int base_input = ((n * C_in + ci) * H + y) * W + x;
            int base_weight = ((co * C_in + ci) * K) * K;

            for (int ky = 0; ky < K; ky++) {
                int in_offset_y = base_input + ky * W;
                int w_offset_y  = base_weight + ky * K;

                #pragma unroll 4
                for (int kx = 0; kx < K; kx++) {
                    sum += input[in_offset_y + kx] *
                           weight[w_offset_y + kx];
                }
            }
        }

        if (apply_relu)
            sum = fmaxf(sum, 0.0f);

        output[out_idx] = sum;
    }
}

void gpu_conv2d_naive(const float* d_input, const float* d_weight,
                      const float* d_bias, float* d_output,
                      int N, int C_in, int H, int W,
                      int C_out, int K, int H_out, int W_out)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((H_out + 15) / 16, (C_out + 15) / 16, N);

    conv2d_naive_kernel<<<gridSize, blockSize>>>(
        d_input, d_weight, d_bias, d_output,
        N, C_in, H, W, C_out, K, H_out, W_out,
        false);

    checkCuda(cudaGetLastError(), "conv2d_naive_kernel launch");
}

void gpu_conv2d_relu_naive(const float* d_input, const float* d_weight,
                           const float* d_bias, float* d_output,
                           int N, int C_in, int H, int W,
                           int C_out, int K, int H_out, int W_out)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((H_out + 15) / 16, (C_out + 15) / 16, N);

    conv2d_naive_kernel<<<gridSize, blockSize>>>(
        d_input, d_weight, d_bias, d_output,
        N, C_in, H, W, C_out, K, H_out, W_out,
        true);

    checkCuda(cudaGetLastError(), "conv2d_relu_naive_kernel launch");
}

//
// ────────────────────────────────────────────────────────────────
// Tanh
// ────────────────────────────────────────────────────────────────
//

__global__ void tanh_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        data[idx] = tanhf(data[idx]);
}

void gpu_tanh(float* d_data, int size) {
    constexpr int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    tanh_kernel<<<gridSize, blockSize>>>(d_data, size);
    checkCuda(cudaGetLastError(), "tanh_kernel launch");
}
