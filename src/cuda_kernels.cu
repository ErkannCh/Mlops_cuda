#include <iostream>

#include "cuda_kernels.cuh"

void checkCuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error (" << msg << "): " << cudaGetErrorString(result) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

__global__ void matrix_add_kernel(const float* A, const float* B, float* C, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = rows * cols;
    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

void gpu_matrix_add(const float* d_A, const float* d_B, float* d_C, int rows, int cols) {
    int size = rows * cols;
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    matrix_add_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, rows, cols);
    checkCuda(cudaGetLastError(), "matrix_add_kernel launch");
}

__global__ void matrix_mul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void gpu_matrix_mul(const float* d_A, const float* d_B, float* d_C, int M, int K, int N) {
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    matrix_mul_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);
    checkCuda(cudaGetLastError(), "matrix_mul_kernel launch");
}

__global__ void feedforward_layer_kernel(const float* input, const float* weight, const float* bias, float* output, int batch_size, int in_features, int out_features) {
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch_size && j < out_features) {
        float sum = 0.0f;
        for (int i = 0; i < in_features; ++i) {
            float x = input[b * in_features + i];
            float w = weight[j * in_features + i];
            sum += x * w;
        }
        sum += bias[j];
        output[b * out_features + j] = sum;
    }
}

void gpu_feedforward_layer(const float* d_input, const float* d_weight, const float* d_bias, float* d_output, int batch_size, int in_features, int out_features) {
    dim3 blockSize(16, 16);
    dim3 gridSize((out_features + blockSize.x - 1) / blockSize.x, (batch_size + blockSize.y - 1) / blockSize.y);

    feedforward_layer_kernel<<<gridSize, blockSize>>>(d_input, d_weight, d_bias, d_output, batch_size, in_features, out_features);
    checkCuda(cudaGetLastError(), "feedforward_layer_kernel launch");
}

__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = data[idx];
        data[idx] = x > 0.0f ? x : 0.0f;
    }
}

void gpu_relu(float* d_data, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    relu_kernel<<<gridSize, blockSize>>>(d_data, size);
    checkCuda(cudaGetLastError(), "relu_kernel launch");
}

__global__ void conv2d_naive_kernel(const float* input, const float* weight, const float* bias, float* output, int N, int C_in, int H, int W, int C_out, int K, int H_out, int W_out, bool apply_relu) {
    int n = blockIdx.z;
    int co = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= N || co >= C_out || y >= H_out) return;

    for (int x = 0; x < W_out; ++x) {
        float sum = bias ? bias[co] : 0.0f;

        for (int ci = 0; ci < C_in; ++ci) {
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    int in_y = y + ky;
                    int in_x = x + kx;

                    float v_in = input[(((n * C_in + ci) * H) + in_y) * W + in_x];
                    float v_w = weight[(((co * C_in + ci) * K) + ky) * K + kx];

                    sum += v_in * v_w;
                }
            }
        }

        if (apply_relu && sum < 0.0f) sum = 0.0f;

        output[(((n * C_out + co) * H_out) + y) * W_out + x] = sum;
    }
}

void gpu_conv2d_naive(const float* d_input, const float* d_weight, const float* d_bias, float* d_output, int N, int C_in, int H, int W, int C_out, int K, int H_out, int W_out) {
    dim3 blockSize(16, 16);
    dim3 gridSize((H_out + blockSize.x - 1) / blockSize.x, (C_out + blockSize.y - 1) / blockSize.y, N);

    conv2d_naive_kernel<<<gridSize, blockSize>>>(d_input, d_weight, d_bias, d_output, N, C_in, H, W, C_out, K, H_out, W_out, false);
    checkCuda(cudaGetLastError(), "conv2d_naive_kernel launch");
}

void gpu_conv2d_relu_naive(const float* d_input, const float* d_weight, const float* d_bias, float* d_output, int N, int C_in, int H, int W, int C_out, int K, int H_out, int W_out) {
    dim3 blockSize(16, 16);
    dim3 gridSize((H_out + blockSize.x - 1) / blockSize.x, (C_out + blockSize.y - 1) / blockSize.y, N);

    conv2d_naive_kernel<<<gridSize, blockSize>>>(d_input, d_weight, d_bias, d_output, N, C_in, H, W, C_out, K, H_out, W_out, true);
    checkCuda(cudaGetLastError(), "conv2d_relu_naive_kernel launch");
}

__global__ void tanh_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = tanhf(data[idx]);
    }
}

void gpu_tanh(float* d_data, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    tanh_kernel<<<gridSize, blockSize>>>(d_data, size);
    checkCuda(cudaGetLastError(), "tanh_kernel launch");
}