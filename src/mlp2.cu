#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <fstream>
#include <iostream>

#include "cuda_kernels.cuh"
#include "mlp2.hpp"
#include "utils.hpp"


MLP2::MLP2(const MLPConfig2& cfg) : config_(cfg) {
    h_W1.resize(config_.hidden_dim * config_.input_dim);
    h_b1.resize(config_.hidden_dim);
    h_W2.resize(config_.output_dim * config_.hidden_dim);
    h_b2.resize(config_.output_dim);

    checkCublas(cublasCreate(&cublas_handle_), "cublasCreate");
}

MLP2::~MLP2() {
    if (cublas_handle_) cublasDestroy(cublas_handle_);

    if (d_W1) cudaFree(d_W1);
    if (d_b1) cudaFree(d_b1);
    if (d_W2) cudaFree(d_W2);
    if (d_b2) cudaFree(d_b2);

    if (d_input) cudaFree(d_input);
    if (d_hidden) cudaFree(d_hidden);
    if (d_output) cudaFree(d_output);
}


void MLP2::allocate_device_buffers(int batch_size) {
    if (batch_size <= current_buffer_batch_size_) return;

    if (d_input) cudaFree(d_input);
    if (d_hidden) cudaFree(d_hidden);
    if (d_output) cudaFree(d_output);

    checkCuda(cudaMalloc(&d_input, sizeof(float) * batch_size * config_.input_dim), "cudaMalloc d_input");
    checkCuda(cudaMalloc(&d_hidden, sizeof(float) * batch_size * config_.hidden_dim), "cudaMalloc d_hidden");
    checkCuda(cudaMalloc(&d_output, sizeof(float) * batch_size * config_.output_dim), "cudaMalloc d_output");

    current_buffer_batch_size_ = batch_size;
}

void MLP2::copy_weights_to_device() {
    if (!d_W1) {
        checkCuda(cudaMalloc(&d_W1, sizeof(float) * h_W1.size()), "cudaMalloc d_W1");
        checkCuda(cudaMalloc(&d_b1, sizeof(float) * h_b1.size()), "cudaMalloc d_b1");
        checkCuda(cudaMalloc(&d_W2, sizeof(float) * h_W2.size()), "cudaMalloc d_W2");
        checkCuda(cudaMalloc(&d_b2, sizeof(float) * h_b2.size()), "cudaMalloc d_b2");
    }

    checkCuda(cudaMemcpy(d_W1, h_W1.data(), sizeof(float) * h_W1.size(), cudaMemcpyHostToDevice), "memcpy W1");
    checkCuda(cudaMemcpy(d_b1, h_b1.data(), sizeof(float) * h_b1.size(), cudaMemcpyHostToDevice), "memcpy b1");
    checkCuda(cudaMemcpy(d_W2, h_W2.data(), sizeof(float) * h_W2.size(), cudaMemcpyHostToDevice), "memcpy W2");
    checkCuda(cudaMemcpy(d_b2, h_b2.data(), sizeof(float) * h_b2.size(), cudaMemcpyHostToDevice), "memcpy b2");
}

void MLP2::load_weights_from_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Impossible d'ouvrir les poids: " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int in_dim, hid_dim, out_dim;
    f >> in_dim >> hid_dim >> out_dim;
    if (in_dim != config_.input_dim || hid_dim != config_.hidden_dim || out_dim != config_.output_dim) {
        std::cerr << "Dimensions de poids incompatibles avec la config." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    for (auto& v : h_W1) f >> v;

    for (auto& v : h_b1) f >> v;

    for (auto& v : h_W2) f >> v;

    for (auto& v : h_b2) f >> v;

    copy_weights_to_device();
}


void MLP2::forward(const float* input_host, float* output_host, int batch_size) {
    allocate_device_buffers(batch_size);

    checkCuda(cudaMemcpy(d_input, input_host,
                         sizeof(float) * batch_size * config_.input_dim,
                         cudaMemcpyHostToDevice),
              "memcpy input_host -> d_input");

    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    const int M1 = batch_size;
    const int K1 = config_.input_dim;
    const int N1 = config_.hidden_dim;

    checkCublas(cublasSgemm(cublas_handle_,
                          CUBLAS_OP_N,  
                          CUBLAS_OP_T,  
                          N1,           
                          M1,           
                          K1,       
                          &alpha,      
                          d_W1,     
                          N1,       
                          d_input,  
                          M1,       
                          &beta, 
                          d_hidden, 
                          N1),          
              "cublasSgemm layer1");
    
    {
        int threads = 256;
        int elements = batch_size * config_.hidden_dim;
        int blocks = (elements + threads - 1) / threads;

        bias_and_relu_kernel<<<blocks, threads>>>(
            d_hidden, d_b1, batch_size, config_.hidden_dim
        );
        checkCuda(cudaGetLastError(), "bias_and_relu_kernel layer1");
    }

    const int M2 = batch_size;
    const int K2 = config_.hidden_dim;
    const int N2 = config_.output_dim;

    checkCublas(cublasSgemm(cublas_handle_,
                          CUBLAS_OP_N,  
                          CUBLAS_OP_T, 
                          N2,       
                          M2,         
                          K2,       
                          &alpha,      
                          d_W2,       
                          N2,           
                          d_hidden,    
                          M2,           
                          &beta,        
                          d_output,     
                          N2),          
              "cublasSgemm layer2");
    
    {
        int threads = 256;
        int elements = batch_size * config_.output_dim;
        int blocks = (elements + threads - 1) / threads;

        bias_and_relu_kernel<<<blocks, threads>>>(
            d_output, d_b2, batch_size, config_.output_dim
        );
        checkCuda(cudaGetLastError(), "bias_and_relu_kernel layer2");
    }

    checkCuda(cudaMemcpy(output_host, d_output,
                         sizeof(float) * batch_size * config_.output_dim,
                         cudaMemcpyDeviceToHost),
              "memcpy d_output -> output_host");
}