#include <cuda_runtime.h>

#include <fstream>
#include <iostream>

#include "cuda_kernels.cuh"
#include "mlp.hpp"
#include "utils.hpp"

MLP::MLP(const MLPConfig& cfg) : config_(cfg) {
    h_W1.resize(config_.hidden_dim * config_.input_dim);
    h_b1.resize(config_.hidden_dim);
    h_W2.resize(config_.output_dim * config_.hidden_dim);
    h_b2.resize(config_.output_dim);
}

MLP::~MLP() {
    if (d_W1) cudaFree(d_W1);
    if (d_b1) cudaFree(d_b1);
    if (d_W2) cudaFree(d_W2);
    if (d_b2) cudaFree(d_b2);

    if (d_input) cudaFree(d_input);
    if (d_hidden) cudaFree(d_hidden);
    if (d_output) cudaFree(d_output);
}

void MLP::allocate_device_buffers(int batch_size) {
    if (batch_size <= current_buffer_batch_size_) return;

    if (d_input) cudaFree(d_input);
    if (d_hidden) cudaFree(d_hidden);
    if (d_output) cudaFree(d_output);

    checkCuda(cudaMalloc(&d_input, sizeof(float) * batch_size * config_.input_dim), "cudaMalloc d_input");
    checkCuda(cudaMalloc(&d_hidden, sizeof(float) * batch_size * config_.hidden_dim), "cudaMalloc d_hidden");
    checkCuda(cudaMalloc(&d_output, sizeof(float) * batch_size * config_.output_dim), "cudaMalloc d_output");

    current_buffer_batch_size_ = batch_size;
}

void MLP::copy_weights_to_device() {
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

void MLP::load_weights_from_file(const std::string& path) {
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

void MLP::forward(const float* input_host, float* output_host, int batch_size) {
    allocate_device_buffers(batch_size);

    checkCuda(cudaMemcpy(d_input, input_host,
                         sizeof(float) * batch_size * config_.input_dim,
                         cudaMemcpyHostToDevice),
              "memcpy input_host -> d_input");

    {
        int threads = 256;
        int elements = batch_size * config_.hidden_dim;
        int blocks = (elements + threads - 1) / threads;

        feedforward_layer_kernel_optimized<<<blocks, threads, threads * sizeof(float)>>>(
            d_input, d_W1, d_b1, d_hidden,
            batch_size, config_.input_dim, config_.hidden_dim
        );

        checkCuda(cudaGetLastError(), "feedforward_layer_kernel_optimized layer1");
    }

    gpu_relu(d_hidden, batch_size * config_.hidden_dim);

    {
        int threads = 256;
        int elements = batch_size * config_.output_dim;
        int blocks = (elements + threads - 1) / threads;

        feedforward_layer_kernel_optimized<<<blocks, threads, threads * sizeof(float)>>>(
            d_hidden, d_W2, d_b2, d_output,
            batch_size, config_.hidden_dim, config_.output_dim
        );

        checkCuda(cudaGetLastError(), "feedforward_layer_kernel_optimized layer2");
    }

    checkCuda(cudaMemcpy(output_host, d_output,
                         sizeof(float) * batch_size * config_.output_dim,
                         cudaMemcpyDeviceToHost),
              "memcpy d_output -> output_host");
}


