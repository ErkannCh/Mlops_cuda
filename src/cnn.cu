#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>

#include "cnn.hpp"
#include "cuda_kernels.cuh"
#include "utils.hpp"

SimpleCNN::SimpleCNN(const CNNConfig& cfg) : cfg_(cfg) {
    H_out_ = cfg_.H - cfg_.K + 1;
    W_out_ = cfg_.W - cfg_.K + 1;

    int conv_w_size = cfg_.C_out_conv * cfg_.C_in * cfg_.K * cfg_.K;
    int conv_b_size = cfg_.C_out_conv;
    int fc_in_dim = cfg_.C_out_conv * H_out_ * W_out_;
    int fc_w_size = cfg_.fc_out * fc_in_dim;
    int fc_b_size = cfg_.fc_out;

    h_conv_w.resize(conv_w_size);
    h_conv_b.resize(conv_b_size);
    h_fc_w.resize(fc_w_size);
    h_fc_b.resize(fc_b_size);
}

SimpleCNN::~SimpleCNN() {
    if(d_conv_w) cudaFree(d_conv_w);
    if(d_conv_b) cudaFree(d_conv_b);
    if(d_fc_w) cudaFree(d_fc_w);
    if(d_fc_b) cudaFree(d_fc_b);
    if(d_input) cudaFree(d_input);
    if(d_conv_out) cudaFree(d_conv_out);
    if(d_fc_in) cudaFree(d_fc_in);
    if(d_output) cudaFree(d_output);
}

void SimpleCNN::allocate_device_buffers() {
    if(!d_input) {
        int N = cfg_.N;
        int fc_in_dim = cfg_.C_out_conv * H_out_ * W_out_;
        checkCuda(cudaMalloc(&d_input, sizeof(float) * N * cfg_.C_in * cfg_.H * cfg_.W), "malloc d_input");
        checkCuda(cudaMalloc(&d_conv_out, sizeof(float) * N * cfg_.C_out_conv * H_out_ * W_out_), "malloc d_conv_out");
        checkCuda(cudaMalloc(&d_fc_in, sizeof(float) * N * fc_in_dim), "malloc d_fc_in");
        checkCuda(cudaMalloc(&d_output, sizeof(float) * N * cfg_.fc_out), "malloc d_output");
    }
}

void SimpleCNN::copy_weights_to_device() {
    int conv_w_size = h_conv_w.size();
    int conv_b_size = h_conv_b.size();
    int fc_w_size = h_fc_w.size();
    int fc_b_size = h_fc_b.size();

    if(!d_conv_w) {
        checkCuda(cudaMalloc(&d_conv_w, sizeof(float) * conv_w_size), "malloc d_conv_w");
        checkCuda(cudaMalloc(&d_conv_b, sizeof(float) * conv_b_size), "malloc d_conv_b");
        checkCuda(cudaMalloc(&d_fc_w, sizeof(float) * fc_w_size), "malloc d_fc_w");
        checkCuda(cudaMalloc(&d_fc_b, sizeof(float) * fc_b_size), "malloc d_fc_b");
    }

    checkCuda(cudaMemcpy(d_conv_w, h_conv_w.data(), sizeof(float) * conv_w_size, cudaMemcpyHostToDevice), "memcpy conv_w");
    checkCuda(cudaMemcpy(d_conv_b, h_conv_b.data(), sizeof(float) * conv_b_size, cudaMemcpyHostToDevice), "memcpy conv_b");
    checkCuda(cudaMemcpy(d_fc_w, h_fc_w.data(), sizeof(float) * fc_w_size, cudaMemcpyHostToDevice), "memcpy fc_w");
    checkCuda(cudaMemcpy(d_fc_b, h_fc_b.data(), sizeof(float) * fc_b_size, cudaMemcpyHostToDevice), "memcpy fc_b");
}

void SimpleCNN::load_weights_from_file(const std::string& path) {
    std::ifstream f(path);
    if(!f.is_open()) {
        std::cerr << "Impossible d'ouvrir les poids CNN: " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int N, C_in, H, W, C_out_conv, K, fc_out;
    f >> N >> C_in >> H >> W >> C_out_conv >> K >> fc_out;

    if(N != cfg_.N || C_in != cfg_.C_in || H != cfg_.H || W != cfg_.W || C_out_conv != cfg_.C_out_conv || K != cfg_.K || fc_out != cfg_.fc_out) {
        std::cerr << "Config CNN incompatible avec les poids." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    for(auto &v : h_conv_w) f >> v;
    for(auto &v : h_conv_b) f >> v;
    for(auto &v : h_fc_w) f >> v;
    for(auto &v : h_fc_b) f >> v;

    copy_weights_to_device();
    allocate_device_buffers();
}

// Flatten NCHW -> FC input
static __global__ void flatten_nchw_kernel(const float* input, float* output, int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if(idx >= total) return;
    output[idx] = input[idx];
}

void SimpleCNN::forward_device(const float* d_input_external, float* d_output_external, ConvMode mode) {
    int N = cfg_.N;
    int C_in = cfg_.C_in;
    int H = cfg_.H;
    int W = cfg_.W;
    int C_out_conv = cfg_.C_out_conv;
    int K = cfg_.K;
    int fc_in_dim = C_out_conv * H_out_ * W_out_;

    // Convolution selon le mode
    switch(mode) {
        case ConvMode::NAIVE:
            gpu_conv2d_relu_naive(d_input_external, d_conv_w, d_conv_b, d_conv_out,
                                  N, C_in, H, W, C_out_conv, K, H_out_, W_out_);
            break;
        case ConvMode::TILED:
            gpu_conv2d_tiled(d_input_external, d_conv_w, d_conv_b, d_conv_out,
                             N, C_in, H, W, C_out_conv, K, H_out_, W_out_);
            break;
        case ConvMode::WARP:
            gpu_conv2d_warp(d_input_external, d_conv_w, d_conv_b, d_conv_out,
                            N, C_in, H, W, C_out_conv, K, H_out_, W_out_);
            break;
    }

    // Flatten conv output
    int total_conv = N * C_out_conv * H_out_ * W_out_;
    int blockSize = 256;
    int gridSize = (total_conv + blockSize - 1) / blockSize;
    flatten_nchw_kernel<<<gridSize, blockSize>>>(d_conv_out, d_fc_in, N, C_out_conv, H_out_, W_out_);
    checkCuda(cudaGetLastError(), "flatten kernel");

    // Feedforward layer optimisé
    dim3 threads(256);
    dim3 blocks(cfg_.fc_out, N);
    feedforward_layer_kernel_optimized<<<blocks, threads, threads.x * sizeof(float)>>>(d_fc_in, d_fc_w, d_fc_b, d_output_external,
                                                                                       N, fc_in_dim, cfg_.fc_out);
    checkCuda(cudaGetLastError(), "feedforward_layer_kernel_optimized launch");
}


void SimpleCNN::forward_gpu_only(ConvMode mode) {
    allocate_device_buffers();
    forward_device(d_input, d_output, mode);
}

void SimpleCNN::forward(const float* input_host, float* output_host, ConvMode mode) {
    allocate_device_buffers();

    // Copier l'input sur le device
    checkCuda(cudaMemcpy(d_input, input_host, sizeof(float) * cfg_.N * cfg_.C_in * cfg_.H * cfg_.W,
                         cudaMemcpyHostToDevice), "memcpy input_host->d_input");

    // Choix du mode de convolution
    switch (mode) {
        case ConvMode::NAIVE:
            gpu_conv2d_relu_naive(d_input, d_conv_w, d_conv_b, d_conv_out,
                                  cfg_.N, cfg_.C_in, cfg_.H, cfg_.W,
                                  cfg_.C_out_conv, cfg_.K, H_out_, W_out_);
            break;

        case ConvMode::TILED:
            gpu_conv2d_tiled(d_input, d_conv_w, d_conv_b, d_conv_out,
                             cfg_.N, cfg_.C_in, cfg_.H, cfg_.W,
                             cfg_.C_out_conv, cfg_.K, H_out_, W_out_);
            break;

        case ConvMode::WARP:
            gpu_conv2d_warp(d_input, d_conv_w, d_conv_b, d_conv_out,
                            cfg_.N, cfg_.C_in, cfg_.H, cfg_.W,
                            cfg_.C_out_conv, cfg_.K, H_out_, W_out_);
            break;
    }

    // Flatten pour FC
    int total_conv = cfg_.N * cfg_.C_out_conv * H_out_ * W_out_;
    int blockSize = 256;
    int gridSize = (total_conv + blockSize - 1) / blockSize;
    flatten_nchw_kernel<<<gridSize, blockSize>>>(d_conv_out, d_fc_in, cfg_.N, cfg_.C_out_conv, H_out_, W_out_);
    checkCuda(cudaGetLastError(), "flatten kernel");

    // Feedforward optimisé
    dim3 threads(256);
    dim3 blocks(cfg_.fc_out, cfg_.N);
    size_t shared_mem_bytes = threads.x * sizeof(float);

    feedforward_layer_kernel_optimized<<<blocks, threads, shared_mem_bytes>>>(
        d_fc_in, d_fc_w, d_fc_b, d_output, cfg_.N, cfg_.C_out_conv * H_out_ * W_out_, cfg_.fc_out
    );
    checkCuda(cudaGetLastError(), "feedforward_layer_kernel_optimized launch");

    // Copier le résultat sur l'hôte
    checkCuda(cudaMemcpy(output_host, d_output, sizeof(float) * cfg_.N * cfg_.fc_out,
                         cudaMemcpyDeviceToHost), "memcpy d_output->output_host");
}