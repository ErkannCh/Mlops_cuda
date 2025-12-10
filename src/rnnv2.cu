#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
#include <cublas_v2.h>

#include "cuda_kernels.cuh"
#include "rnnv2.hpp"

SimpleRNNv2::SimpleRNNv2(const RNNv2Config& cfg) : cfg_(cfg) {
    h_W_xh.resize(cfg_.hidden_dim * cfg_.input_dim);
    h_W_hh.resize(cfg_.hidden_dim * cfg_.hidden_dim);
    h_b_h.resize(cfg_.hidden_dim);
    
    checkCublas(cublasCreate(&cublas_handle_), "cublasCreate");
    checkCuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
    checkCublas(cublasSetStream(cublas_handle_, stream_), "cublasSetStream");
}

SimpleRNNv2::~SimpleRNNv2() {
    if (d_W_xh) cudaFree(d_W_xh);
    if (d_W_hh) cudaFree(d_W_hh);
    if (d_b_h) cudaFree(d_b_h);

    if (d_input_seq) cudaFree(d_input_seq);
    if (d_h_prev) cudaFree(d_h_prev);
    if (d_h_t) cudaFree(d_h_t);
    if (d_lin_x) cudaFree(d_lin_x);
    if (d_lin_h) cudaFree(d_lin_h);
    if (d_zero_bias) cudaFree(d_zero_bias);

    if (cublas_handle_) cublasDestroy(cublas_handle_);
    if (stream_) cudaStreamDestroy(stream_);
}

void SimpleRNNv2::allocate_device_buffers(int batch_size) {
    if (batch_size <= current_batch_size_) return;

    if (d_input_seq) cudaFree(d_input_seq);
    if (d_h_prev) cudaFree(d_h_prev);
    if (d_h_t) cudaFree(d_h_t);
    if (d_lin_x) cudaFree(d_lin_x);
    if (d_lin_h) cudaFree(d_lin_h);

    size_t seq_size = static_cast<size_t>(cfg_.seq_len) * static_cast<size_t>(batch_size) * static_cast<size_t>(cfg_.input_dim);

    size_t hid_size = static_cast<size_t>(batch_size) * static_cast<size_t>(cfg_.hidden_dim);

    checkCuda(cudaMalloc(&d_input_seq, sizeof(float) * seq_size), "malloc d_input_seq");
    checkCuda(cudaMalloc(&d_h_prev, sizeof(float) * hid_size), "malloc d_h_prev");
    checkCuda(cudaMalloc(&d_h_t, sizeof(float) * hid_size), "malloc d_h_t");
    checkCuda(cudaMalloc(&d_lin_x, sizeof(float) * hid_size), "malloc d_lin_x");
    checkCuda(cudaMalloc(&d_lin_h, sizeof(float) * hid_size), "malloc d_lin_h");

    current_batch_size_ = batch_size;
}

void SimpleRNNv2::copy_weights_to_device() {
    if (!d_W_xh) {
        checkCuda(cudaMalloc(&d_W_xh, sizeof(float) * h_W_xh.size()), "malloc d_W_xh");
        checkCuda(cudaMalloc(&d_W_hh, sizeof(float) * h_W_hh.size()), "malloc d_W_hh");
        checkCuda(cudaMalloc(&d_b_h, sizeof(float) * h_b_h.size()), "malloc d_b_h");
    }

    checkCuda(cudaMemcpy(d_W_xh, h_W_xh.data(), sizeof(float) * h_W_xh.size(), cudaMemcpyHostToDevice), "memcpy W_xh");
    checkCuda(cudaMemcpy(d_W_hh, h_W_hh.data(), sizeof(float) * h_W_hh.size(), cudaMemcpyHostToDevice), "memcpy W_hh");
    checkCuda(cudaMemcpy(d_b_h, h_b_h.data(), sizeof(float) * h_b_h.size(), cudaMemcpyHostToDevice), "memcpy b_h");
}

void SimpleRNNv2::load_weights_from_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Impossible d'ouvrir les poids RNN: " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int in_dim, hid_dim, seq_len;
    f >> in_dim >> hid_dim >> seq_len;

    if (in_dim != cfg_.input_dim || hid_dim != cfg_.hidden_dim || seq_len != cfg_.seq_len) {
        std::cerr << "Config RNN incompatible avec les poids." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    for (auto& v : h_W_xh) f >> v;
    for (auto& v : h_W_hh) f >> v;
    for (auto& v : h_b_h) f >> v;

    copy_weights_to_device();
}

void SimpleRNNv2::forward(const float* input_host, float* output_host, int batch_size) {
    allocate_device_buffers(batch_size);

    size_t seq_size = static_cast<size_t>(cfg_.seq_len) * batch_size * cfg_.input_dim;
    size_t hid_size = static_cast<size_t>(batch_size) * cfg_.hidden_dim;
    
    checkCuda(cudaMemcpyAsync(d_input_seq, input_host, sizeof(float) * seq_size, 
                              cudaMemcpyHostToDevice, stream_), "memcpy input_seq");

    checkCuda(cudaMemsetAsync(d_h_prev, 0, sizeof(float) * hid_size, stream_), "memset h_prev");

    const float alpha = 1.0f;
    const float beta_zero = 0.0f;

    for (int t = 0; t < cfg_.seq_len; ++t) {
        float* d_x_t = d_input_seq + static_cast<size_t>(t) * batch_size * cfg_.input_dim;

        checkCublas(cublasSgemm(cublas_handle_,
                            CUBLAS_OP_T, CUBLAS_OP_N, 
                            cfg_.hidden_dim, batch_size, cfg_.input_dim,
                            &alpha,
                            d_W_xh, cfg_.input_dim, 
                            d_x_t, cfg_.input_dim,   
                            &beta_zero,
                            d_lin_x, cfg_.hidden_dim), "CUBLAS X->H");

        checkCublas(cublasSgemm(cublas_handle_,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            cfg_.hidden_dim, batch_size, cfg_.hidden_dim,
                            &alpha,
                            d_W_hh, cfg_.hidden_dim,
                            d_h_prev, cfg_.hidden_dim,
                            &beta_zero,
                            d_lin_h, cfg_.hidden_dim), "CUBLAS H->H");

        gpu_fused_add_bias_tanh(d_lin_x, d_lin_h, d_b_h, d_h_t, batch_size, cfg_.hidden_dim, stream_);

        std::swap(d_h_prev, d_h_t);
    }

    checkCuda(cudaMemcpyAsync(output_host, d_h_prev, sizeof(float) * hid_size, 
                            cudaMemcpyDeviceToHost, stream_), "memcpy h_T -> output_host");

    cudaStreamSynchronize(stream_);
}

