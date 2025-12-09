#include <cuda_runtime.h>

#include <fstream>
#include <iostream>

#include "cuda_kernels.cuh"
#include "rnn.hpp"

SimpleRNN::SimpleRNN(const RNNConfig& cfg) : cfg_(cfg) {
    h_W_xh.resize(cfg_.hidden_dim * cfg_.input_dim);
    h_W_hh.resize(cfg_.hidden_dim * cfg_.hidden_dim);
    h_b_h.resize(cfg_.hidden_dim);
}

SimpleRNN::~SimpleRNN() {
    if (d_W_xh) cudaFree(d_W_xh);
    if (d_W_hh) cudaFree(d_W_hh);
    if (d_b_h) cudaFree(d_b_h);

    if (d_input_seq) cudaFree(d_input_seq);
    if (d_h_prev) cudaFree(d_h_prev);
    if (d_h_t) cudaFree(d_h_t);
    if (d_lin_x) cudaFree(d_lin_x);
    if (d_lin_h) cudaFree(d_lin_h);
    if (d_zero_bias) cudaFree(d_zero_bias);
}

void SimpleRNN::allocate_device_buffers(int batch_size) {
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

void SimpleRNN::copy_weights_to_device() {
    if (!d_W_xh) {
        checkCuda(cudaMalloc(&d_W_xh, sizeof(float) * h_W_xh.size()), "malloc d_W_xh");
        checkCuda(cudaMalloc(&d_W_hh, sizeof(float) * h_W_hh.size()), "malloc d_W_hh");
        checkCuda(cudaMalloc(&d_b_h, sizeof(float) * h_b_h.size()), "malloc d_b_h");

        checkCuda(cudaMalloc(&d_zero_bias, sizeof(float) * h_b_h.size()), "malloc d_zero_bias");
        checkCuda(cudaMemset(d_zero_bias, 0, sizeof(float) * h_b_h.size()), "memset d_zero_bias");
    }

    checkCuda(cudaMemcpy(d_W_xh, h_W_xh.data(), sizeof(float) * h_W_xh.size(), cudaMemcpyHostToDevice), "memcpy W_xh");
    checkCuda(cudaMemcpy(d_W_hh, h_W_hh.data(), sizeof(float) * h_W_hh.size(), cudaMemcpyHostToDevice), "memcpy W_hh");
    checkCuda(cudaMemcpy(d_b_h, h_b_h.data(), sizeof(float) * h_b_h.size(), cudaMemcpyHostToDevice), "memcpy b_h");
}

void SimpleRNN::load_weights_from_file(const std::string& path) {
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

void SimpleRNN::forward(const float* input_host, float* output_host, int batch_size) {
    allocate_device_buffers(batch_size);

    size_t seq_size = static_cast<size_t>(cfg_.seq_len) * batch_size * cfg_.input_dim;
    checkCuda(cudaMemcpy(d_input_seq, input_host, sizeof(float) * seq_size, cudaMemcpyHostToDevice), "memcpy input_seq");

    size_t hid_size = static_cast<size_t>(batch_size) * cfg_.hidden_dim;
    checkCuda(cudaMemset(d_h_prev, 0, sizeof(float) * hid_size), "memset h_prev");

    int threads = 256;
    size_t sharedMem = threads * sizeof(float);

    for (int t = 0; t < cfg_.seq_len; ++t) {
        float* d_x_t = d_input_seq + static_cast<size_t>(t) * batch_size * cfg_.input_dim;

        // input -> hidden
        dim3 grid_x(cfg_.hidden_dim, batch_size);
        dim3 threads_x(threads);
        feedforward_layer_kernel_optimized<<<grid_x, threads_x, sharedMem>>>(
            d_x_t, d_W_xh, d_b_h, d_lin_x,
            batch_size, cfg_.input_dim, cfg_.hidden_dim
        );
        checkCuda(cudaGetLastError(), "feedforward_layer_kernel_optimized x->h");

        // hidden -> hidden
        feedforward_layer_kernel_optimized<<<grid_x, threads_x, sharedMem>>>(
            d_h_prev, d_W_hh, d_zero_bias, d_lin_h,
            batch_size, cfg_.hidden_dim, cfg_.hidden_dim
        );
        checkCuda(cudaGetLastError(), "feedforward_layer_kernel_optimized h->h");

        // sum lin_x + lin_h -> h_t
        gpu_matrix_add(d_lin_x, d_lin_h, d_h_t, batch_size, cfg_.hidden_dim);

        // activation tanh
        gpu_tanh(d_h_t, batch_size * cfg_.hidden_dim);

        std::swap(d_h_prev, d_h_t);
    }

    checkCuda(cudaMemcpy(output_host, d_h_prev, sizeof(float) * hid_size, cudaMemcpyDeviceToHost), "memcpy h_T -> output_host");
}

