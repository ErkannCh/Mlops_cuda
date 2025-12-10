#include "rnnv2.hpp"
#include "cuda_kernels.cuh"
#include "utils.hpp"
#include <fstream>
#include <iostream>
#include <algorithm> 


SimpleRNNv2::SimpleRNNv2(const RNNv2Config &cfg) : cfg_(cfg), current_batch_size_(0) {
    checkCublas(cublasCreate(&cublas_handle_), "cublasCreate failed");
    checkCuda(cudaStreamCreate(&stream_), "cudaStreamCreate failed");
    checkCublas(cublasSetStream(cublas_handle_, stream_), "cublasSetStream failed");
    
    h_W_xh.resize(cfg.hidden_dim * cfg.input_dim);
    h_W_hh.resize(cfg.hidden_dim * cfg.hidden_dim);
    h_b_h.resize(cfg.hidden_dim);
    
    checkCuda(cudaMalloc(&d_W_xh, h_W_xh.size() * sizeof(float)), "cudaMalloc d_W_xh");
    checkCuda(cudaMalloc(&d_W_hh, h_W_hh.size() * sizeof(float)), "cudaMalloc d_W_hh");
    checkCuda(cudaMalloc(&d_b_h, h_b_h.size() * sizeof(float)), "cudaMalloc d_b_h");
}

SimpleRNNv2::~SimpleRNNv2() {
    checkCuda(cudaFree(d_W_xh), "cudaFree d_W_xh");
    checkCuda(cudaFree(d_W_hh), "cudaFree d_W_hh");
    checkCuda(cudaFree(d_b_h), "cudaFree d_b_h");
    
    checkCuda(cudaFree(d_input_seq), "cudaFree d_input_seq");
    checkCuda(cudaFree(d_h_prev), "cudaFree d_h_prev");
    checkCuda(cudaFree(d_h_t), "cudaFree d_h_t");
    checkCuda(cudaFree(d_lin_x), "cudaFree d_lin_x");
    checkCuda(cudaFree(d_lin_h), "cudaFree d_lin_h");
    checkCuda(cudaFree(d_zero_bias), "cudaFree d_zero_bias");

    checkCublas(cublasDestroy(cublas_handle_), "cublasDestroy failed");
    checkCuda(cudaStreamDestroy(stream_), "cudaStreamDestroy failed");
}

void SimpleRNNv2::load_weights_from_file(const std::string &path) {
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

void SimpleRNNv2::copy_weights_to_device() {
    checkCuda(cudaMemcpy(d_W_xh, h_W_xh.data(), h_W_xh.size() * sizeof(float), cudaMemcpyHostToDevice), "copy d_W_xh");
    checkCuda(cudaMemcpy(d_W_hh, h_W_hh.data(), h_W_hh.size() * sizeof(float), cudaMemcpyHostToDevice), "copy d_W_hh");
    checkCuda(cudaMemcpy(d_b_h, h_b_h.data(), h_b_h.size() * sizeof(float), cudaMemcpyHostToDevice), "copy d_b_h");
}

void SimpleRNNv2::allocate_device_buffers(int batch_size) {
    if (batch_size != current_batch_size_) {
        
        if (d_input_seq) checkCuda(cudaFree(d_input_seq), "cudaFree d_input_seq");
        if (d_h_prev) checkCuda(cudaFree(d_h_prev), "cudaFree d_h_prev");
        if (d_h_t) checkCuda(cudaFree(d_h_t), "cudaFree d_h_t");
        if (d_lin_x) checkCuda(cudaFree(d_lin_x), "cudaFree d_lin_x");

        size_t input_size = (size_t)cfg_.seq_len * batch_size * cfg_.input_dim * sizeof(float);
        size_t hidden_size = (size_t)batch_size * cfg_.hidden_dim * sizeof(float);
        size_t pre_act_size = hidden_size; 

        checkCuda(cudaMalloc(&d_input_seq, input_size), "cudaMalloc d_input_seq");
        checkCuda(cudaMalloc(&d_h_prev, hidden_size), "cudaMalloc d_h_prev");
        checkCuda(cudaMalloc(&d_h_t, hidden_size), "cudaMalloc d_h_t");
        checkCuda(cudaMalloc(&d_lin_x, pre_act_size), "cudaMalloc d_lin_x (pre_act buffer)");

        checkCuda(cudaMemset(d_h_prev, 0, hidden_size), "cudaMemset d_h_prev (H0)");

        current_batch_size_ = batch_size;
    }
}

void SimpleRNNv2::forward(const float *input_host, float *output_host, int batch_size) {
    allocate_device_buffers(batch_size);

    size_t hidden_size = (size_t)batch_size * cfg_.hidden_dim * sizeof(float);
    
    checkCuda(cudaMemsetAsync(d_h_prev, 0, hidden_size, stream_), "cudaMemset H0 to zero"); 
    size_t input_size = (size_t)cfg_.seq_len * batch_size * cfg_.input_dim * sizeof(float);
    checkCuda(cudaMemcpyAsync(d_input_seq, input_host, input_size, cudaMemcpyHostToDevice, stream_), "copy input_host to device");

    const int M = cfg_.hidden_dim;
    const int N = batch_size;   
    const int K_X = cfg_.input_dim;
    const int K_H = cfg_.hidden_dim;

    const float alpha = 1.0f;
    const float beta_init = 0.0f; 
    const float beta_add = 1.0f; 
    
    float *h_prev_ptr = d_h_prev;
    float *h_t_ptr = d_h_t;

    for (int t = 0; t < cfg_.seq_len; ++t) {
        const float *d_X_t = d_input_seq + (size_t)t * batch_size * cfg_.input_dim;
        
        checkCublas(cublasSgemm(cublas_handle_,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              M, N, K_X,
                              &alpha,
                              d_W_xh, M,         
                              d_X_t, K_X,         
                              &beta_init,         
                              d_lin_x, M), "Sgemm W_xh");

        checkCublas(cublasSgemm(cublas_handle_,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              M, N, K_H,
                              &alpha,
                              d_W_hh, M,          
                              h_prev_ptr, K_H,     
                              &beta_add,           
                              d_lin_x, M), "Sgemm W_hh"); 
        
        int num_elements = batch_size * cfg_.hidden_dim;
        launch_bias_add_and_tanh_kernel(
            h_t_ptr, d_lin_x, d_b_h, num_elements, cfg_.hidden_dim, stream_
        );
        
        std::swap(h_prev_ptr, h_t_ptr); 
    }

    float *d_final_output = h_prev_ptr;
    
    size_t output_size = (size_t)batch_size * cfg_.hidden_dim * sizeof(float);
    checkCuda(cudaMemcpyAsync(output_host, d_final_output, output_size, cudaMemcpyDeviceToHost, stream_), "copy final output to host");

}

void SimpleRNNv2::forward_gpu_only(int batch_size) {
    allocate_device_buffers(batch_size); 

    size_t hidden_size = (size_t)batch_size * cfg_.hidden_dim * sizeof(float);
    checkCuda(cudaMemsetAsync(d_h_prev, 0, hidden_size, stream_), "cudaMemset H0 to zero"); 
    
    const int M = cfg_.hidden_dim;
    const int N = batch_size;
    const int K_X = cfg_.input_dim;
    const int K_H = cfg_.hidden_dim;

    const float alpha = 1.0f;
    const float beta_init = 0.0f; 
    const float beta_add = 1.0f;  
    
    float *h_prev_ptr = d_h_prev;
    float *h_t_ptr = d_h_t;

    for (int t = 0; t < cfg_.seq_len; ++t) {
        const float *d_X_t = d_input_seq + (size_t)t * batch_size * cfg_.input_dim;
        
        checkCublas(cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K_X, &alpha,
                              d_W_xh, M, d_X_t, K_X, &beta_init, d_lin_x, M), "Sgemm W_xh"); 

        checkCublas(cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K_H, &alpha,
                              d_W_hh, M, h_prev_ptr, K_H, &beta_add, d_lin_x, M), "Sgemm W_hh"); 
        
        int num_elements = batch_size * cfg_.hidden_dim;
        launch_bias_add_and_tanh_kernel(h_t_ptr, d_lin_x, d_b_h, num_elements, cfg_.hidden_dim, stream_);
        
        std::swap(h_prev_ptr, h_t_ptr); 
    }
}