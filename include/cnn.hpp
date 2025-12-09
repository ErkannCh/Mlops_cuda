#pragma once

#include <string>
#include <vector>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cudnn.h>

struct CNNConfig {
    int N;
    int C_in;
    int H;
    int W;

    int C_out_conv;
    int K;

    int fc_out;
};

class SimpleCNN {
   public:
    SimpleCNN(const CNNConfig &cfg);
    ~SimpleCNN();

    void load_weights_from_file(const std::string &path);

    void forward(const float *input_host, float *output_host);

    void forward_device(const float *d_input, float *d_output);

   private:
    CNNConfig cfg_;

    std::vector<float> h_conv_w;
    std::vector<float> h_conv_b;
    std::vector<float> h_fc_w;
    std::vector<float> h_fc_b;

    float *d_conv_w = nullptr;
    float *d_conv_b = nullptr;
    float *d_fc_w = nullptr;
    float *d_fc_b = nullptr;

    float *d_input = nullptr;
    float *d_conv_out = nullptr;
    float *d_fc_in = nullptr;
    float *d_output = nullptr;

    int H_out_;
    int W_out_;

    void allocate_device_buffers();
    void copy_weights_to_device();
};
