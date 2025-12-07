#pragma once

#include <string>
#include <vector>

struct RNNConfig {
    int input_dim;
    int hidden_dim;
    int seq_len;
};

class SimpleRNN {
   public:
    SimpleRNN(const RNNConfig &cfg);
    ~SimpleRNN();

    void load_weights_from_file(const std::string &path);

    void forward(const float *input_host, float *output_host, int batch_size);

   private:
    RNNConfig cfg_;

    std::vector<float> h_W_xh;
    std::vector<float> h_W_hh;
    std::vector<float> h_b_h;

    float *d_W_xh = nullptr;
    float *d_W_hh = nullptr;
    float *d_b_h = nullptr;

    float *d_input_seq = nullptr;
    float *d_h_prev = nullptr;
    float *d_h_t = nullptr;
    float *d_lin_x = nullptr;
    float *d_lin_h = nullptr;
    float *d_zero_bias = nullptr;

    int current_batch_size_ = 0;

    void allocate_device_buffers(int batch_size);
    void copy_weights_to_device();
};
