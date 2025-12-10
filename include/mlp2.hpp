#pragma once

#include <string>
#include <vector>
#include <cublas_v2.h>

struct MLPConfig2 {
    int input_dim;
    int hidden_dim;
    int output_dim;
};

class MLP2 {
   public:
    MLP2(const MLPConfig2 &cfg);
    ~MLP2();

    void load_weights_from_file(const std::string &path);

    void forward(const float* input_host, float* output_host, int batch_size);

    int input_dim() const { return config_.input_dim; }
    int hidden_dim() const { return config_.hidden_dim; }
    int output_dim() const { return config_.output_dim; }

   private:
    MLPConfig2 config_;

    // Handle pour la librairie cuBLAS (essentiel pour les GEMM optimisées)
    cublasHandle_t cublas_handle_ = nullptr; // <-- Ajout du handle cuBLAS

    std::vector<float> h_W1, h_b1;
    std::vector<float> h_W2, h_b2;

    float *d_W1 = nullptr, *d_b1 = nullptr;
    float *d_W2 = nullptr, *d_b2 = nullptr;

    float *d_input = nullptr;
    float *d_hidden = nullptr;
    float *d_output = nullptr;

    int current_buffer_batch_size_ = 0;

    void allocate_device_buffers(int batch_size);
    void copy_weights_to_device();
};