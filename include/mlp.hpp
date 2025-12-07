#pragma once

#include <vector>
#include <string>

struct MLPConfig {
    int input_dim;
    int hidden_dim;
    int output_dim;
};

// MLP simple à 2 couches: input -> hidden (ReLU) -> output (linéaire)
class MLP {
public:
    MLP(const MLPConfig& cfg);
    ~MLP();

    // Charge les poids depuis un fichier texte
    // Format décrit dans utils.hpp
    void load_weights_from_file(const std::string& path);

    // Forward sur GPU : input_host de taille (batch_size x input_dim)
    // output_host de taille (batch_size x output_dim)
    void forward(const float* input_host, float* output_host, int batch_size);

    int input_dim() const { return config_.input_dim; }
    int hidden_dim() const { return config_.hidden_dim; }
    int output_dim() const { return config_.output_dim; }

private:
    MLPConfig config_;

    // Poids sur host
    std::vector<float> h_W1, h_b1;
    std::vector<float> h_W2, h_b2;

    // Poids sur device
    float *d_W1 = nullptr, *d_b1 = nullptr;
    float *d_W2 = nullptr, *d_b2 = nullptr;

    // Buffers intermediaires sur device
    float *d_input = nullptr;
    float *d_hidden = nullptr;
    float *d_output = nullptr;

    int current_buffer_batch_size_ = 0;

    void allocate_device_buffers(int batch_size);
    void copy_weights_to_device();
};
