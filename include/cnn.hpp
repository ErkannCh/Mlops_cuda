#pragma once

#include <vector>
#include <string>

struct CNNConfig {
    int N;          // batch size pour l'inférence
    int C_in;
    int H;
    int W;

    int C_out_conv; // nb de filtres de la conv
    int K;          // taille du kernel (K x K)

    int fc_out;     // taille de la couche fully-connected de sortie
};

// CNN simple: Conv -> ReLU -> Flatten -> Linear
class SimpleCNN {
public:
    SimpleCNN(const CNNConfig& cfg);
    ~SimpleCNN();

    // Charge les poids depuis un fichier texte
    void load_weights_from_file(const std::string& path);

    // Forward depuis le host: input_host (N x C_in x H x W), output_host (N x fc_out)
    void forward(const float* input_host, float* output_host);

    // Optionnel : forward device -> device si tu veux benchmark sans memcpy
    void forward_device(const float* d_input, float* d_output);

private:
    CNNConfig cfg_;

    // Poids host
    std::vector<float> h_conv_w;  // (C_out_conv x C_in x K x K)
    std::vector<float> h_conv_b;  // (C_out_conv)
    std::vector<float> h_fc_w;    // (fc_out x (C_out_conv*H_out*W_out))
    std::vector<float> h_fc_b;    // (fc_out)

    // Poids device
    float *d_conv_w = nullptr;
    float *d_conv_b = nullptr;
    float *d_fc_w   = nullptr;
    float *d_fc_b   = nullptr;

    // Buffers device
    float *d_input   = nullptr; // N x C_in x H x W
    float *d_conv_out = nullptr; // N x C_out_conv x H_out x W_out
    float *d_fc_in    = nullptr; // N x (C_out_conv*H_out*W_out)
    float *d_output   = nullptr; // N x fc_out

    int H_out_;
    int W_out_;

    void allocate_device_buffers();
    void copy_weights_to_device();
};
