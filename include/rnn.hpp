#pragma once

#include <vector>
#include <string>

struct RNNConfig {
    int input_dim;   // dimension d'entrée x_t
    int hidden_dim;  // dimension de l'état caché h_t
    int seq_len;     // longueur de séquence T
};

// RNN simple (tanh) :
// h_0 = 0
// h_t = tanh( x_t W_xh^T + h_{t-1} W_hh^T + b )
//
// On retourne le dernier état caché h_T pour chaque élément du batch.
class SimpleRNN {
public:
    SimpleRNN(const RNNConfig& cfg);
    ~SimpleRNN();

    // Format du fichier de poids (texte) :
    //  input_dim hidden_dim seq_len
    //  W_xh (hidden_dim x input_dim)
    //  W_hh (hidden_dim x hidden_dim)
    //  b_h  (hidden_dim)
    void load_weights_from_file(const std::string& path);

    // input_host : [seq_len, batch_size, input_dim]
    // output_host : [batch_size, hidden_dim]  (dernier h_T)
    void forward(const float* input_host, float* output_host, int batch_size);

private:
    RNNConfig cfg_;

    // Poids host
    std::vector<float> h_W_xh;  // (hidden_dim x input_dim)
    std::vector<float> h_W_hh;  // (hidden_dim x hidden_dim)
    std::vector<float> h_b_h;   // (hidden_dim)

    // Poids device
    float *d_W_xh = nullptr;
    float *d_W_hh = nullptr;
    float *d_b_h  = nullptr;

    // Buffers device
    float *d_input_seq = nullptr; // [seq_len * batch_size * input_dim]
    float *d_h_prev    = nullptr; // [batch_size * hidden_dim]
    float *d_h_t       = nullptr; // [batch_size * hidden_dim]
    float *d_lin_x     = nullptr; // [batch_size * hidden_dim]
    float *d_lin_h     = nullptr; // [batch_size * hidden_dim]
    float *d_zero_bias = nullptr; // [hidden_dim] (vecteur de zéros)

    int current_batch_size_ = 0;

    void allocate_device_buffers(int batch_size);
    void copy_weights_to_device();
};
