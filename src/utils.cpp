#include "utils.hpp"
#include <fstream>
#include <random>
#include <iostream>

bool load_floats_from_file(const std::string& path, std::vector<float>& data) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Impossible d'ouvrir le fichier: " << path << std::endl;
        return false;
    }
    data.clear();
    float x;
    while (f >> x) {
        data.push_back(x);
    }
    return true;
}

bool create_example_weights_file(const std::string& path,
                                 int input_dim,
                                 int hidden_dim,
                                 int output_dim) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "Impossible de créer le fichier: " << path << std::endl;
        return false;
    }

    f << input_dim << " " << hidden_dim << " " << output_dim << "\n";

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    auto write_matrix = [&](int rows, int cols) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                f << dist(rng) << " ";
            }
            f << "\n";
        }
    };

    auto write_vector = [&](int size) {
        for (int i = 0; i < size; ++i) {
            f << dist(rng) << " ";
        }
        f << "\n";
    };

    // W1 (hidden x input)
    write_matrix(hidden_dim, input_dim);
    // b1 (hidden)
    write_vector(hidden_dim);
    // W2 (output x hidden)
    write_matrix(output_dim, hidden_dim);
    // b2 (output)
    write_vector(output_dim);

    return true;
}

bool create_example_cnn_weights_file(const std::string& path,
                                     int N,
                                     int C_in,
                                     int H,
                                     int W,
                                     int C_out_conv,
                                     int K,
                                     int fc_out) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "Impossible de créer le fichier CNN: " << path << std::endl;
        return false;
    }

    // Dimensions de sortie de la conv (stride=1, pad=0)
    int H_out = H - K + 1;
    int W_out = W - K + 1;

    // 1ère ligne : config
    f << N << " " << C_in << " " << H << " " << W << " "
      << C_out_conv << " " << K << " " << fc_out << "\n";

    std::mt19937 rng(123);  // seed fixe pour reproductibilité
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    auto write_matrix = [&](int rows, int cols) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                f << dist(rng) << " ";
            }
            f << "\n";
        }
    };

    auto write_vector = [&](int size) {
        for (int i = 0; i < size; ++i) {
            f << dist(rng) << " ";
        }
        f << "\n";
    };

    // conv_w : (C_out_conv x C_in x K x K)
    // On l'écrit comme C_out_conv "lignes", chacune avec (C_in*K*K) valeurs
    write_matrix(C_out_conv, C_in * K * K);

    // conv_b : (C_out_conv)
    write_vector(C_out_conv);

    // fc_w : (fc_out x (C_out_conv * H_out * W_out))
    int fc_in_dim = C_out_conv * H_out * W_out;
    write_matrix(fc_out, fc_in_dim);

    // fc_b : (fc_out)
    write_vector(fc_out);

    return true;
}
