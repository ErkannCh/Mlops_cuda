#include <cuda_runtime.h>

#include <filesystem>
#include <iostream>
#include <vector>

#include "cuda_kernels.cuh"
#include "rnn.hpp"

int main() {
    RNNConfig cfg;
    cfg.input_dim = 128;
    cfg.hidden_dim = 256;
    cfg.seq_len = 32;

    int batch_size = 64;

    std::string weights_path = "weights/rnn_weights.txt";
    if (!std::filesystem::exists("weights")) {
        std::filesystem::create_directory("weights");
    }

    if (!std::filesystem::exists(weights_path)) {
        std::cerr << "Fichier de poids RNN manquant: " << weights_path << std::endl;
        std::cerr << "Crée-le au format: input_dim hidden_dim seq_len + W_xh + W_hh + b_h" << std::endl;
        return EXIT_FAILURE;
    }

    SimpleRNN rnn(cfg);
    rnn.load_weights_from_file(weights_path);

    std::vector<float> input(cfg.seq_len * batch_size * cfg.input_dim);
    std::vector<float> output(batch_size * cfg.hidden_dim);

    for (auto &v : input) v = 1.0f;

    int warmup = 10;
    int iters = 100;

    for (int i = 0; i < warmup; ++i) {
        rnn.forward(input.data(), output.data(), batch_size);
    }
    checkCuda(cudaDeviceSynchronize(), "sync warmup");

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "eventCreate start");
    checkCuda(cudaEventCreate(&stop), "eventCreate stop");

    checkCuda(cudaEventRecord(start), "eventRecord start");
    for (int i = 0; i < iters; ++i) {
        rnn.forward(input.data(), output.data(), batch_size);
    }
    checkCuda(cudaEventRecord(stop), "eventRecord stop");
    checkCuda(cudaEventSynchronize(stop), "eventSync stop");

    float ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms, start, stop), "eventElapsedTime");
    float avg_ms = ms / iters;

    std::cout << "Temps moyen d'inférence RNN (batch_size=" << batch_size << ", seq_len=" << cfg.seq_len << "): " << avg_ms << " ms\n";

    std::cout << "Premiers outputs (h_T[0]): ";
    for (int i = 0; i < std::min(5, cfg.hidden_dim); ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
