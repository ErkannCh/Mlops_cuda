#include <cuda_runtime.h>

#include <filesystem>
#include <iostream>
#include <vector>

#include "cuda_kernels.cuh"
#include "rnnv2.hpp"    
#include "utils.hpp"

int main() {
    RNNv2Config cfg;
    cfg.input_dim = 256;
    cfg.hidden_dim = 512;
    cfg.seq_len = 64;

    int batch_size = 1024;

    std::string weights_path = "weights/rnn_weights.txt";
    if (!std::filesystem::exists("weights")) {
        std::filesystem::create_directory("weights");
    }

    if (!std::filesystem::exists(weights_path)) {
        std::cout << "Création d'un fichier de poids RNN d'exemple..." << std::endl;
        if (!create_example_rnn_weights_file(weights_path, cfg.input_dim, cfg.hidden_dim, cfg.seq_len)) {
            return EXIT_FAILURE;
        }
    }

    SimpleRNNv2 rnn(cfg);
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
