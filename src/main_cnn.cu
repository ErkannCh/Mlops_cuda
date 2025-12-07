#include <cuda_runtime.h>

#include <filesystem>
#include <iostream>
#include <vector>

#include "cnn.hpp"
#include "cuda_kernels.cuh"
#include "utils.hpp"

int main() {
    CNNConfig cfg;
    cfg.N = 64;
    cfg.C_in = 1;
    cfg.H = 28;
    cfg.W = 28;
    cfg.C_out_conv = 8;
    cfg.K = 3;
    cfg.fc_out = 10;

    std::string weights_path = "weights/cnn_weights.txt";
    if (!std::filesystem::exists("weights")) {
        std::filesystem::create_directory("weights");
    }

    if (!std::filesystem::exists(weights_path)) {
        std::cout << "Création d'un fichier de poids CNN d'exemple..." << std::endl;
        bool ok = create_example_cnn_weights_file(weights_path, cfg.N, cfg.C_in, cfg.H, cfg.W, cfg.C_out_conv, cfg.K, cfg.fc_out);
        if (!ok) {
            return EXIT_FAILURE;
        }
    }

    SimpleCNN cnn(cfg);
    cnn.load_weights_from_file(weights_path);

    std::vector<float> input(cfg.N * cfg.C_in * cfg.H * cfg.W);
    std::vector<float> output(cfg.N * cfg.fc_out);

    for (auto &v : input) v = 1.0f;

    int warmup = 10;
    int iters = 100;

    for (int i = 0; i < warmup; ++i) {
        cnn.forward(input.data(), output.data());
    }
    checkCuda(cudaDeviceSynchronize(), "sync warmup");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        cnn.forward(input.data(), output.data());
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / iters;

    std::cout << "Temps moyen d'inférence CNN (batch_size=" << cfg.N << "): " << avg_ms << " ms\n";

    std::cout << "Premiers outputs: ";
    for (int i = 0; i < std::min(5, cfg.fc_out); ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
