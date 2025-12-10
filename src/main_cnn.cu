#include <cuda_runtime.h>
#include <filesystem>
#include <iostream>
#include <vector>
#include <iomanip>

#include "cnn.hpp"
#include "cuda_kernels.cuh"
#include "utils.hpp"

int main() {
    CNNConfig cfg;
    cfg.N = 64;
    cfg.C_in = 1;
    cfg.H = 56;
    cfg.W = 56;
    cfg.C_out_conv = 16;
    cfg.K = 6;
    cfg.fc_out = 10;

    std::string weights_path = "weights/cnn_weights.txt";
    if (!std::filesystem::exists("weights")) std::filesystem::create_directory("weights");

    if (!std::filesystem::exists(weights_path)) {
        std::cout << "Création d'un fichier de poids CNN d'exemple..." << std::endl;
        if (!create_example_cnn_weights_file(weights_path, cfg.N, cfg.C_in, cfg.H, cfg.W, cfg.C_out_conv, cfg.K, cfg.fc_out))
            return EXIT_FAILURE;
    }

    SimpleCNN cnn(cfg);
    cnn.load_weights_from_file(weights_path);

    std::vector<float> input(cfg.N * cfg.C_in * cfg.H * cfg.W, 1.0f);
    std::vector<float> output(cfg.N * cfg.fc_out);

    int warmup = 10;
    int iters = 100;

    std::cout << "=== Benchmark CNN ===\n";
    std::cout << "Batch size: " << cfg.N << ", Input: " << cfg.C_in << "x" << cfg.H << "x" << cfg.W
              << ", Conv filters: " << cfg.C_out_conv << ", Kernel: " << cfg.K 
              << ", FC output: " << cfg.fc_out << "\n";

    // Modes à tester
    std::vector<ConvMode> modes = {ConvMode::NAIVE, ConvMode::TILED, ConvMode::WARP};
    std::vector<std::string> mode_names = {"NAIVE", "TILED", "WARP"};

    for (size_t m = 0; m < modes.size(); ++m) {
        // Warmup
        for (int i = 0; i < warmup; ++i) {
            cnn.forward(input.data(), output.data(), modes[m]);
        }
        checkCuda(cudaDeviceSynchronize(), "sync warmup");

        // Benchmark total (avec copies H<->D)
        cudaEvent_t start_total, stop_total;
        cudaEventCreate(&start_total);
        cudaEventCreate(&stop_total);

        cudaEventRecord(start_total);
        for (int i = 0; i < iters; ++i) {
            cnn.forward(input.data(), output.data(), modes[m]);
        }
        cudaEventRecord(stop_total);
        cudaEventSynchronize(stop_total);

        float ms_total = 0.0f;
        cudaEventElapsedTime(&ms_total, start_total, stop_total);
        float avg_total = ms_total / iters;

        std::cout << std::setw(6) << mode_names[m] 
                  << " : Temps moyen inference TOTAL = " 
                  << std::fixed << std::setprecision(3) << avg_total << " ms\n";

        // Benchmark GPU-only (input déjà sur device)
        cudaEvent_t start_gpu, stop_gpu;
        cudaEventCreate(&start_gpu);
        cudaEventCreate(&stop_gpu);

        // Warmup GPU-only
        for (int i = 0; i < warmup; ++i) {
            cnn.forward_gpu_only(modes[m]); // GPU-only
        }
        checkCuda(cudaDeviceSynchronize(), "sync warmup GPU");

        cudaEventRecord(start_gpu);
        for (int i = 0; i < iters; ++i) {
            cnn.forward_gpu_only(modes[m]);
        }
        cudaEventRecord(stop_gpu);
        cudaEventSynchronize(stop_gpu);

        float ms_gpu = 0.0f;
        cudaEventElapsedTime(&ms_gpu, start_gpu, stop_gpu);
        float avg_gpu = ms_gpu / iters;

        std::cout << std::setw(6) << mode_names[m] 
                  << " : Temps GPU-only = " 
                  << std::fixed << std::setprecision(3) << avg_gpu << " ms\n";

        // Affiche les 5 premiers outputs
        std::cout << "Premiers outputs: ";
        for (int i = 0; i < std::min(5, cfg.fc_out); ++i) {
            std::cout << output[i] << " ";
        }
        std::cout << "\n\n";
    }

    return 0;
}
