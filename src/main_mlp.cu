#include <cuda_runtime.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <vector>

#include "cuda_kernels.cuh"
#include "mlp.hpp"
#include "utils.hpp"

struct ArchConfig {
    int input_dim;
    int hidden_dim;
    int output_dim;
    int batch_size;
};

float benchmark_mlp_cuda(int input_dim,
                         int hidden_dim,
                         int output_dim,
                         int batch_size,
                         int iters = 100) {
    // Chemin du fichier de poids spécifique à cette archi
    std::ostringstream oss;
    oss << "weights/mlp_"
        << "in" << input_dim << "_hid" << hidden_dim << "_out" << output_dim
        << ".txt";
    std::string weights_path = oss.str();

    if (!std::filesystem::exists("weights")) {
        std::filesystem::create_directory("weights");
    }

    if (!std::filesystem::exists(weights_path)) {
        if (!create_example_weights_file(
                weights_path, input_dim, hidden_dim, output_dim)) {
            return -1.0f;
        }
    }

    MLPConfig cfg{input_dim, hidden_dim, output_dim};
    MLP mlp(cfg);
    mlp.load_weights_from_file(weights_path);

    std::vector<float> input(batch_size * input_dim);
    std::vector<float> output(batch_size * output_dim);

    for (auto &v : input) v = 1.0f;

    int warmup = 10;

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        mlp.forward(input.data(), output.data(), batch_size);
    }
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize warmup");

    // Timing avec cudaEvent
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "eventCreate start");
    checkCuda(cudaEventCreate(&stop), "eventCreate stop");

    checkCuda(cudaEventRecord(start), "eventRecord start");
    for (int i = 0; i < iters; ++i) {
        mlp.forward(input.data(), output.data(), batch_size);
    }
    checkCuda(cudaEventRecord(stop), "eventRecord stop");
    checkCuda(cudaEventSynchronize(stop), "eventSync stop");

    float ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms, start, stop), "eventElapsedTime");
    float avg_ms = ms / iters;
    return avg_ms;
}

int main() {
    std::vector<ArchConfig> archs = {
        {128,   64, 10,   32},
		{128,   64, 10,  128},
		{128,   64, 10,  512},
		{128,   64, 10, 1024},
		{128,  128, 10,   32},
		{128,  128, 10,  128},
		{128,  128, 10,  512},
		{128,  128, 10, 1024},
		{128,  256, 10,   32},
		{128,  256, 10,  128},
		{128,  256, 10,  512},
		{128,  256, 10, 1024},
		{128,  512, 10,   32},
		{128,  512, 10,  128},
		{128,  512, 10,  512},
		{128,  512, 10, 1024},
		{128, 1024, 10,   32},
		{128, 1024, 10,  128},
		{128, 1024, 10,  512},
		{128, 1024, 10, 1024},
    };
    for (const auto &a : archs) {
        float avg_ms = benchmark_mlp_cuda(
            a.input_dim, a.hidden_dim, a.output_dim, a.batch_size);

        if (avg_ms < 0.0f) {
            std::cerr << "  Erreur pour cette architecture." << std::endl;
            continue;
        }

        std::cout << "  in=" << a.input_dim
                  << ", hid=" << a.hidden_dim
                  << ", out=" << a.output_dim
                  << ", batch=" << a.batch_size
                  << " -> " << avg_ms << " ms" << std::endl;
    }

    return 0;
}
