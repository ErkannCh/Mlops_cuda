#include "mlp.hpp"
#include "utils.hpp"
#include "cuda_kernels.cuh"

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <filesystem>
#include <chrono>

int main() {
    // Config du réseau
    int input_dim = 128;
    int hidden_dim = 256;
    int output_dim = 10;
    int batch_size = 1024;

    std::string weights_path = "weights/mlp_weights.txt";
    if (!std::filesystem::exists("weights")) {
        std::filesystem::create_directory("weights");
    }

    if (!std::filesystem::exists(weights_path)) {
        std::cout << "Création d'un fichier de poids d'exemple..." << std::endl;
        if (!create_example_weights_file(weights_path,
                                         input_dim, hidden_dim, output_dim)) {
            return EXIT_FAILURE;
        }
    }

    MLPConfig cfg{input_dim, hidden_dim, output_dim};
    MLP mlp(cfg);

    mlp.load_weights_from_file(weights_path);

    // Données d'entrée aléatoires (sur host)
    std::vector<float> input(batch_size * input_dim);
    std::vector<float> output(batch_size * output_dim);

    for (auto &v : input) v = 1.0f; // ou générer aléatoire

    // Mesure de temps: plusieurs itérations pour lisser
    int warmup = 10;
    int iters = 100;

    // warmup
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
    std::cout << "Temps moyen d'inférence (batch_size=" << batch_size
              << "): " << avg_ms << " ms" << std::endl;

    // Exemple : afficher un bout de la sortie
    std::cout << "Premiers outputs: ";
    for (int i = 0; i < std::min(5, output_dim); ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
