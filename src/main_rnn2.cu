#include <cuda_runtime.h>
#include <filesystem>
#include <iostream>
#include <vector>
#include <algorithm>

#include "cuda_kernels.cuh"
#include "rnnv2.hpp"
#include "utils.hpp"

void sync_and_check(cudaEvent_t event) {
    checkCuda(cudaEventRecord(event), "eventRecord failed");
    checkCuda(cudaEventSynchronize(event), "eventSync failed");
}


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
    
    rnn.forward(input.data(), output.data(), batch_size);
    checkCuda(cudaDeviceSynchronize(), "sync initial setup");
    

    // ----------------------------------------------------------------------
    // TEST 1 : Performance Totale (Calcul + Copies H->D et D->H)
    // Mesure le temps complet, y compris les transferts PCIe.
    // ----------------------------------------------------------------------
    std::cout << "\n--- TEST 1 : Performance Totale (Calcul + Copies) ---\n";

    cudaEvent_t start1, stop1;
    checkCuda(cudaEventCreate(&start1), "eventCreate start1");
    checkCuda(cudaEventCreate(&stop1), "eventCreate stop1");

    for (int i = 0; i < warmup; ++i) {
        rnn.forward(input.data(), output.data(), batch_size);
    }
    checkCuda(cudaDeviceSynchronize(), "sync warmup 1");

    checkCuda(cudaEventRecord(start1), "eventRecord start1");
    for (int i = 0; i < iters; ++i) {
        rnn.forward(input.data(), output.data(), batch_size);
    }
    sync_and_check(stop1);

    float ms1 = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms1, start1, stop1), "eventElapsedTime 1");
    float avg_ms1 = ms1 / iters;

    std::cout << "Temps moyen d'inférence RNN TOTAL (batch=" << batch_size << ", seq=" << cfg.seq_len << "): " << avg_ms1 << " ms\n";
    
    // Affichage des outputs pour validation
    std::cout << "Premiers outputs (h_T[0]): ";
    for (int i = 0; i < std::min(5, cfg.hidden_dim); ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    // ----------------------------------------------------------------------
    // TEST 2 : Performance du Calcul Pur (GPU-Only)
    // Isole la boucle temporelle (64x Sgemm + Kernel Fusionné) SANS copie H<->D.
    // ----------------------------------------------------------------------
    std::cout << "\n--- TEST 2 : Performance Calcul Pur (GPU-Only) ---\n";

    cudaEvent_t start2, stop2;
    checkCuda(cudaEventCreate(&start2), "eventCreate start2");
    checkCuda(cudaEventCreate(&stop2), "eventCreate stop2");
    
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        rnn.forward_gpu_only(batch_size);
    }
    checkCuda(cudaDeviceSynchronize(), "sync warmup 2");

    // Mesure
    checkCuda(cudaEventRecord(start2), "eventRecord start2");
    for (int i = 0; i < iters; ++i) {
        rnn.forward_gpu_only(batch_size);
    }
    sync_and_check(stop2);

    float ms2 = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms2, start2, stop2), "eventElapsedTime 2");
    float avg_ms2 = ms2 / iters;

    std::cout << "Temps moyen d'inférence RNN GPU-ONLY (batch=" << batch_size << ", seq=" << cfg.seq_len << "): " << avg_ms2 << " ms\n";

    // ----------------------------------------------------------------------
    // Analyse
    // ----------------------------------------------------------------------
    float copy_overhead = avg_ms1 - avg_ms2;
    std::cout << "\n--- ANALYSE DE LA PERFORMANCE ---\n";
    std::cout << "Temps total (Test 1) : " << avg_ms1 << " ms\n";
    std::cout << "Temps calcul seul (Test 2) : " << avg_ms2 << " ms\n";
    std::cout << "Latence des copies H<->D estimée : " << copy_overhead << " ms\n";
    std::cout << "\nObjectif PyTorch : 1.99 ms\n";
    
    return 0;
}