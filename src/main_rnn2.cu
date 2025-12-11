#include <cuda_runtime.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <tuple> 
#include <fstream>   // Ajout pour l'écriture de fichiers
#include <cstdio>    // Nécessaire pour l'utilisation de printf
#include <cstdarg>

#include "cuda_kernels.cuh"
#include "rnn2.hpp" 
#include "utils.hpp"

// Nom du fichier de sortie
const std::string OUTPUT_FILENAME_V2 = "rnn2_benchmark_results.txt";

// Structure pour stocker les configurations d'architecture RNNv2
struct RNNv2ArchConfig {
    int input_dim;
    int hidden_dim;
    int seq_len;
    int batch_size;
};

// --- Fonctions utilitaires ---

/**
 * @brief Écrit une ligne formatée dans la console et dans le fichier de sortie.
 * Cette version utilise printf pour un alignement précis.
 */
void print_and_log_line_rnnv2(std::ofstream& log_file, const char* format, ...) {
    // 1. Écrire dans une chaîne temporaire (pour le fichier)
    char buffer[512];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);

    // 2. Écrire dans le stdout
    std::cout << buffer;

    // 3. Écrire dans le fichier
    if (log_file.is_open()) {
        log_file << buffer;
    }
}

// Fonction utilitaire pour synchroniser et vérifier l'événement
void sync_and_check(cudaEvent_t event) {
    checkCuda(cudaEventRecord(event), "eventRecord failed");
    checkCuda(cudaEventSynchronize(event), "eventSync failed");
}

/**
 * @brief Mesure le temps d'inférence moyen du SimpleRNNv2 (Total et GPU-Only) sur CUDA.
 * @return std::tuple<float, float> {temps_total_ms, temps_gpu_only_ms}
 */
std::tuple<float, float> benchmark_rnnv2_cuda(int input_dim,
                                             int hidden_dim,
                                             int seq_len,
                                             int batch_size,
                                             int iters = 100) {

    // [La logique de création/chargement des poids et des événements CUDA reste inchangée]
    std::ostringstream oss;
    oss << "weights/rnn2_"
        << "in" << input_dim << "_hid" << hidden_dim << "_seq" << seq_len
        << ".txt";
    std::string weights_path = oss.str();

    if (!std::filesystem::exists("weights")) {
        std::filesystem::create_directory("weights");
    }

    if (!std::filesystem::exists(weights_path)) {
        if (!create_example_rnn_weights_file(
                weights_path, input_dim, hidden_dim, seq_len)) {
            return {-1.0f, -1.0f};
        }
    }

    RNNv2Config cfg{input_dim, hidden_dim, seq_len};
    SimpleRNNv2 rnn(cfg);
    rnn.load_weights_from_file(weights_path);

    std::vector<float> input(seq_len * batch_size * input_dim);
    std::vector<float> output(batch_size * hidden_dim);
    for (auto &v : input) v = 1.0f;

    int warmup = 10;
    
    float avg_ms_total = -1.0f;
    float avg_ms_gpu_only = -1.0f;
    
    rnn.forward(input.data(), output.data(), batch_size);
    checkCuda(cudaDeviceSynchronize(), "sync initial setup");
    
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "eventCreate start");
    checkCuda(cudaEventCreate(&stop), "eventCreate stop");

    // ----------------------------------------------------------------------
    // TEST 1 : Performance Totale (Calcul + Copies H->D et D->H)
    // ----------------------------------------------------------------------
    for (int i = 0; i < warmup; ++i) { rnn.forward(input.data(), output.data(), batch_size); }
    checkCuda(cudaDeviceSynchronize(), "sync warmup 1");

    checkCuda(cudaEventRecord(start), "eventRecord start 1");
    for (int i = 0; i < iters; ++i) { rnn.forward(input.data(), output.data(), batch_size); }
    sync_and_check(stop);

    float ms_total = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms_total, start, stop), "eventElapsedTime 1");
    avg_ms_total = ms_total / iters;


    // ----------------------------------------------------------------------
    // TEST 2 : Performance du Calcul Pur (GPU-Only)
    // ----------------------------------------------------------------------
    for (int i = 0; i < warmup; ++i) { rnn.forward_gpu_only(batch_size); }
    checkCuda(cudaDeviceSynchronize(), "sync warmup 2");

    checkCuda(cudaEventRecord(start), "eventRecord start 2");
    for (int i = 0; i < iters; ++i) { rnn.forward_gpu_only(batch_size); }
    sync_and_check(stop);

    float ms_gpu_only = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms_gpu_only, start, stop), "eventElapsedTime 2");
    avg_ms_gpu_only = ms_gpu_only / iters;
    
    checkCuda(cudaEventDestroy(start), "eventDestroy start");
    checkCuda(cudaEventDestroy(stop), "eventDestroy stop");
    
    return {avg_ms_total, avg_ms_gpu_only};
}

int main() {
    std::ofstream log_file(OUTPUT_FILENAME_V2);
    if (!log_file.is_open()) {
        std::cerr << "ATTENTION: Impossible d'ouvrir le fichier de log " << OUTPUT_FILENAME_V2 << ". Les résultats ne seront affichés que dans la console." << std::endl;
    }

    std::vector<RNNv2ArchConfig> archs = {
        {256, 128, 32, 32}, {256, 128, 32, 64}, {256, 128, 32, 128}, {256, 128, 32, 256}, {256, 128, 32, 512}, {256, 128, 32, 1024},
        
        {256, 256, 32, 32}, {256, 256, 32, 64}, {256, 256, 32, 128}, {256, 256, 32, 256}, {256, 256, 32, 512}, {256, 256, 32, 1024},
        
        {256, 512, 32, 32}, {256, 512, 32, 64}, {256, 512, 32, 128}, {256, 512, 32, 256}, {256, 512, 32, 512}, {256, 512, 32, 1024},
        
        {256, 1024, 32, 32}, {256, 1024, 32, 64}, {256, 1024, 32, 128}, {256, 1024, 32, 256}, {256, 1024, 32, 512}, {256, 1024, 32, 1024},

        {256, 128, 64, 32}, {256, 128, 64, 64}, {256, 128, 64, 128}, {256, 128, 64, 256}, {256, 128, 64, 512}, {256, 128, 64, 1024},
        
        {256, 256, 64, 32}, {256, 256, 64, 64}, {256, 256, 64, 128}, {256, 256, 64, 256}, {256, 256, 64, 512}, {256, 256, 64, 1024},
        
        {256, 512, 64, 32}, {256, 512, 64, 64}, {256, 512, 64, 128}, {256, 512, 64, 256}, {256, 512, 64, 512}, {256, 512, 64, 1024},
       
        {256, 1024, 64, 32}, {256, 1024, 64, 64}, {256, 1024, 64, 128}, {256, 1024, 64, 256}, {256, 1024, 64, 512}, {256, 1024, 64, 1024},

        {256, 128, 128, 32}, {256, 128, 128, 64}, {256, 128, 128, 128}, {256, 128, 128, 256}, {256, 128, 128, 512}, {256, 128, 128, 1024},
       
        {256, 256, 128, 32}, {256, 256, 128, 64}, {256, 256, 128, 128}, {256, 256, 128, 256}, {256, 256, 128, 512}, {256, 256, 128, 1024},
     
        {256, 512, 128, 32}, {256, 512, 128, 64}, {256, 512, 128, 128}, {256, 512, 128, 256}, {256, 512, 128, 512}, {256, 512, 128, 1024},
        
        {256, 1024, 128, 32}, {256, 1024, 128, 64}, {256, 1024, 128, 128}, {256, 1024, 128, 256}, {256, 1024, 128, 512}, {256, 1024, 128, 1024},

        {256, 128, 256, 32}, {256, 128, 256, 64}, {256, 128, 256, 128}, {256, 128, 256, 256}, {256, 128, 256, 512}, {256, 128, 256, 1024},

        {256, 256, 256, 32}, {256, 256, 256, 64}, {256, 256, 256, 128}, {256, 256, 256, 256}, {256, 256, 256, 512}, {256, 256, 256, 1024},
 
        {256, 512, 256, 32}, {256, 512, 256, 64}, {256, 512, 256, 128}, {256, 512, 256, 256}, {256, 512, 256, 512}, {256, 512, 256, 1024},
    
        {256, 1024, 256, 32}, {256, 1024, 256, 64}, {256, 1024, 256, 128}, {256, 1024, 256, 256}, {256, 1024, 256, 512}, {256, 1024, 256, 1024},

        
        {256, 128, 512, 32}, {256, 128, 512, 64}, {256, 128, 512, 128}, {256, 128, 512, 256}, {256, 128, 512, 512}, {256, 128, 512, 1024},
     
        {256, 256, 512, 32}, {256, 256, 512, 64}, {256, 256, 512, 128}, {256, 256, 512, 256}, {256, 256, 512, 512}, {256, 256, 512, 1024},
    
        {256, 512, 512, 32}, {256, 512, 512, 64}, {256, 512, 512, 128}, {256, 512, 512, 256}, {256, 512, 512, 512}, {256, 512, 512, 1024},
      
        {256, 1024, 512, 32}, {256, 1024, 512, 64}, {256, 1024, 512, 128}, {256, 1024, 512, 256}, {256, 1024, 512, 512}, {256, 1024, 512, 1024},
    };
    
    print_and_log_line_rnnv2(log_file, "Démarrage du benchmarking RNNv2 sur CUDA avec multiples configurations...\n");

    const char* header = 
        "-----------------------------------------------------------------------------------------------------\n"
        "| D (Input) | H (Hidden) | T (SeqLen) | N (Batch) | Temps TOTAL (ms) | Temps GPU-ONLY (ms) | Latence Copies (ms) |\n"
        "-----------------------------------------------------------------------------------------------------\n";
    print_and_log_line_rnnv2(log_file, header);

    for (const auto &a : archs) {
        auto [avg_ms_total, avg_ms_gpu_only] = benchmark_rnnv2_cuda(
            a.input_dim, a.hidden_dim, a.seq_len, a.batch_size);

        if (avg_ms_total < 0.0f) {
            print_and_log_line_rnnv2(log_file, 
                "| %9d | %10d | %10d | %9d | \t\tERREUR\t | \t\tERREUR\t\t | \t\tERREUR\t\t |\n",
                a.input_dim, a.hidden_dim, a.seq_len, a.batch_size);
        } else {
            float copy_overhead = avg_ms_total - avg_ms_gpu_only;
            
            print_and_log_line_rnnv2(log_file, 
                "| %9d | %10d | %10d | %9d | %16.4f | %17.4f | %19.4f |\n",
                a.input_dim, a.hidden_dim, a.seq_len, a.batch_size, 
                avg_ms_total, avg_ms_gpu_only, copy_overhead);
        }
    }
    
    print_and_log_line_rnnv2(log_file, "-----------------------------------------------------------------------------------------------------\n");

    const auto &a = archs.back();
    RNNv2Config cfg_last{a.input_dim, a.hidden_dim, a.seq_len};
    SimpleRNNv2 rnn_last(cfg_last);
    
    std::ostringstream oss_last;
    oss_last << "weights/rnn2_"
        << "in" << a.input_dim << "_hid" << a.hidden_dim << "_seq" << a.seq_len
        << ".txt";
    rnn_last.load_weights_from_file(oss_last.str());

    std::vector<float> input(a.seq_len * a.batch_size * a.input_dim);
    std::vector<float> output(a.batch_size * a.hidden_dim);
    for (auto &v : input) v = 1.0f;
    
    rnn_last.forward(input.data(), output.data(), a.batch_size);

    std::ostringstream preview_oss;
    preview_oss << "\n[Aperçu pour la dernière configuration (D=" << a.input_dim << ", H=" << a.hidden_dim << ", T=" << a.seq_len << ", N=" << a.batch_size << ")]\n";
    preview_oss << "Premiers outputs (h_T[0]): ";
    for (int i = 0; i < std::min(5, a.hidden_dim); ++i) {
        preview_oss << output[i] << " ";
    }
    preview_oss << "\n";
    
    print_and_log_line_rnnv2(log_file, preview_oss.str().c_str());

    if (log_file.is_open()) {
        log_file.close();
        std::cout << "\nLes résultats complets ont été enregistrés dans " << OUTPUT_FILENAME_V2 << std::endl;
    }

    return 0;
}