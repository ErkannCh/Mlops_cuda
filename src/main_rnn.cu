#include <cuda_runtime.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <fstream>   
#include <iomanip>   

#include "cuda_kernels.cuh"
#include "rnn.hpp"
#include "utils.hpp"

const std::string OUTPUT_FILENAME = "rnn_benchmark_results.txt";

struct RNNv2ArchConfig {
    int input_dim;  
    int hidden_dim; 
    int seq_len;    
    int batch_size; 
};

void print_and_log_line(std::ofstream& log_file, const std::string& line, bool console_only = false) {
    std::cout << line;
    if (!console_only && log_file.is_open()) {
        log_file << line;
    }
}

float benchmark_rnn_cuda(int input_dim,
                         int hidden_dim,
                         int seq_len,
                         int batch_size,
                         int iters = 100) {

    std::ostringstream oss;
    oss << "weights/rnn_"
        << "in" << input_dim << "_hid" << hidden_dim << "_seq" << seq_len
        << ".txt";
    std::string weights_path = oss.str();

    if (!std::filesystem::exists("weights")) {
        std::filesystem::create_directory("weights");
    }

    if (!std::filesystem::exists(weights_path)) {
        if (!create_example_rnn_weights_file(
                weights_path, input_dim, hidden_dim, seq_len)) {
                    
            std::cerr << "  Erreur de création des poids d'exemple." << std::endl;
            return -1.0f;
        }
    }

    RNNConfig cfg{input_dim, hidden_dim, seq_len};
    SimpleRNN rnn(cfg);
    rnn.load_weights_from_file(weights_path);

    std::vector<float> input(seq_len * batch_size * input_dim);
    std::vector<float> output(batch_size * hidden_dim);
    for (auto &v : input) v = 1.0f;

    int warmup = 10;

    for (int i = 0; i < warmup; ++i) {
        rnn.forward(input.data(), output.data(), batch_size);
    }
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize warmup");

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
    
    checkCuda(cudaEventDestroy(start), "eventDestroy start");
    checkCuda(cudaEventDestroy(stop), "eventDestroy stop");
    
    return avg_ms;
}

int main() {
    std::ofstream log_file(OUTPUT_FILENAME);
    if (!log_file.is_open()) {
        std::cerr << "ATTENTION: Impossible d'ouvrir le fichier de log " << OUTPUT_FILENAME << ". Les résultats ne seront affichés que dans la console." << std::endl;
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
    
    print_and_log_line(log_file, "Démarrage du benchmarking RNN sur CUDA...\n");

    std::string header = 
        "------------------------------------------------------------------------------------------------\n"
        "| D_in | H_hid | T_seq | Batch | Temps moyen (ms) | Remarques                     |\n"
        "------------------------------------------------------------------------------------------------\n";
    print_and_log_line(log_file, header);

    for (const auto &a : archs) {
        float avg_ms = benchmark_rnn_cuda(
            a.input_dim, a.hidden_dim, a.seq_len, a.batch_size);
        
        std::ostringstream line;
        line << std::fixed << std::setprecision(4);

        line << "| " << std::setw(4) << a.input_dim
             << " | " << std::setw(5) << a.hidden_dim
             << " | " << std::setw(5) << a.seq_len
             << " | " << std::setw(5) << a.batch_size;

        if (avg_ms < 0.0f) {
            line << " | " << std::setw(16) << "ECHEC"
                 << " | " << std::left << std::setw(29) << "Erreur lors du benchmark." << " |\n";
        } else {
            line << " | " << std::setw(16) << avg_ms
                 << " | " << std::left << std::setw(29) << "" << " |\n";
        }

        print_and_log_line(log_file, line.str());
    }

    print_and_log_line(log_file, "------------------------------------------------------------------------------------------------\n");

    const auto &a = archs.back();
    RNNConfig cfg{a.input_dim, a.hidden_dim, a.seq_len};
    SimpleRNN rnn(cfg);
    
    std::ostringstream oss_last;
    oss_last << "weights/rnn_"
        << "in" << a.input_dim << "_hid" << a.hidden_dim << "_seq" << a.seq_len
        << ".txt";
    rnn.load_weights_from_file(oss_last.str());

    std::vector<float> input(a.seq_len * a.batch_size * a.input_dim);
    std::vector<float> output(a.batch_size * a.hidden_dim);
    for (auto &v : input) v = 1.0f;
    
    rnn.forward(input.data(), output.data(), a.batch_size);

    std::ostringstream preview;
    preview << "\n" << "Aperçu de la sortie (h_T[0]) pour la dernière config testée:\n"
            << "D=" << a.input_dim << ", H=" << a.hidden_dim
            << ", T=" << a.seq_len << ", N=" << a.batch_size << "\n"
            << "Outputs: ";
    for (int i = 0; i < std::min(5, a.hidden_dim); ++i) {
        preview << std::fixed << std::setprecision(4) << output[i] << " ";
    }
    preview << "\n";
    print_and_log_line(log_file, preview.str(), true);

    if (log_file.is_open()) {
        log_file.close();
        std::cout << "\nLes résultats complets ont été enregistrés dans " << OUTPUT_FILENAME << std::endl;
    }

    return 0;
}