#include <cuda_runtime.h>
#include <filesystem>
#include <iostream>
#include <vector>
#include <iomanip>
#include <sstream>
#include <tuple>
#include <algorithm>
#include <fstream>     
#include <cstdio>      
#include <cstdarg>     

#include "cnn.hpp"
#include "cuda_kernels.cuh"
#include "utils.hpp"

const std::string OUTPUT_FILENAME_CNN = "cnn_benchmark_results.txt";

struct CNNArchConfig {
    int N;             
    int C_in;          
    int H;            
    int W;             
    int C_out_conv;    
    int K;             
    int fc_out;        
};

void print_and_log_line_cnn(std::ofstream& log_file, const char* format, ...) {
    
    char buffer[512];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);


    std::cout << buffer;


    if (log_file.is_open()) {
        log_file << buffer;
    }
}


void destroy_events(cudaEvent_t start, cudaEvent_t stop) {
    checkCuda(cudaEventDestroy(start), "eventDestroy start");
    checkCuda(cudaEventDestroy(stop), "eventDestroy stop");
}


std::string generate_weights_path(const CNNArchConfig& cfg) {
    std::ostringstream oss;
    oss << "weights/cnn_"
        << "N" << cfg.N << "_C" << cfg.C_in << "x" << cfg.C_out_conv 
        << "_H" << cfg.H << "_K" << cfg.K << "_FC" << cfg.fc_out
        << ".txt";
    return oss.str();
}


std::tuple<float, float> benchmark_cnn_cuda(const CNNArchConfig& cfg, ConvMode mode, std::vector<float>& output, int iters = 100) {

    std::string weights_path = generate_weights_path(cfg);

    if (!std::filesystem::exists("weights")) std::filesystem::create_directory("weights");

    if (!std::filesystem::exists(weights_path)) {
        if (!create_example_cnn_weights_file(weights_path, cfg.N, cfg.C_in, cfg.H, cfg.W, cfg.C_out_conv, cfg.K, cfg.fc_out)) {
            return {-1.0f, -1.0f};
        }
    }

    CNNConfig cnn_cfg{cfg.N, cfg.C_in, cfg.H, cfg.W, cfg.C_out_conv, cfg.K, cfg.fc_out};
    SimpleCNN cnn(cnn_cfg);
    cnn.load_weights_from_file(weights_path);

    output.resize(cfg.N * cfg.fc_out);
    std::vector<float> input(cfg.N * cfg.C_in * cfg.H * cfg.W, 1.0f);

    int warmup = 10;
    float avg_total = -1.0f;
    float avg_gpu = -1.0f;
    
    cnn.forward(input.data(), output.data(), mode); 
    checkCuda(cudaDeviceSynchronize(), "sync initial setup");

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "eventCreate start");
    checkCuda(cudaEventCreate(&stop), "eventCreate stop");

    for (int i = 0; i < warmup; ++i) { cnn.forward(input.data(), output.data(), mode); }
    checkCuda(cudaDeviceSynchronize(), "sync warmup 1");

    checkCuda(cudaEventRecord(start), "eventRecord start 1");
    for (int i = 0; i < iters; ++i) { cnn.forward(input.data(), output.data(), mode); }
    checkCuda(cudaEventRecord(stop), "eventRecord stop 1");
    checkCuda(cudaEventSynchronize(stop), "eventSync stop 1");

    float ms_total = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms_total, start, stop), "eventElapsedTime 1");
    avg_total = ms_total / iters;

    for (int i = 0; i < warmup; ++i) { cnn.forward_gpu_only(mode); }
    checkCuda(cudaDeviceSynchronize(), "sync warmup 2");

    checkCuda(cudaEventRecord(start), "eventRecord start 2");
    for (int i = 0; i < iters; ++i) { cnn.forward_gpu_only(mode); }
    checkCuda(cudaEventRecord(stop), "eventRecord stop 2");
    checkCuda(cudaEventSynchronize(stop), "eventSync stop 2");

    float ms_gpu = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms_gpu, start, stop), "eventElapsedTime 2");
    avg_gpu = ms_gpu / iters;
    
    destroy_events(start, stop);

    return {avg_total, avg_gpu};
}

int main() {
    
    std::ofstream log_file(OUTPUT_FILENAME_CNN);
    if (!log_file.is_open()) {
        std::cerr << "ATTENTION: Impossible d'ouvrir le fichier de log " << OUTPUT_FILENAME_CNN << ". Les résultats ne seront affichés que dans la console." << std::endl;
    }

    std::vector<int> N_vals = {32, 64, 128};
    std::vector<int> C_in_vals = {3}; // Fixé à 3
    std::vector<int> H_W_vals = {32}; // H=W
    std::vector<int> C_out_conv_vals = {16};
    std::vector<int> K_vals = {3};
    std::vector<int> FC_OUT_vals = {10};

    std::vector<CNNArchConfig> archs;
    
    for (int N : N_vals) {
        for (int C_in : C_in_vals) {
            for (int HW : H_W_vals) {
                for (int C_out_conv : C_out_conv_vals) {
                    for (int K : K_vals) {
                        for (int fc_out : FC_OUT_vals) {
                            archs.push_back({N, C_in, HW, HW, C_out_conv, K, fc_out});
                        }
                    }
                }
            }
        }
    }

    std::vector<ConvMode> modes = {ConvMode::NAIVE, ConvMode::TILED};
    std::vector<std::string> mode_names = {"NAIVE", "TILED"};
    
    print_and_log_line_cnn(log_file, "Démarrage du benchmarking SimpleCNN sur CUDA avec %zu configurations (Exhaustif)...\n", archs.size());

    const char* separator = "----------------------------------------------------------------------------------------------------------------------------\n";
    const char* header = "| Configuration             | Mode  | Temps TOTAL (ms) | Temps GPU-ONLY (ms) | Latence Copies (ms) |\n";
    
    print_and_log_line_cnn(log_file, "\n%s", separator);
    print_and_log_line_cnn(log_file, "%s", header);
    print_and_log_line_cnn(log_file, "%s", separator);

    std::vector<float> output_validation; 
    std::ostringstream preview_oss;

    for (const auto &cfg : archs) {
        std::ostringstream cfg_label;
        cfg_label << "N=" << cfg.N << ", Cin=" << cfg.C_in << ", HxW=" << cfg.H << "x" << cfg.W 
                  << ", Cout=" << cfg.C_out_conv << ", K=" << cfg.K << ", FC=" << cfg.fc_out;
        
        for (size_t m = 0; m < modes.size(); ++m) {
            
            auto [avg_ms_total, avg_ms_gpu_only] = benchmark_cnn_cuda(cfg, modes[m], output_validation);
            float copy_overhead = avg_ms_total - avg_ms_gpu_only;

            if (avg_ms_total < 0.0f) {
                print_and_log_line_cnn(log_file, 
                    "| %-25s | %-5s | %16s | %19s | %19s |\n", 
                    cfg_label.str().c_str(), mode_names[m].c_str(), "ERREUR", "ERREUR", "ERREUR");
                continue;
            }
            
            print_and_log_line_cnn(log_file, 
                "| %-25s | %-5s | %16.4f | %19.4f | %19.4f |\n",
                cfg_label.str().c_str(), mode_names[m].c_str(), 
                avg_ms_total, avg_ms_gpu_only, copy_overhead);

            if (m == modes.size() - 1) {
                
                preview_oss.str("");
                preview_oss.clear();
                
                preview_oss << "|                           |       | (Aperçu) Premiers outputs (" << mode_names[m] << "): ";
                
                for (int i = 0; i < std::min(5, cfg.fc_out); ++i) {
                    preview_oss << std::fixed << std::setprecision(4) << output_validation[i] << " ";
                }
                preview_oss << "\n";
                
                print_and_log_line_cnn(log_file, "%s", preview_oss.str().c_str());
            }
        }
        print_and_log_line_cnn(log_file, "%s", separator);
    }

    if (log_file.is_open()) {
        log_file.close();
        std::cout << "\nLes résultats complets ont été enregistrés dans " << OUTPUT_FILENAME_CNN << std::endl;
    }

    return 0;
}