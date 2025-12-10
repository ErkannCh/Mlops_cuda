#pragma once

#include <string>
#include <vector>

bool load_floats_from_file(const std::string& path, std::vector<float>& data);

bool create_example_weights_file(const std::string& path, int input_dim, int hidden_dim, int output_dim);

bool create_example_rnn_weights_file(const std::string& path, int input_dim, int hidden_dim, int seq_len);

bool create_example_cnn_weights_file(const std::string& path, int N, int C_in, int H, int W, int C_out_conv, int K, int fc_out);