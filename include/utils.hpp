#pragma once

#include <string>
#include <vector>

// Charge un vecteur de float depuis un fichier texte
// On lit simplement N floats séparés par des espaces / retours à la ligne.
bool load_floats_from_file(const std::string& path, std::vector<float>& data);

// Création d'un fichier d'exemple de poids pour un MLP
// Format proposé pour weights/mlp_weights.txt :
//  input_dim hidden_dim output_dim
//  <W1 (hidden_dim x input_dim)>
//  <b1 (hidden_dim)>
//  <W2 (output_dim x hidden_dim)>
//  <b2 (output_dim)>
bool create_example_weights_file(const std::string& path,
                                 int input_dim,
                                 int hidden_dim,
                                 int output_dim);

bool create_example_cnn_weights_file(const std::string& path,
                                     int N,
                                     int C_in,
                                     int H,
                                     int W,
                                     int C_out_conv,
                                     int K,
                                     int fc_out);