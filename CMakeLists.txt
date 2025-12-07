cmake_minimum_required(VERSION 3.18)
project(cuda_nn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Warnings généraux
if (MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
include_directories(${CMAKE_SOURCE_DIR}/include)

# ---- Exécutable MLP (niveau 1) ----
add_executable(cuda_mlp
    src/main.cu        # <= MLP seulement
    src/cuda_kernels.cu
    src/mlp.cu
    src/utils.cpp
)

# ---- Exécutable CNN (niveau 2) ----
add_executable(cuda_cnn
    src/main_cnn.cu    # <= CNN seulement
    src/cuda_kernels.cu
    src/cnn.cu
    src/utils.cpp
)

add_executable(cuda_rnn
    src/main_rnn.cu
    src/rnn.cu
    src/cuda_kernels.cu
    src/utils.cpp
)