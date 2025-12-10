cmake_minimum_required(VERSION 3.18)
project(cuda_nn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if (MSVC)
add_compile_options(/W4)
else()
add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Ignore le warning "style of line directive is a GCC extension"
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wno-cpp")
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-w>)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-w>)


set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
include_directories(${CMAKE_SOURCE_DIR}/include)

# Définition des exécutables
# ----------------------------

add_executable(cuda_mlp
    src/main_mlp.cu
    src/cuda_kernels.cu
    src/mlp.cu
    src/utils.cpp
)
# Ajout de la liaison vers CUBLAS
target_link_libraries(cuda_mlp PRIVATE cublas)

add_executable(cuda_cnn
    src/main_cnn.cu
    src/cuda_kernels.cu
    src/cnn.cu
    src/utils.cpp
)
# Ajout de la liaison vers CUBLAS
target_link_libraries(cuda_cnn PRIVATE cublas)

add_executable(cuda_rnn
    src/main_rnn.cu
    src/rnn.cu
    src/cuda_kernels.cu
    src/utils.cpp
)
# Ajout de la liaison vers CUBLAS
target_link_libraries(cuda_rnn PRIVATE cublas)

add_executable(cuda_rnnv2
    src/main_rnn2.cu
    src/rnnv2.cu
    src/cuda_kernels.cu
    src/utils.cpp
)
# Ajout de la liaison vers CUBLAS
target_link_libraries(cuda_rnnv2 PRIVATE cublas)