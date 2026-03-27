#pragma once

#define TILE_LEN 16

void matmul_add_tiling_ext(
    float* A, float* B,
    unsigned int m, // output rows
    unsigned int n, // output cols
    unsigned int k, // intermediate col/row len
    unsigned int batch_size,
    float* C, float* D
);

__global__ void batch_matmul_tiled_add(
    float* A, float* B,
    unsigned int m, 
    unsigned int n, 
    unsigned int k, 
    unsigned int batch_size,
    float* C, float* D
);