#pragma once

void matmul_add_reduce_ext(
    float* a, float* b,
    unsigned int m, // output rows
    unsigned int n, // output cols
    unsigned int k, // intermediate col/row len
    unsigned int batch_size,
    float* c, float* d
);

__global__ void batch_matmul_reduce(
    float* A, float* B,
    unsigned int m, unsigned int n, unsigned int k, unsigned int batch_size,
    float* C
);

void mean_axis_reduction(
    float* original_array, unsigned int z0, unsigned int y0, unsigned int x0,
    float* temp_array, unsigned int z1, unsigned int y1, unsigned int x1,
    float* output, unsigned int z2, unsigned int y2, unsigned int x2, 
    unsigned int first_pass, unsigned int second_pass,
    int axis
);

__global__ void sum_axis_reduce(
    // assume dst and src have the same shape
    float* dst_array, unsigned int dst_z, unsigned int dst_y, unsigned int dst_x,
    float* original_array, unsigned int array_z, unsigned int array_y, unsigned int array_x,
    int axis
);