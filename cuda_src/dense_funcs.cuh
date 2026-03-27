#pragma once

// defines the total shape of the tensor
typedef struct Tuple4D { unsigned int a, z, y, x; } Tuple4D;

extern "C"
{
    __declspec(dllexport) void matmul_bias_ext(
        float* mat0, unsigned int z0, unsigned int y0, unsigned int x0,
        float* mat1, unsigned int z1, unsigned int y1, unsigned int x1,
        float* result_mat, float* mat2
    );

    __declspec(dllexport) void matmul_bias_tiled_ext(
        float* mat0, unsigned int z0, unsigned int y0, unsigned int x0,
        float* mat1, unsigned int z1, unsigned int y1, unsigned int x1,
        float* result_mat, float* mat2, bool use_bias, bool zero_output
    );

    __declspec(dllexport) void matmul_bias_back_ext(
        float* input_grads, unsigned int z0, unsigned int y0, unsigned int x0, // (3, 2, 5, 1)
        float* weight_grads, unsigned int z1, unsigned int y1, unsigned int x1, // (3, 1, 5, 4)
        float* bias_grads, unsigned int z2, unsigned int y2, unsigned int x2, // (3, 2, 5, 4) -> (3, 2, 4)
        float* original_grads, // (3, 2, 4)
        float* input_tensor,
        float* weight_tensor,
        bool use_bias,
        bool zero_input_grad,
        bool zero_weight_grad
    );
    __declspec(dllexport) void transpose_2d_ext(float* arr_t, float* arr, unsigned int z0, unsigned int y0, unsigned int x0);
}

__global__ void matmul_bias_kernel(
    float* mat0, unsigned int z0, unsigned int y0, unsigned int x0, // (3, 2, 5)
    float* mat1, unsigned int z1, unsigned int y1, unsigned int x1, // (3, 5, 4), actually (3, 1, 5, 4) for compatibility with
    //float* broadcast_buffer,// broadcast buffer shape = (3, 2, 5, 4)
    float* result_mat, unsigned int z2, unsigned int y2, unsigned int x2, // (3, 2, 4)
    float* biases
);
__global__ void matmul_bias_back_kernel(
    float* input_grads, unsigned int z0, unsigned int y0, unsigned int x0, // (3, 2, 5, 1)
    float* weight_grads, unsigned int z1, unsigned int y1, unsigned int x1, // (3, 1, 5, 4)
    float* bias_grads, unsigned int z2, unsigned int y2, unsigned int x2, // (3, 2, 5, 4) -> (3, 2, 4)
    float* original_grads, // (3, 2, 4)
    float* input_tensor, // (3, 2, 5, 1)
    float* weight_tensor // (3, 1, 5, 4)
);

__global__ void transpose_2d_kernel(
    float* arr_t, float* arr, unsigned int z0, unsigned int y0, unsigned int x0
);

__device__ void transpose_2d(
    float* arr_t, float* arr, 
    unsigned int z, unsigned int y, unsigned int x,
    unsigned int z0, unsigned int y0, unsigned int x0
);
__device__ void matmul(
    float* mat0, unsigned int z0, unsigned int y0, unsigned int x0, bool transpose_mat0, // (3, 2, 5)
    float* mat1, unsigned int z1, unsigned int y1, unsigned int x1, bool transpose_mat1, // (3, 5, 4), actually (3, 1, 5, 4) for compatibility with
    //float* broadcast_buffer,// broadcast buffer shape = (3, 2, 5, 4)
    unsigned int val,
    float* result_mat, unsigned int z2, unsigned int y2, unsigned int x2, // (3, 2, 4)
    unsigned int z, unsigned int y, unsigned int x,
    bool add // 3d index of result matrix, within result dimensions
);
__device__ void add_biases(
    float* result_mat, float* biases, 
    unsigned int z, unsigned int y, unsigned int x,
    unsigned int z2, unsigned int y2, unsigned int x2
);
__device__ Tuple4D idx1_to_idx4(unsigned int flattened_idx, Tuple4D shape);
__device__ unsigned int idx4_to_idx1(
    unsigned int a, unsigned int z, unsigned int y, unsigned int x, 
    Tuple4D shape
);

/*
__global__ void matmul_add_back_kernel(
    float* input_grads, unsigned int z0, unsigned int y0, unsigned int x0, // (3, 2, 5, 1)
    float* weight_grads, unsigned int z1, unsigned int y1, unsigned int x1, // (3, 1, 5, 4)
    float* bias_grads, unsigned int z2, unsigned int y2, unsigned int x2, // (3, 2, 5, 4) -> (3, 2, 4)
    float* original_grads, // (3, 2, 4)
    float* input_tensor, // (3, 2, 5, 1)
    float* weight_tensor // (3, 1, 5, 4)
);
*/
__global__ void calc_input_grads_kernel(
    float* input_grads, unsigned int z0, unsigned int y0, unsigned int x0, // (3, 2, 5, 1)
    float* weight_tensor, unsigned int z1, unsigned int y1, unsigned int x1, // (3, 1, 5, 4)
    float* original_grads, unsigned int z2, unsigned int y2, unsigned int x2, // (3, 2, 4)
    bool add
);
__global__ void calc_weight_grads_kernel(
    float* weight_grads, unsigned int z1, unsigned int y1, unsigned int x1, // (3, 1, 5, 4)
    float* bias_grads, unsigned int z2, unsigned int y2, unsigned int x2, // (3, 2, 5, 4) -> (3, 2, 4)
    float* original_grads, // (3, 2, 4), same dimensions as bias grads
    float* input_tensor, unsigned int z0, unsigned int y0, unsigned int x0,// (3, 2, 5, 1)
    bool add
);
__global__ void calc_bias_grads_kernel(
    float* bias_grads, unsigned int z2, unsigned int y2, unsigned int x2, // (3, 2, 5, 4) -> (3, 2, 4)
    float* original_grads
);
/**/