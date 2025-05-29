#pragma once

#define LAYER_NORM_BLOCK_SIZE 256

extern "C"
{
    __declspec(dllexport) void layer_norm_ext(
        float* output_ptr,
        float* normalized_ptr, 
        float* original_ptr, 
        float* mean_ptr, 
        //unsigned int first_pass, unsigned int second_pass,
        //float* mean_temp, unsigned int temp_z, unsigned int temp_y, unsigned int temp_x,
        float* var_ptr, // e.g. both (1, y0, x0) if performing along z axis
        float* alphas, float* betas,
        unsigned int z0, unsigned int y0, unsigned x0, 
        int axis
    );

    __declspec(dllexport) void layer_norm_back_ext(
        float* return_grads,
        float* original_grads,
        float* original_inputs_ptr,
        float* normalized_ptr, 
        float* mean_ptr, float* var_ptr, // e.g. both (1, y0, x0) if performing along z axis
        float* mean_grads, float* var_grads, // (z0, y0, x0)
        float* alpha_ptr, float* beta_ptr, float* alpha_grad, float* beta_grad,
        int z0, int y0, int x0,
        int axis
    );
}

__global__ void standarize_tensor_kernel(
    float* output_ptr,
    float* normalized_ptr, 
    float* original_ptr, 
    float* mean_ptr, float* var_ptr, // e.g. both (1, y0, x0) if performing along z axis
    float* alphas, float* betas,
    unsigned int z0, unsigned int y0, unsigned x0, 
    int axis
);

__global__ void z_scoring(
    float* output_ptr,
    float* normalized_ptr, 
    float* original_ptr, 
    float* mean_ptr, float* var_ptr, // e.g. both (1, y0, x0) if performing along z axis
    float* alphas, float* betas,
    unsigned int z0, unsigned int y0, unsigned x0, 
    int axis
);

__global__ void layer_norm_back_kernel(
    float* return_grads,
    float* original_grads,
    float* original_inputs_ptr,
    float* normalized_ptr, 
    float* mean_ptr, float* var_ptr, // e.g. both (1, y0, x0) if performing along z axis
    float* mean_grads, float* var_grads, // (z0, y0, x0)
    float* alpha_ptr, float* beta_ptr, float* alpha_grad, float* beta_grad,
    int z0, int y0, int x0,
    int axis
);

__device__ unsigned int get_sliced_idx(
    int z, int y, int x,
    int z0, int y0, int x0, int axis,
    int idx_on_axis
);

__global__ void mean_axis_kernel_reduction(float* mean,
    float* original, unsigned int z0, unsigned y0, unsigned x0,
    int axis, bool zeroed, int shared_memory_len);

__global__ void var_axis_kernel_reduction(
    float* var,
    float* mean,
    float* original, unsigned int z0, unsigned y0, unsigned x0,
    int axis, bool zeroed, int shared_memory_len);