#pragma once

extern "C" 
{
    __declspec(dllexport) void l2norm_forward_ext(
        float* original, float* original_pow2, unsigned int z0, unsigned y0, unsigned x0,
        float* power_sum, float* normalized, bool zeroed
    );
    __declspec(dllexport) void l2norm_backward_ext(
        float* original_grads, unsigned int z0, unsigned y0, unsigned x0,
        float* power_sum, float* original_inputs, float* original_outputs, float* result_grads,
        bool zeroed
    );
}

__global__ void calc_pow2sum_kernel_reduction(float* summed,
    float* original, unsigned int z0, unsigned y0, unsigned x0,
    bool zeroed, int shared_memory_len);
__global__ void l2norm_backward_kernel(
    float* result_grads, float* original_grads, 
    float* inputs, float* outputs, float* power_sum, int z0, int y0, int x0,
    bool zeroed
);
__global__ void divide_inputs_kernel(
    float* broadcasted, float* original, float* power_sum, 
    unsigned int z0, unsigned y0, unsigned x0, int axis
);