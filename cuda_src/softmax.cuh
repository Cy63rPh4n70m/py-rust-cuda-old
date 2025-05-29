#pragma once

extern "C" 
{
    __declspec(dllexport) void softmax_forward_ext(
        float* input, float* exp_input, float* exp_sum,
        float* result, float* broadcast_temp, float temperature,
        unsigned int z0, unsigned y0, unsigned x0, bool zero_output
    );

    __declspec(dllexport) void softmax_backward_ext(
        float* original_grads,
        float* exp_input, float* result_grads,
        float* exp_sum, float temperature,
        unsigned int z0, unsigned y0, unsigned x0, bool zero_input_grad
    );

    __declspec(dllexport) void softmax_ce_loss_ext(
        float* pred, float* classes, float* loss_vals, float* grad_ptr, int z0, int y0, int x0
    );

}

__global__ void softmax_backward_kernel(
    float* original_grads,
    float* exp_input, float* result_grads,
    float* exp_sum, float temperature,
    unsigned int z0, unsigned y0, unsigned x0, bool zero_input_grad
);

__global__ void obtain_max_logit(
    int* max_value, float* original, float scale, unsigned int z0, unsigned y0, unsigned x0,
    bool zeroed, int shared_memory_len
);

__global__ void broadcast_division_kernel(
    float* broadcasted, unsigned int z0, unsigned y0, unsigned x0,
    float* original, int axis
);

__global__ void fill_3d_kernel_int(int* arr, int value, int z0, int y0, int x0);
__global__ void int_to_float_kernel(float* dst, int* src, float scale, int z0, int y0, int x0);
__global__ void softmax_ce_loss(float* pred, float* classes, float* loss_vals, float* grad_ptr, int z0, int y0, int x0);