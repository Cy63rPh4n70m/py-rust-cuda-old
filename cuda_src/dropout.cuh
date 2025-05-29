#pragma once
#include <curand_kernel.h>

extern "C" 
{
    __declspec(dllexport) void dropout_forward_ext(
        float* input, float* output, float* mask, void* states,
        unsigned int z0, unsigned int y0, unsigned int x0, float dropout_rate
    );

    __declspec(dllexport) void dropout_backward_ext(
        float* original_grads,
        float* dropout_mask, float* result_grads,
        unsigned int z0, unsigned int y0, unsigned int x0
    );

    __declspec(dllexport) void* init_random_states_ext(
        unsigned int z0, unsigned int y0, unsigned int x0
    );

    __declspec(dllexport) void elementwise_dropout_forward_ext(
        float* input, float* weights, float* output, float* mask, void* states,
        unsigned int z0, unsigned int y0, unsigned int x0, float dropout_rate, unsigned int op,
        int activation_id, float act_scale, bool use_dropout, bool zero_output
    );

    __declspec(dllexport) void elementwise_dropout_backward_ext(
        float* original_grads,
        float* dropout_mask, 
        float* input, float* weights, float* weight_grads,
        float* result_grads, bool use_dropout, int activation_id, float act_scale,
        unsigned int z0, unsigned int y0, unsigned int x0,
        unsigned int op, bool zero_input_grad, bool zero_weight_grad
    );

}

__global__ void init_random_states(
    curandState* states, unsigned long long int seed, 
    unsigned int z0, unsigned int y0, unsigned int x0
);

__device__ void random_dropout(
    curandState* states, float* dropout_mask, float dropout_rate, int idx
);

__global__ void dropout_forward_kernel(
    float* tensor, float* output, float* dropout_mask,
    unsigned int z0, unsigned int y0, unsigned int x0, 
    curandState* states, float dropout_rate
);

__global__ void dropout_backward_kernel(
    float* original_grads,
    float* dropout_mask, float* result_grads,
    unsigned int z0, unsigned int y0, unsigned int x0
);

__global__ void elementwise_dropout_forward_kernel(
    float* tensor, float* weights, float* output, float* dropout_mask,
    unsigned int z0, unsigned int y0, unsigned int x0, 
    curandState* states, float dropout_rate, unsigned int op, int activation_id, float act_scale,
    bool use_dropout, bool zero_output
);

__global__ void elementwise_dropout_backward_kernel(
    float* original_grads,
    float* dropout_mask, 
    float* input, float* weights, float* weight_grads,
    float* result_grads, bool use_dropout, int activation_id, float act_scale,
    unsigned int z0, unsigned int y0, unsigned int x0,
    unsigned int op, bool zero_input_grad, bool zero_weight_grad
);