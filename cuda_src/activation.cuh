#pragma once

extern "C" {
    __declspec(dllexport) void activation3d_cuda_ext(
        float* result, float* mat, int z, int y, int x, int func_id,
        float scale, bool zero_output
    );
    __declspec(dllexport) void activation3d_backward_cuda_ext(
        float* chained_grads, float* inputs, float* original_grad, 
        int z, int y, int x, int func_id,
        float scale, bool zero_input_grad
    );
}

__global__ void activation_kernel(
    float* result, float* mat, int z0, int y0, int x0, int func_id,
    float scale, bool zero_output
);
__global__ void activation_back_kernel(
    float* chained_grads, float* inputs, float* original_grads, 
    int z0, int y0, int x0, int func_id,
    float scale, bool zero_input_grad
);
__device__ float function(float val, int func_id);
__device__ float function_deriv(float val, int func_id);
__device__ float sigmoid(float v);
__device__ float sigmoid_deriv(float v);
__device__ float tanh_deriv(float v);
__device__ float silu(float v);
__device__ float silu_deriv(float v);
__device__ float gelu(float v);
__device__ float gelu_deriv(float v);
__device__ float tanh2(float v);
__device__ float tanh2_deriv(float v);
__device__ float softplus(float v);
__device__ float softplus_deriv(float v);
__device__ float linear(float v);
__device__ float linear_deriv(float v);