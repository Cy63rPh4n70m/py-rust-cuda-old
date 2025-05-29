#pragma once

extern "C"
{
    __declspec(dllexport) void element_op_3d_ext(float* dst, float* src, int op, int z0, int y0, int x0);
    __declspec(dllexport) void element_op_3d_ret_ext(float* c, float* a, float* b, int op, int z0, int y0, int x0);
    __declspec(dllexport) void scalar_op_3d_inplace_ext(float* dst, float scalar, int op, int z0, int y0, int x0);
    __declspec(dllexport) void zeroes_3d_ext(float* dst, int z0, int y0, int x0);
    __declspec(dllexport) void round_3d_ext(float* arr, int places, int z0, int y0, int x0);
    __declspec(dllexport) void gradient_desc_3d_ext(
        float lr, float l2,
        float* weight_ptr, float* weight_ptr_grad, float* weight_velocity, float* weight_momentum,
        unsigned int z0, unsigned int y0, unsigned int x0,
        float* bias_ptr, float* bias_ptr_grad, float* bias_velocity, float* bias_momentum,
        unsigned int z1, unsigned int y1, unsigned int x1,
        bool only_beta, bool non_neg, float batch_size, int optimizer_type, 
        float alpha, float beta
    );
    __declspec(dllexport) void broadcast_2d_to_3d_ext(
        float* broadcasted, unsigned int z0, unsigned y0, unsigned x0,
        float* original, int axis
    );
    __declspec(dllexport) void sum_axis_ext(
        float* summed,
        float* original, unsigned int z0, unsigned y0, unsigned x0,
        int axis, bool zeroed
    );
    __declspec(dllexport) void mean_axis_ext(float* averaged,
        float* original, unsigned int z0, unsigned y0, unsigned x0,
        int axis
    );
}

__global__ void element_op_3d_kernel(float* dst, float* src, int op, int z0, int y0, int x0);
__global__ void element_op_3d_ret_kernel(float* c, float* a, float* b, int op, int z0, int y0, int x0);
__global__ void scalar_op_3d_inplace_kernel(float* dst, float scalar, int op, int z0, int y0, int x0);
__global__ void power_2_kernel(float* dst, int z0, int y0, int x0);
__global__ void exp_kernel(float* dst, int z0, int y0, int x0, float* max, float temperature);
__global__ void zeroes_3d_inplace_kernel(float* arr, int z0, int y0, int x0);
__global__ void round_3d_inplace_kernel(float* arr, int places, int z0, int y0, int x0);
__global__ void gradient_desc_3d_kernel(
    float* ptr, float* ptr_grad, float* velocity, float* momentum, float lr, float l2, 
    unsigned int z0, unsigned int y0, unsigned int x0, bool non_neg, float batch_size,
    int optimizer_type, float alpha, float beta
);
__global__ void broadcast_2d_to_3d_kernel(
    float* broadcasted, unsigned int z0, unsigned y0, unsigned x0,
    float* original, int axis
);
__global__ void sum_axis_kernel(
    float* summed,
    float* original, unsigned int z0, unsigned y0, unsigned x0,
    int axis, bool zeroed);
__global__ void sum_axis_kernel_reduction(float* summed,
    float* original, unsigned int z0, unsigned y0, unsigned x0,
    int axis, bool zeroed, int shared_memory_len);