#pragma once

extern "C"
{
    __declspec(dllexport) void conv2d_forward_ext(
        float* input_img, float* filters, float* output_img,
        unsigned int in_z, unsigned int in_y, unsigned int in_x,
        unsigned int out_z, unsigned int out_y, unsigned int out_x,
        unsigned int filter_dim, unsigned int strides, float* biases, bool zero_output
    );

    __declspec(dllexport) void conv2d_backward_ext(
        float* input_img, float* input_grads, float* input_grads_count,
        float* filters, float* filter_grads, float* filter_grads_count,
        float* output_grads, float* bias_grads,
        unsigned int in_z, unsigned int in_y, unsigned int in_x,
        unsigned int out_z, unsigned int out_y, unsigned int out_x,
        unsigned int filter_dim, unsigned int strides
    );
}

__global__ void conv2d_forward_kernel(
    float* input_img, float* filters, float* output,
    unsigned int in_z, unsigned int in_y, unsigned int in_x,
    unsigned int out_z, unsigned int out_y, unsigned int out_x,
    unsigned int filter_dim, unsigned int strides, float* biases, bool zero_output
);

__global__ void conv2d_backward_kernel(
    float* input_img, float* input_grads, float* input_grads_count,
    float* filters, float* filter_grads, float* filter_grads_count,
    float* output_grads, float* bias_grads,
    unsigned int in_z, unsigned int in_y, unsigned int in_x,
    unsigned int out_z, unsigned int out_y, unsigned int out_x,
    unsigned int filter_dim, unsigned int strides
);