#include "external.cuh"
#include "array_ops.cuh"
#include "activation.cuh"
#include <stdio.h>
#include <curand_kernel.h>
#include "dropout.cuh"

void dropout_forward_ext(
    float* input, float* output, float* mask, void* states,
    unsigned int z0, unsigned int y0, unsigned int x0, float dropout_rate
)
{
    KernelDim kernel_shape = get_kernel_dim1(z0, y0, x0, 1, 1, 128);
    curandState* states_casted = (curandState*) states;
    
    dropout_forward_kernel<<<kernel_shape.grid_dim, kernel_shape.block_dim>>>
    (input, output, mask, z0, y0, x0, states_casted, dropout_rate);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

void dropout_backward_ext(
    float* original_grads,
    float* dropout_mask, float* result_grads,
    unsigned int z0, unsigned int y0, unsigned int x0
)
{
    KernelDim kernel_shape;
    kernel_shape = get_kernel_dim1(z0, y0, x0, 1, 1, 128);

    dropout_backward_kernel<<<kernel_shape.grid_dim, kernel_shape.block_dim>>>
    (original_grads, dropout_mask, result_grads, z0, y0, x0);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

void* init_random_states_ext(
    unsigned int z0, unsigned int y0, unsigned int x0
)
{
    KernelDim kernel_shape = get_kernel_dim1(z0, y0, x0, 1, 1, 128);
    curandState* states;
    cudaMalloc((void**)&states, sizeof(curandState) * z0 * y0 * x0);
    
    init_random_states<<<kernel_shape.grid_dim, kernel_shape.block_dim>>>
    (states, time(NULL), z0, y0, x0);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }

    return (void*) states;
}

__global__ void init_random_states(
    curandState* states, unsigned long long int seed, 
    unsigned int z0, unsigned int y0, unsigned int x0)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (z < z0 && y < y0 && x < x0)
    {
        int idx = (z * y0 * x0) + (y * x0) + x;
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__device__ void random_dropout(
    curandState* states, float* dropout_mask, float dropout_rate, int idx
)
{
    float random_val = curand_uniform(&states[idx]);
    if (random_val <= dropout_rate)
    {
        dropout_mask[idx] = 0.0f;
    }
    else
    {
        dropout_mask[idx] = 1.0f;
    }
    dropout_mask[idx] /= (1.0f - dropout_rate);
}

__global__ void dropout_forward_kernel(
    float* tensor, float* output, float* dropout_mask,
    unsigned int z0, unsigned int y0, unsigned int x0, 
    curandState* states, float dropout_rate
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (z < z0 && y < y0 && x < x0)
    {
        int idx = (z * y0 * x0) + (y * x0) + x;

        random_dropout(states, dropout_mask, dropout_rate, idx);

        output[idx] = tensor[idx] * dropout_mask[idx];
    }
}

__global__ void dropout_backward_kernel(
    float* original_grads,
    float* dropout_mask, float* result_grads,
    unsigned int z0, unsigned int y0, unsigned int x0
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (z < z0 && y < y0 && x < x0)
    {
        int idx = (z * y0 * x0) + (y * x0) + x;
        // y = x * (mask/(1 - p))
        // gradient for x is (mask/(1 - p))
        // dropout mask is already divided by dropout prob sub by 1
        result_grads[idx] = original_grads[idx] * dropout_mask[idx];
    }
}

// combine hadamard and dropout for efficiency
void elementwise_dropout_forward_ext(
    float* input, float* weights, float* output, float* mask, void* states,
    unsigned int z0, unsigned int y0, unsigned int x0, float dropout_rate, unsigned int op,
    int activation_id, float act_scale, bool use_dropout, bool zero_output
)
{
    KernelDim kernel_shape = get_kernel_dim1(z0, y0, x0, 1, 1, 128);
    curandState* states_casted = (curandState*) states;
    
    elementwise_dropout_forward_kernel<<<kernel_shape.grid_dim, kernel_shape.block_dim>>>
    (   
        input, weights, output, mask, z0, y0, x0, states_casted, dropout_rate, op, 
        activation_id, act_scale, use_dropout, zero_output
    );

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

void elementwise_dropout_backward_ext(
    float* original_grads,
    float* dropout_mask, 
    float* input, float* weights, float* weight_grads,
    float* result_grads, bool use_dropout, int activation_id, float act_scale,
    unsigned int z0, unsigned int y0, unsigned int x0,
    unsigned int op, bool zero_input_grad, bool zero_weight_grad
)
{
    KernelDim kernel_shape;
    kernel_shape = get_kernel_dim1(z0, y0, x0, 1, 1, 128);

    elementwise_dropout_backward_kernel<<<kernel_shape.grid_dim, kernel_shape.block_dim>>>
    (
        original_grads, dropout_mask, input, weights, weight_grads, result_grads, use_dropout, 
        activation_id, act_scale, z0, y0, x0, op, zero_input_grad, zero_weight_grad
    );

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

__global__ void elementwise_dropout_forward_kernel(
    float* tensor, float* weights, float* output, float* dropout_mask,
    unsigned int z0, unsigned int y0, unsigned int x0, 
    curandState* states, float dropout_rate, unsigned int op, int activation_id, float act_scale,
    bool use_dropout, bool zero_output
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (z < z0 && y < y0 && x < x0)
    {
        int idx = (z * y0 * x0) + (y * x0) + x;

        float temp_prod;
        switch (op)
        {
            case 0:
                temp_prod = tensor[idx] + weights[idx];
                break;
            
            case 1:
                temp_prod = tensor[idx] * weights[idx];
                break;
        }

        // activation
        temp_prod = act_scale * function(temp_prod, activation_id);

        if (use_dropout)
        {
            random_dropout(states, dropout_mask, dropout_rate, idx);
            temp_prod *= dropout_mask[idx];
        }

        if (zero_output)
        {
            output[idx] = temp_prod;
        }
        else
        {
            output[idx] += temp_prod;
        }
    }
}

__global__ void elementwise_dropout_backward_kernel(
    float* original_grads,
    float* dropout_mask, 
    float* input, float* weights, float* weight_grads,
    float* result_grads, bool use_dropout, int activation_id, float act_scale,
    unsigned int z0, unsigned int y0, unsigned int x0, 
    unsigned int op, bool zero_input_grad, bool zero_weight_grad
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (z < z0 && y < y0 && x < x0)
    {
        int idx = (z * y0 * x0) + (y * x0) + x;
        // y = (a * b) * (mask/(1 - p))
        // gradient for a is: b * (mask/(1 - p))
        // dropout mask is already divided by dropout prob sub by 1
        float chained_grad = original_grads[idx];

        if (use_dropout)
        {
            chained_grad *= dropout_mask[idx];
        }
        
        float input_grad, weight_grad;
        switch (op)
        {
            case 0:
                chained_grad *= act_scale * function_deriv(input[idx] + weights[idx], activation_id);
                input_grad = chained_grad;
                weight_grad = chained_grad;
                break;
            
            case 1:
                chained_grad *= act_scale * function_deriv(input[idx] * weights[idx], activation_id);
                input_grad = chained_grad * weights[idx];
                weight_grad = chained_grad * input[idx];
                break;
            
            default:
                break;
        }

        if (zero_input_grad)
        {
            result_grads[idx] = input_grad;
        }
        else
        {
            result_grads[idx] += input_grad;
        }

        if (zero_weight_grad)
        {
            weight_grads[idx] = weight_grad;
        }
        else
        {
            weight_grads[idx] += weight_grad;
        }
    }
}