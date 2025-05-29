#include "external.cuh"
#include "activation.cuh"
#include "array_ops.cuh"
#include "dense_funcs.cuh"
#include <math.h>
#include <stdio.h>

void activation3d_cuda_ext(
    float* result, float* mat, int z, int y, int x, int func_id,
    float scale, bool zero_output
)
{
    //int thread_size = 32;
    //int gridx = (int) ceil(((float) x) / ((float) thread_size));
    //int gridy = (int) ceil(((float) y) / ((float) thread_size));
    //int gridz = (int) ceil(((float) z) / ((float) thread_size));
    
    //dim3 gridsize(gridx, gridy, gridz);
    //dim3 blocksize(thread_size, thread_size, 1);
    //int minGridSize, blockSize;
    //cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, activation_kernel, 0, 0);
    //printf("%d, %d\n", minGridSize, blockSize);

    KernelDim kernel_size = get_kernel_dim1(z, y, x, 1, 1, 128);
    //KernelDim kernel_size = get_kernel_dim(z * y * x);
    //printf("%d", zero_output);
    activation_kernel<<<kernel_size.grid_dim, kernel_size.block_dim>>>(result, mat, z, y, x, func_id, scale, zero_output);
    cudaDeviceSynchronize();
}

void activation3d_backward_cuda_ext(
    float* chained_grads, float* inputs, float* original_grad, int z, int y, int x, int func_id,
    float scale, bool zero_input_grad
)
{
    //int thread_size = 32;
    //int gridx = (int) ceil(((float) x) / ((float) thread_size));
    //int gridy = (int) ceil(((float) y) / ((float) thread_size));
    //int gridz = (int) ceil(((float) z) / ((float) thread_size));
    
    //dim3 gridsize(gridx, gridy, gridz);
    //dim3 blocksize(thread_size, thread_size, 1);

    //int minGridSize, blockSize;
    //cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, activation_back_kernel, 0, 0);
    //printf("%d, %d\n", minGridSize, blockSize);

    KernelDim kernel_size = get_kernel_dim1(z, y, x, 1, 1, 128);
    //KernelDim kernel_size = get_kernel_dim(z * y * x);
    activation_back_kernel<<<kernel_size.grid_dim, kernel_size.block_dim>>>
    (chained_grads, inputs, original_grad, z, y, x, func_id, scale, zero_input_grad); 
        //a, a_grads, b, b_grads);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    //printf("%d\n", gridx);
}

__global__ void activation_kernel(
    float* result, float* mat, int z0, int y0, int x0, int func_id,
    float scale, bool zero_output
    )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    /*
    unsigned int flattened_idx = idx4_to_idx1(
        0, z, y, x,
        Tuple4D
        {
            1,
            gridDim.z * blockDim.z,
            gridDim.y * blockDim.y,
            gridDim.x * blockDim.x
        }
    );
    */

    if (x < x0 && y < y0 && z < z0)
    {
        int flattened_idx = z * y0 * x0 + y * x0 + x;

        if (zero_output)
        {
            result[flattened_idx] = 0.0f;
        }
        // f(a * x + b)
        result[flattened_idx] = 
            scale * function(mat[flattened_idx], func_id);
        //function(result, mat, func_id, flattened_idx);
    }
}

// all tensors have the same shape for activation
__global__ void activation_back_kernel(
    float* chained_grads, float* inputs, float* original_grads, 
    int z0, int y0, int x0, int func_id,
    float scale, bool zero_input_grad
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    /*
    unsigned int flattened_idx = idx4_to_idx1(
        0, z, y, x,
        Tuple4D
        {
            1,
            gridDim.z * blockDim.z,
            gridDim.y * blockDim.y,
            gridDim.x * blockDim.x
        }
    );
    */

    //printf("%d\n", func_id);
    if (x < x0 && y < y0 && z < z0)
    //if (flattened_idx < x0 * y0 * z0)
    {
        int flattened_idx = z * y0 * x0 + y * x0 + x;

        // calculate derivative of activation function and perform chain rule
        // input pointer will be overwritten by next forward pass
        //function_deriv(inputs, func_id, flattened_idx);
        if (zero_input_grad)
        {
            chained_grads[flattened_idx] = 0.0f;
        }
        chained_grads[flattened_idx] = original_grads[flattened_idx] * 
            scale * function_deriv(inputs[flattened_idx], func_id);
        //a_grads[flattened_idx] = original_grads[flattened_idx] *
        //    inputs[flattened_idx] * function_deriv(a[flattened_idx] * inputs[flattened_idx] + b[flattened_idx], func_id);
        //b_grads[flattened_idx] = original_grads[flattened_idx] * 
        //    function_deriv(a[flattened_idx] * inputs[flattened_idx] + b[flattened_idx], func_id);
    }
}

// used for activation functions and derivatives
__device__ float function(float val, int func_id)
{   
    float activated;
    switch (func_id)
    {
        case 0:
            activated = sigmoid(val);
            break;
        case 1:
            activated = tanhf(val);
            break;
        case 2:
            activated = silu(val);
            break;
        case 3:
            activated = gelu(val);
            break;
        case 4:
            activated = tanh2(val);
            break;
        case 5:
            activated = softplus(val);
            break;
        case 6:
            activated = linear(val);
            break;
    }

    return activated;
}

// used for activation functions and derivatives
__device__ float function_deriv(float val, int func_id)
{   
    float activation_grad;
    switch (func_id)
    {
        case 0:
            //printf("%f\n", dst[flattened_idx]);
            activation_grad = sigmoid_deriv(val);
            //printf("deriv: %f\n", dst[flattened_idx]);
            break;
        case 1:
            //printf("%f\n", dst[flattened_idx]);
            activation_grad = tanh_deriv(val);
            //printf("%f\n", dst[flattened_idx]);
            break;
        case 2:
            //printf("%f\n", dst[flattened_idx]);
            activation_grad = silu_deriv(val);
            //printf("%f\n", dst[flattened_idx]);
            break;
        case 3:
            activation_grad = gelu_deriv(val);
            break;
        case 4:
            activation_grad = tanh2_deriv(val);
            break;
        case 5:
            activation_grad = softplus_deriv(val);
            break;
        case 6:
            activation_grad = linear_deriv(val);
            break;
    }
    return activation_grad;
}

__device__ float sigmoid(float v)
{
    return (1.0f + tanhf(v)) / 2.0f;
}

__device__ float sigmoid_deriv(float v)
{
    return (1.0f - (tanhf(v) * tanhf(v))) / 2.0f;
}

__device__ float tanh_deriv(float v)
{
    return 1.0f - (tanhf(v) * tanhf(v));
}

__device__ float tanh2(float v)
{
    return 2.0f * tanhf(v);
    //return sinf(v);
}

__device__ float tanh2_deriv(float v)
{
    return 2.0f * tanh_deriv(v);
    //return cosf(v);
}

__device__ float silu(float v)
{
    return v * sigmoid(v);
}

__device__ float silu_deriv(float v)
{
    return (1.0f + tanhf(v) + v * tanh_deriv(v)) / 2.0f;
}

__device__ float gelu(float v)
{
    return 0.5f * v * (1.0f + tanhf(0.79788456f * v * (1.0f + 0.044715f * v * v)));
}

__device__ float gelu_deriv(float v)
{
    return 0.5f * (1.0f + tanh(0.79788456f * v * (0.044715f * v * v + 1.0f))) + 
            0.79788456f * v * tanh_deriv(0.79788456f * v * (0.044715f * v * v + 1.0f)) * 
            (0.134145f * v * v + 1.0f);
}

__device__ float softplus(float v)
{
    return logf(1.0f + expf(v));
}

__device__ float softplus_deriv(float v)
{
    return 1.0f / (1.0f + exp(-v));
}

__device__ float linear(float v)
{
    return v;
}

__device__ float linear_deriv(float v)
{
    return 1.0f;
}