#include "external.cuh"
#include "array_ops.cuh"
#include "l2norm.cuh"
#include "softmax.cuh"
#include <stdio.h>

void l2norm_forward_ext(
    float* original, float* original_pow2, unsigned int z0, unsigned y0, unsigned x0,
    float* power_sum, float* normalized, bool zeroed
)
{
    KernelDim kernel_shape;
    int shared_memory_len;
    unsigned int z1, y1, x1;
    kernel_shape = get_kernel_dim1(z0, y0, x0, 1, 1, 128);
    shared_memory_len = 128;
    z1 = z0; y1 = y0; x1 = 1;    

    // calculate l2 norm
    calc_pow2sum_kernel_reduction<<<kernel_shape.grid_dim, kernel_shape.block_dim, shared_memory_len * sizeof(float)>>>
    (power_sum, original, z0, y0, x0, zeroed, shared_memory_len);

    cudaDeviceSynchronize();

    // broadcast divide the norm across input
    //cuda_to_cuda_ext(normalized, original, z0 * y0 * x0);
    divide_inputs_kernel<<<kernel_shape.grid_dim, kernel_shape.block_dim>>>(
        normalized, original, power_sum, z0, y0, x0, 2
    );

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

void l2norm_backward_ext(
    float* original_grads, unsigned int z0, unsigned y0, unsigned x0,
    float* power_sum, float* original_inputs, float* original_outputs, float* result_grads,
    bool zeroed
)
{
    KernelDim kernel_shape;
    int shared_memory_len;
    unsigned int z1, y1, x1;
    kernel_shape = get_kernel_dim1(z0, y0, x0, 1, 1, 128);
    shared_memory_len = 128;
    z1 = z0; y1 = y0; x1 = 1;

    l2norm_backward_kernel<<<kernel_shape.grid_dim, kernel_shape.block_dim>>>
    (
        result_grads, original_grads, original_inputs, original_outputs, 
        power_sum, z0, y0, x0, zeroed
    );

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
}


__global__ void calc_pow2sum_kernel_reduction(float* summed,
    float* original, unsigned int z0, unsigned y0, unsigned x0,
    bool zeroed, int shared_memory_len)
{
    int thread_level_idx, axis_idx, block_axis_idx;
    unsigned int global_idx, target_idx, axis_dim;

    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float shared_block_buffer[];

    //if (z < z0 && y < y0 && x < x0)
    {
        // get global index
        global_idx = idx4_to_idx1(0, z, y, x, Tuple4D {1, z0, y0, x0});

        //printf("%u, %u, %u\n", z, y, x);
        thread_level_idx = threadIdx.x;
        axis_idx = x;
        axis_dim = x0;
        target_idx = idx4_to_idx1(0, z, y, 0, Tuple4D {1, z0, y0, 1});
        block_axis_idx = blockIdx.x;

        if (zeroed)
        {
            summed[target_idx] = 0.0f;
        }

        // copy elements from array to shared block memory
        if (axis_idx < axis_dim)
        {
            // raise to power of 2 before reduction
            // l2norm = sum(x^2)
            shared_block_buffer[thread_level_idx] = original[global_idx] * original[global_idx];
            //printf("%d, %d, %u\n", thread_level_idx, axis_idx, axis_dim);
        }
        else
        {
            // if global index is larger, only will happen in the last block
            shared_block_buffer[thread_level_idx] = 0.0f;
            //printf("thread_idx: %d, axis_idx: %d, axis_dim: %u\n", thread_level_idx, axis_idx, axis_dim);
        }

        // ensure all elements in shared memory have been assigned
        __syncthreads();

        // parallel reduction
        // log2(1024) = 10
        int length = shared_memory_len / 2;
        int loop_count = (int) log2f(shared_memory_len);
        for (int i = 0; i < loop_count; i++)
        {
            if (thread_level_idx < length)
            {
                shared_block_buffer[thread_level_idx] += 
                    shared_block_buffer[thread_level_idx + length];
            }
            __syncthreads();

            length /= 2;
        }
        
        // atomic add to summed at target index based on axis
        if (thread_level_idx == 0)
        {
            atomicAdd(&summed[target_idx], shared_block_buffer[0]);
        }
    }
}

__global__ void divide_inputs_kernel(
    float* broadcasted, float* original, float* power_sum, 
    unsigned int z0, unsigned y0, unsigned x0, int axis
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

    if (z < z0 && y < y0 && x < x0)
    //if (flattened_idx < z0 * y0 * x0)
    {
        //Tuple4D idx3d = idx1_to_idx4(flattened_idx, Tuple4D {1, z0, y0, x0});
        //int z = idx3d.z;
        //int y = idx3d.y;
        //int x = idx3d.x;
        int flattened_broadcast_idx = (z * y0 * x0) + (y * x0) + x;
        int flattened_original_idx;
        switch (axis)
        {
            case 0:
                flattened_original_idx = (0 * y0 * x0) + (y * x0) + x;
                break;
            case 1:
                flattened_original_idx = (z * 1 * x0) + (0 * x0) + x;
                break;
            case 2:
                flattened_original_idx = (z * y0 * 1) + (y * 1) + 0;
                break;
        }

        //if (zero_output)
        //{
        //    broadcasted[flattened_broadcast_idx] = 0.0f;
        //}
        broadcasted[flattened_broadcast_idx] = 
            original[flattened_broadcast_idx] / sqrtf(power_sum[flattened_original_idx]);
    }
}

__global__ void l2norm_backward_kernel(
    float* result_grads, float* original_grads, 
    float* inputs, float* outputs, float* power_sum, int z0, int y0, int x0,
    bool zeroed
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (z < z0 && y < y0 && x < x0)
    {
        int result_idx = (z * y0 * x0) + (y * x0) + x;
        int norm_idx = (z * y0 * 1) + (y * 1) + 0;

        //y = x / l2norm(x)
        if (zeroed)
        {
            result_grads[result_idx] = 0.0f;
        }

        // chain rule from y to x
        result_grads[result_idx] += (original_grads[result_idx]) / (sqrtf(power_sum[norm_idx]));
    }
}