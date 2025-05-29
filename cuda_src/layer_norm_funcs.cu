#include "layer_norm_funcs.cuh"
#include "external.cuh"
#include "array_ops.cuh"
#include "reduction_funcs.cuh"
#include <stdio.h>
#include <iostream>

void layer_norm_ext(
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
)
{   
    int z1, y1, x1;
    switch (axis)
    {
        case 0:
            z1 = 1; y1 = y0; x1 = x0;
            break;
        case 1:
            z1 = z0; y1 = 1; x1 = x0;
            break;
        case 2:
            z1 = z0; y1 = y0; x1 = 1;
            break;
    }

    KernelDim kernel_shape, kernel_shape1;
    int shared_memory_len;
    float axis_dim;
    switch (axis)
    {
        case 0:
            kernel_shape = get_kernel_dim1(1, y0, x0, 1, 16, 16);
            kernel_shape1 = get_kernel_dim1(z0, y0, x0, LAYER_NORM_BLOCK_SIZE, 1, 1);
            axis_dim = (float) z0;
            shared_memory_len = LAYER_NORM_BLOCK_SIZE;
            break;
        case 1:
            kernel_shape = get_kernel_dim1(z0, 1, x0, 16, 1, 16);
            kernel_shape1 = get_kernel_dim1(z0, y0, x0, 1, LAYER_NORM_BLOCK_SIZE, 1);
            axis_dim = (float) y0;
            shared_memory_len = LAYER_NORM_BLOCK_SIZE;
            break;
        case 2:
            kernel_shape = get_kernel_dim1(z0, y0, 1, 16, 16, 1);
            kernel_shape1 = get_kernel_dim1(z0, y0, x0, 1, 1, LAYER_NORM_BLOCK_SIZE);
            shared_memory_len = LAYER_NORM_BLOCK_SIZE;
            axis_dim = (float) x0;
            break;
    }

    // calculate mean and var
    //cudaEvent_t start, stop;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);

    //cudaEventRecord(start);

    /**/
    zeroes_3d_ext(mean_ptr, z1, y1, x1);
    mean_axis_kernel_reduction
    <<<kernel_shape1.grid_dim, kernel_shape1.block_dim>>>
    (mean_ptr, original_ptr, z0, y0, x0, axis, true, shared_memory_len);
    scalar_op_3d_inplace_ext(mean_ptr, axis_dim, 3, z1, y1, x1);

    cudaDeviceSynchronize();

    zeroes_3d_ext(var_ptr, z1, y1, x1);
    var_axis_kernel_reduction
    <<<kernel_shape1.grid_dim, kernel_shape1.block_dim>>>
    (var_ptr, mean_ptr, original_ptr, z0, y0, x0, axis, true, shared_memory_len);
    scalar_op_3d_inplace_ext(var_ptr, axis_dim, 3, z1, y1, x1);
    
    cudaDeviceSynchronize();

    z_scoring
    <<<kernel_shape1.grid_dim, kernel_shape1.block_dim>>>
    (output_ptr, normalized_ptr, original_ptr, mean_ptr, var_ptr, alphas, betas, z0, y0, x0, axis);
    
    cudaDeviceSynchronize();
    /**/

    // perform standarization
    // multiply results by alpha and add beta
    // can all be done all in one kernel

    //standarize_tensor_kernel
    //<<<kernel_shape.grid_dim, kernel_shape.block_dim>>>
    //(output_ptr, normalized_ptr, original_ptr, mean_ptr, var_ptr, alphas, betas, z0, y0, x0, axis);

    //cudaEventRecord(stop);
    //cudaEventSynchronize(stop);
    //float milliseconds = 0;
    //cudaEventElapsedTime(&milliseconds, start, stop);

    //std::cout << "Kernel execution time (layer_norm): " << milliseconds / 1000.0f << " ms" << std::endl;
    //cudaEventDestroy(start);
    //cudaEventDestroy(stop);
}

__global__ void standarize_tensor_kernel(
    float* output_ptr,
    float* normalized_ptr, 
    float* original_ptr, 
    float* mean_ptr, float* var_ptr, // e.g. both (1, y0, x0) if performing along z axis
    float* alphas, float* betas,
    unsigned int z0, unsigned int y0, unsigned x0, 
    int axis
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int loop_count;
    unsigned int z1, y1, x1;
    switch (axis)
    {
        case 0:
            loop_count = z0; z1 = 1; y1 = y0; x1 = x0;
            break;
        case 1:
            loop_count = y0; z1 = z0; y1 = 1; x1 = x0; 
            break;
        case 2:
            loop_count = x0; z1 = z0; y1 = y0; x1 = 1;
            break;
    }

    // limit is placed on one of the axis
    if (z < z1 && y < y1 && x < x1)
    {
        int target_idx = (z * y1 * x1) + (y * x1) + x; // always stays on axis due to limit (e.g. 1, 10, 10)

        // initialize results to zeros
        mean_ptr[target_idx] = 0.0f;

        int broadcasted_idx;

        // sum along axis
        for (int i = 0; i < loop_count; i++)
        {
            broadcasted_idx = get_sliced_idx(z, y, x, z0, y0, x0, axis, i);

            mean_ptr[target_idx] += original_ptr[broadcasted_idx];
        }

        // divide by loop count
        mean_ptr[target_idx] /= (float) loop_count;

        // calculate variance
        // initialize results to zeros
        var_ptr[target_idx] = 0.0f;
        
        // sum along axis
        for (int i = 0; i < loop_count; i++)
        {
            broadcasted_idx = get_sliced_idx(z, y, x, z0, y0, x0, axis, i);
            var_ptr[target_idx] += 
                (original_ptr[broadcasted_idx] - mean_ptr[target_idx]) *
                    (original_ptr[broadcasted_idx] - mean_ptr[target_idx]);
        }

        // divide by loop count
        var_ptr[target_idx] /= (float) loop_count;

        // standarize (x - mean) / std
        for (int i = 0; i < loop_count; i++)
        {
            broadcasted_idx = get_sliced_idx(z, y, x, z0, y0, x0, axis, i);
            normalized_ptr[broadcasted_idx] = 
                (original_ptr[broadcasted_idx] - mean_ptr[target_idx]) / sqrtf(var_ptr[target_idx] + 1e-8f);

            // apply alpha and beta parameters
            output_ptr[broadcasted_idx] = alphas[broadcasted_idx] * normalized_ptr[broadcasted_idx] + betas[broadcasted_idx];
        }
    }
}

__global__ void z_scoring(
    float* output_ptr,
    float* normalized_ptr, 
    float* original_ptr, 
    float* mean_ptr, float* var_ptr, // e.g. both (1, y0, x0) if performing along z axis
    float* alphas, float* betas,
    unsigned int z0, unsigned int y0, unsigned x0, 
    int axis)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    unsigned int z1, y1, x1;
    switch (axis)
    {
        case 0:
            z1 = 1; y1 = y0; x1 = x0;
            break;
        case 1:
            z1 = z0; y1 = 1; x1 = x0; 
            break;
        case 2:
            z1 = z0; y1 = y0; x1 = 1;
            break;
    }

    if (z < z0 && y < y0 && x < x0)
    {
        // get index of assigned mean and variance
        unsigned int mean_var_idx = get_sliced_idx(z, y, x, z1, y1, x1, axis, 0);
        unsigned int standarized_idx = (z * y0 * x0) + (y * x0) + x;

        // standarize with mean = 0 and std = 1
        normalized_ptr[standarized_idx] = 
            (original_ptr[standarized_idx] - mean_ptr[mean_var_idx]) / sqrtf(var_ptr[mean_var_idx] + 1e-8f);
        
        // multiply with alpha and add beta
        output_ptr[standarized_idx] = 
            alphas[standarized_idx] * normalized_ptr[standarized_idx] + betas[standarized_idx];
    }
}

void layer_norm_back_ext(
    float* return_grads,
    float* original_grads,
    float* original_inputs_ptr,
    float* normalized_ptr, 
    float* mean_ptr, float* var_ptr, // e.g. both (1, y0, x0) if performing along z axis
    float* mean_grads, float* var_grads, // (z0, y0, x0)
    float* alpha_ptr, float* beta_ptr, float* alpha_grad, float* beta_grad,
    int z0, int y0, int x0,
    int axis
)
{
    KernelDim kernel_shape;
    kernel_shape = get_kernel_dim1(z0, y0, x0, 1, 1, 256);
    
    layer_norm_back_kernel<<<kernel_shape.grid_dim, kernel_shape.block_dim>>>(
        return_grads, original_grads, original_inputs_ptr, normalized_ptr, 
        mean_ptr, var_ptr, mean_grads, var_grads,
        alpha_ptr, beta_ptr, alpha_grad, beta_grad, z0, y0, x0, axis
    );

    cudaDeviceSynchronize();
}

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
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    unsigned int z1, y1, x1;
    float length;
    switch (axis)
    {
        case 0:
            z1 = 1; y1 = y0; x1 = x0; length = (float) z0;
            break;
        case 1:
            z1 = z0; y1 = 1; x1 = x0; length = (float) y0;
            break;
        case 2:
            z1 = z0; y1 = y0; x1 = 1; length = (float) x0;
            break;
    }

    if (z < z0 && y < y0 && x < x0)
    {
        unsigned int idx = (z * y0 * x0) + (y * x0) + x;
        // y = a * x + b
        // gradient for a = x
        // gradient for x = a
        // gradient for b = 1

        alpha_grad[idx] += original_grads[idx] * normalized_ptr[idx];
        beta_grad[idx] += original_grads[idx];

        // chained gradient
        original_grads[idx] *= alpha_ptr[idx];

        // y = (x - mean) / sqrt(var + 1e-8)
        // get index of require mean/variance
        unsigned int mean_var_idx = get_sliced_idx(z, y, x, z1, y1, x1, axis, 0);


        // must refer to draw.io graph to understand backpropagation paths

        return_grads[idx] = 0.0f;
        mean_grads[idx] = 0.0f;
        var_grads[idx] = 0.0f;

        // gradient for x = 1 / sqrt(var + 1e-8)
        return_grads[idx] = original_grads[idx] * (1.0f / sqrtf(var_ptr[mean_var_idx] + 1e-8f));
        // gradient for mean = -1 / sqrt(var + 1e-8)
        mean_grads[idx] = original_grads[idx] * (-1.0f / sqrtf(var_ptr[mean_var_idx] + 1e-8f));
        // gradient for var = -1 / sqrt(var + 1e-8)
        var_grads[idx] = original_grads[idx] * 
            -((original_inputs_ptr[idx] - mean_ptr[mean_var_idx]) / 
                (2.0f * powf(var_ptr[mean_var_idx] + 1e-8f, 3.0f/2.0f)));

        
        // calculate the gradient for the mean going from
        // norm -> var -> mean
        mean_grads[idx] += var_grads[idx] * ((-2.0f * (original_inputs_ptr[idx] - mean_ptr[mean_var_idx])) / length);

        // calculate the gradient for input going from
        // norm -> var -> input
        return_grads[idx] += var_grads[idx] * ((2.0f * (original_inputs_ptr[idx] - mean_ptr[mean_var_idx])) / length);

        // calculate the gradient going from 
        // mean -> inputs
        return_grads[idx] += mean_grads[idx] * (1.0f / length);
    }
}

__device__ unsigned int get_sliced_idx(
    int z, int y, int x,
    int z0, int y0, int x0, int axis,
    int idx_on_axis
)
{
    unsigned int sliced_idx; 
    switch (axis)
    {
        case 0:
            sliced_idx = (idx_on_axis * y0 * x0) + (y * x0) + x;
            break;
        case 1:
            sliced_idx = (z * y0 * x0) + (idx_on_axis * x0) + x;
            break;
        case 2:
            sliced_idx = (z * y0 * x0) + (y * x0) + idx_on_axis;
            break;
    }

    return sliced_idx;
}

__global__ void mean_axis_kernel_reduction(float* mean,
    float* original, unsigned int z0, unsigned y0, unsigned x0,
    int axis, bool zeroed, int shared_memory_len)
{
    int thread_level_idx, axis_idx, block_axis_idx;
    unsigned int global_idx, target_idx, axis_dim;

    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    //extern __shared__ float shared_block_buffer[];
    __shared__ float shared_block_buffer[LAYER_NORM_BLOCK_SIZE];

    //if (z < z0 && y < y0 && x < x0)
    {
        // get global index
        global_idx = idx4_to_idx1(0, z, y, x, Tuple4D {1, z0, y0, x0});

        //printf("%u, %u, %u\n", z, y, x);
        switch (axis)
        {
            case 0:
                thread_level_idx = threadIdx.z; 
                axis_idx = z;
                axis_dim = z0;
                target_idx = idx4_to_idx1(0, 0, y, x, Tuple4D {1, 1, y0, x0});
                block_axis_idx = blockIdx.z;
                break;

            case 1:
                thread_level_idx = threadIdx.y; 
                axis_idx = y;
                axis_dim = y0;
                target_idx = idx4_to_idx1(0, z, 0, x, Tuple4D {1, z0, 1, x0});
                block_axis_idx = blockIdx.y;
                break;

            case 2:
                thread_level_idx = threadIdx.x;
                axis_idx = x;
                axis_dim = x0;
                target_idx = idx4_to_idx1(0, z, y, 0, Tuple4D {1, z0, y0, 1});
                block_axis_idx = blockIdx.x;
                break;
        }

        // copy elements from array to shared block memory
        if (axis_idx < axis_dim)
        {
            // divide by count, equivalent of mean when summing
            shared_block_buffer[thread_level_idx] = original[global_idx];// / (float) axis_dim;
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
            atomicAdd(&mean[target_idx], shared_block_buffer[0]);
        }
    }
}

__global__ void var_axis_kernel_reduction(
    float* var,
    float* mean,
    float* original, unsigned int z0, unsigned y0, unsigned x0,
    int axis, bool zeroed, int shared_memory_len)
{
    int thread_level_idx, axis_idx, block_axis_idx;
    unsigned int global_idx, target_idx, axis_dim;

    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    //extern __shared__ float shared_block_buffer[];
    __shared__ float shared_block_buffer[LAYER_NORM_BLOCK_SIZE];

    //if (z < z0 && y < y0 && x < x0)
    {
        // get global index
        global_idx = idx4_to_idx1(0, z, y, x, Tuple4D {1, z0, y0, x0});

        //printf("%u, %u, %u\n", z, y, x);
        switch (axis)
        {
            case 0:
                thread_level_idx = threadIdx.z; 
                axis_idx = z;
                axis_dim = z0;
                target_idx = idx4_to_idx1(0, 0, y, x, Tuple4D {1, 1, y0, x0});
                block_axis_idx = blockIdx.z;
                break;

            case 1:
                thread_level_idx = threadIdx.y; 
                axis_idx = y;
                axis_dim = y0;
                target_idx = idx4_to_idx1(0, z, 0, x, Tuple4D {1, z0, 1, x0});
                block_axis_idx = blockIdx.y;
                break;

            case 2:
                thread_level_idx = threadIdx.x;
                axis_idx = x;
                axis_dim = x0;
                target_idx = idx4_to_idx1(0, z, y, 0, Tuple4D {1, z0, y0, 1});
                block_axis_idx = blockIdx.x;
                break;
        }

        // copy elements from array to shared block memory
        if (axis_idx < axis_dim)
        {
            // (x - u)^2 / count
            shared_block_buffer[thread_level_idx] = 
                ((original[global_idx] - mean[target_idx]) * 
                    (original[global_idx] - mean[target_idx]));// / 
                        //(float) axis_dim;
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
            atomicAdd(&var[target_idx], shared_block_buffer[0]);
        }
    }
}