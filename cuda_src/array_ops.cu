#include "array_ops.cuh"
#include "external.cuh"
#include "dense_funcs.cuh"
#include <stdio.h>
#include <iostream>

void element_op_3d_ext(float* dst, float* src, int op, int z0, int y0, int x0)
{
    KernelDim size_dim = get_kernel_dim1(z0, y0, x0, 1, 1, 128);

    element_op_3d_kernel<<<size_dim.grid_dim, size_dim.block_dim>>>(dst, src, op, z0, y0, x0);

    cudaDeviceSynchronize();
}

__global__ void element_op_3d_kernel(float* dst, float* src, int op, int z0, int y0, int x0)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    // convert original thread index to flattened
    //unsigned int flattened_idx = idx4_to_idx1(
    //    0, z, y, x, 
    //    Tuple4D { 
    //        1,
    //        gridDim.z * blockDim.z,
    //        gridDim.y * blockDim.y,
    //        gridDim.x * blockDim.x
    //    }
    //);
    
    // check if global flattened index is within the flattened shape of target shape
    if (z < z0 && y < y0 && x < x0)
    {
        int flattened_idx = (z * y0 * x0) + (y * x0) + x;
        switch (op)
        {
            case 0:
                dst[flattened_idx] += src[flattened_idx];
                break;
            case 1:
                dst[flattened_idx] -= src[flattened_idx];
                break;
            case 2:
                dst[flattened_idx] *= src[flattened_idx];
                break;
            case 3:
                // ignore to prevent division by zero
                if (src[flattened_idx] != 0.0f)
                {
                    dst[flattened_idx] /= src[flattened_idx];
                }
                break;
        }
    }
}

void element_op_3d_ret_ext(float* c, float* a, float* b, int op, int z0, int y0, int x0)
{
    KernelDim size_dim = get_kernel_dim1(z0, y0, x0, 1, 1, 128);

    element_op_3d_ret_kernel<<<size_dim.grid_dim, size_dim.block_dim>>>(c, a, b, op, z0, y0, x0);

    cudaDeviceSynchronize();
}

__global__ void element_op_3d_ret_kernel(float* c, float* a, float* b, int op, int z0, int y0, int x0)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    // convert original thread index to flattened
    //unsigned int flattened_idx = idx4_to_idx1(
    //    0, z, y, x, 
    //    Tuple4D { 
    //        1,
    //        gridDim.z * blockDim.z,
    //        gridDim.y * blockDim.y,
    //        gridDim.x * blockDim.x
    //    }
    //);
    
    // check if global flattened index is within the flattened shape of target shape
    if (z < z0 && y < y0 && x < x0)
    {
        int flattened_idx = (z * y0 * x0) + (y * x0) + x;
        switch (op)
        {
            case 0:
                c[flattened_idx] = a[flattened_idx] + b[flattened_idx];
                break;
            case 1:
                c[flattened_idx] = a[flattened_idx] - b[flattened_idx];
                break;
            case 2:
                c[flattened_idx] = a[flattened_idx] * b[flattened_idx];
                break;
            case 3:
                c[flattened_idx] = a[flattened_idx] / b[flattened_idx];
                break;
        }
    }
}

void scalar_op_3d_inplace_ext(float* dst, float scalar, int op, int z0, int y0, int x0)
{
    KernelDim kernel_shape = get_kernel_dim1(z0, y0, x0, 1, 1, 128);
    scalar_op_3d_inplace_kernel<<<kernel_shape.grid_dim, 128>>>(dst, scalar, op, z0, y0, x0);

    cudaDeviceSynchronize();
}

__global__ void scalar_op_3d_inplace_kernel(float* dst, float scalar, int op, int z0, int y0, int x0)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    //unsigned int flattened_idx = idx4_to_idx1(
    //    0, z, y, x, 
    //    Tuple4D { 
    //        1,
    //        gridDim.z * blockDim.z,
    //        gridDim.y * blockDim.y,
    //        gridDim.x * blockDim.x
    //    }
    //);
    
    if (z < z0 && y < y0 && x < x0)
    {
        int flattened_idx = (z * y0 * x0) + (y * x0) + x;
        switch (op)
        {
            case 0:
                dst[flattened_idx] += scalar;
                break;
            case 1:
                dst[flattened_idx] -= scalar;
                break;
            case 2:
                dst[flattened_idx] *= scalar;
                break;
            case 3:
                dst[flattened_idx] /= scalar;
                break;
        }
    }
}

__global__ void power_2_kernel(float* dst, int z0, int y0, int x0)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    // check if global flattened index is within the flattened shape of target shape
    if (z < z0 && y < y0 && x < x0)
    {
        int flattened_idx = (z * y0 * x0) + (y * x0) + x;
        dst[flattened_idx] *= dst[flattened_idx];
    }
}

__global__ void exp_kernel(float* dst, int z0, int y0, int x0, float* max, float temperature)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    // check if global flattened index is within the flattened shape of target shape
    if (z < z0 && y < y0 && x < x0)
    {
        int flattened_idx = (z * y0 * x0) + (y * x0) + x;
        int flattened_max_idx = (z * y0 * 1) + (y * 1) + 0;
        dst[flattened_idx] = expf((dst[flattened_idx] - max[flattened_max_idx]) / temperature);
    }
}

void zeroes_3d_ext(float* dst, int z0, int y0, int x0)
{
    cudaMemset(dst, 0, sizeof(float) * z0 * y0 * x0);
}

void ones_3d_ext(float* dst, int z0, int y0, int x0)
{
    cudaMemset(dst, 1, sizeof(float) * z0 * y0 * x0);
}

__global__ void zeroes_3d_inplace_kernel(float* arr, int z0, int y0, int x0)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    //unsigned int flattened_idx = idx4_to_idx1(
    //    0, z, y, x, 
    //    Tuple4D { 
    //        1,
    //        gridDim.z * blockDim.z,
    //        gridDim.y * blockDim.y,
    //        gridDim.x * blockDim.x
    //    }
    //);

    if (z < z0 && y < y0 && x < x0)
    {
        int flattened_idx = (z * y0 * x0) + (y * x0) + x;
        arr[flattened_idx] = 0.0f;
    }
}

void round_3d_ext(float* arr, int places, int z0, int y0, int x0)
{
    int thread_size = 32;
    int gridx = (int) ceil(((float) x0) / ((float) thread_size));
    int gridy = (int) ceil(((float) y0) / ((float) thread_size));
    int gridz = (int) ceil(((float) z0) / ((float) thread_size));
    
    dim3 gridsize(gridx, gridy, gridz);
    dim3 blocksize(thread_size, thread_size, 1);

    round_3d_inplace_kernel<<<gridsize, blocksize>>>(arr, places, z0, y0, x0);
}

__global__ void round_3d_inplace_kernel(float* arr, int places, int z0, int y0, int x0)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (z < z0 && y < y0 && x < x0)
    {
        int flattened_idx = (z * y0 * x0) + (y * x0) + x;
        float val = arr[flattened_idx] * powf(10.0f, (float) places);
        //printf("%f, %f\n", arr[flattened_idx], val);
        float rounded_val = roundf(val);
        arr[flattened_idx] = rounded_val / powf(10.0f, (float) places);
    }
}

/*
__device__ unsigned int idx4_to_idx1_1(
    unsigned int a, unsigned int z, unsigned int y, unsigned int x, 
    Tuple4D shape
)
{
    //printf("%d, %d, %d, %d\n", shape.a, shape.z, shape.y, shape.x);
    //printf("%d, %d, %d, %d\n", a, z, y, x);
    unsigned int flattened 
        = (a * shape.z * shape.y * shape.x)
        + (z * shape.y * shape.x)
        + (y * shape.x)
        + x;
    
    return flattened;
}
*/

// performs gradient descent in one kernel
void gradient_desc_3d_ext(
    float lr, float l2,
    float* weight_ptr, float* weight_ptr_grad, float* weight_velocity, float* weight_momentum,
    unsigned int z0, unsigned int y0, unsigned int x0,
    float* bias_ptr, float* bias_ptr_grad, float* bias_velocity, float* bias_momentum,
    unsigned int z1, unsigned int y1, unsigned int x1,
    bool only_beta, bool non_neg, float batch_size, int optimizer_type, 
    float alpha, float beta
)
{
    KernelDim kernel_shape = get_kernel_dim1(z0, y0, x0, 1, 1, 128);
    KernelDim kernel_shape1 = get_kernel_dim1(z1, y1, x1, 1, 1, 128);

    //cudaStream_t stream0, stream1;
    //cudaStreamCreate(&stream0);
    //cudaStreamCreate(&stream1);

    if (!only_beta)
    {
        gradient_desc_3d_kernel<<<kernel_shape.grid_dim, kernel_shape.block_dim>>>
        (
            weight_ptr, weight_ptr_grad, weight_velocity, weight_momentum, lr, l2, z0, y0, x0, 
            non_neg, batch_size, optimizer_type, alpha, beta
        );
    }

    cudaDeviceSynchronize();
    
    gradient_desc_3d_kernel<<<kernel_shape1.grid_dim, kernel_shape1.block_dim>>>
    (
        bias_ptr, bias_ptr_grad, bias_velocity, bias_momentum, lr, l2, z1, y1, x1, 
        non_neg, batch_size, optimizer_type, alpha, beta
    );

    cudaDeviceSynchronize();
    //cudaStreamSynchronize(stream0);
    //cudaStreamSynchronize(stream1);
    //cudaStreamDestroy(stream0);
    //cudaStreamDestroy(stream1);
}

__global__ void gradient_desc_3d_kernel(
    float* ptr, float* ptr_grad, float* velocity, float* momentum,
    float lr, float l2, unsigned int z0, unsigned int y0, unsigned int x0, bool non_neg,
    float batch_size, int optimizer_type, float alpha, float beta
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
        int flattened_idx = (z * y0 * x0) + (y * x0) + x;

        //float v = 1e-30f * 1e-30f;
        //printf("%f\n", v);
        if (batch_size == 0.0f)
        {
            printf("Error: batch size is zero");
        }
        ptr_grad[flattened_idx] /= batch_size;


        // sgd + momentum
        if (optimizer_type == 0)
        {
            momentum[flattened_idx] = (alpha * momentum[flattened_idx]) + ((1.0f - alpha) * ptr_grad[flattened_idx]);
            ptr[flattened_idx] -= lr * (momentum[flattened_idx] + l2 * ptr[flattened_idx]);
        }

        // AdamW
        else if (optimizer_type == 1)
        {
            momentum[flattened_idx] = (alpha * momentum[flattened_idx]) + ((1.0f - alpha) * ptr_grad[flattened_idx]);
            velocity[flattened_idx] = (beta * velocity[flattened_idx]) + ((1.0f - beta) * ptr_grad[flattened_idx] * ptr_grad[flattened_idx]);
            
            ptr[flattened_idx] -= lr * ((momentum[flattened_idx] / (sqrtf(velocity[flattened_idx]) + 1e-8)) + (l2 * ptr[flattened_idx]));
        }

        ptr_grad[flattened_idx] = 0.0f;
    }
}

void broadcast_2d_to_3d_ext(
    float* broadcasted, unsigned int z0, unsigned y0, unsigned x0,
    float* original, int axis
)
{
    KernelDim kernel_shape;
    switch (axis)
    {
        case 0:
            kernel_shape = get_kernel_dim1(z0, y0, x0, 1, 1, 128);
            break;
        case 1:
            kernel_shape = get_kernel_dim1(z0, y0, x0, 1, 1, 128);
            break;
        case 2:
            kernel_shape = get_kernel_dim1(z0, y0, x0, 1, 1, 128);
            break;
    }
    //int minGridSize, blockSize;
    //cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, broadcast_2d_to_3d_kernel, 0, 0);
    //printf("%d, %d\n", minGridSize, blockSize);

    //kernel_shape = get_kernel_dim(z0 * y0 * x0);
    //printf("%u, %u, %u\n", kernel_shape.grid_dim.z, kernel_shape.grid_dim.y, kernel_shape.grid_dim.x);
    broadcast_2d_to_3d_kernel<<<kernel_shape.grid_dim, kernel_shape.block_dim>>>
    (broadcasted, z0, y0, x0, original, axis);

    cudaDeviceSynchronize();
}

__global__ void broadcast_2d_to_3d_kernel(
    float* broadcasted, unsigned int z0, unsigned y0, unsigned x0,
    float* original, int axis
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
        broadcasted[flattened_broadcast_idx] = original[flattened_original_idx];

        /*
        // apply operations
        switch (op)
        {
            case 0:
                broadcasted[flattened_broadcast_idx] += original[flattened_original_idx];
                break;
            case 1:
                broadcasted[flattened_broadcast_idx] -= original[flattened_original_idx];
                break;
            case 2:
                broadcasted[flattened_broadcast_idx] *= original[flattened_original_idx];
                break;
            case 3:
                broadcasted[flattened_broadcast_idx] /= original[flattened_original_idx];
                break;
        }
        */
    }
}

void sum_axis_ext(
    float* summed,
    float* original, unsigned int z0, unsigned y0, unsigned x0,
    int axis, bool zeroed)
{
    KernelDim kernel_shape;
    int shared_memory_len;
    unsigned int z1, y1, x1;
    switch (axis)
    {
        case 0:
            kernel_shape = get_kernel_dim1(z0, y0, x0, 32, 1, 1);
            shared_memory_len = 32;
            z1 = 1; y1 = y0; x1 = x0;
            break;
        case 1:
            kernel_shape = get_kernel_dim1(z0, y0, x0, 1, 128, 1);
            shared_memory_len = 128;
            z1 = z0; y1 = 1; x1 = x0;
            break;
        case 2:
            kernel_shape = get_kernel_dim1(z0, y0, x0, 1, 1, 128);
            shared_memory_len = 128;
            z1 = z0; y1 = y0; x1 = 1;
            break;
    }

    //printf("%u, %u, %u\n", kernel_shape.grid_dim.z, kernel_shape.grid_dim.y, kernel_shape.grid_dim.x);
    //printf("%u, %u, %u\n", kernel_shape.block_dim.z, kernel_shape.block_dim.y, kernel_shape.block_dim.x);

    //int minGridSize, blockSize;
    //cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, sum_axis_kernel, shared_memory_len * sizeof(float), 0);
    //printf("%d, %d\n", minGridSize, blockSize);
    
    //cudaEvent_t start, stop;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);

    //cudaEventRecord(start);

    //sum_axis_kernel<<<kernel_shape.grid_dim, kernel_shape.block_dim>>>
    //(summed, original, z0, y0, x0, axis, zeroed);
    //cudaDeviceSynchronize();

    if (zeroed)
    {
        zeroes_3d_ext(summed, z1, y1, x1);
    }

    sum_axis_kernel_reduction<<<kernel_shape.grid_dim, kernel_shape.block_dim, shared_memory_len * sizeof(float)>>>
    (summed, original, z0, y0, x0, axis, zeroed, shared_memory_len);

    cudaDeviceSynchronize();
    
    /*
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Kernel execution time (sum_axis): " << milliseconds / 1000.0f << " ms" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    */
    
    //int device_id;
    //cudaGetDevice(&device_id);  // Get the current device ID

    //cudaDeviceProp device_prop;
    //cudaGetDeviceProperties(&device_prop, device_id);  // Get device properties

    //printf("Device: %s\n", device_prop.name);
    //printf("Total shared memory per block: %d bytes\n", device_prop.sharedMemPerBlock);
    //exit(1);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

__global__ void sum_axis_kernel(
    float* summed,
    float* original, unsigned int z0, unsigned y0, unsigned x0,
    int axis, bool zeroed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int loop_count;
    switch (axis)
    {
        case 0:
            loop_count = z0; z0 = 1; break;
        case 1:
            loop_count = y0; y0 = 1; break;
        case 2:
            loop_count = x0; x0 = 1; break;
    }

    // limit is placed on one of the axis
    if (z < z0 && y < y0 && x < x0)
    {
        int target_idx = (z * y0 * x0) + (y * x0) + x; // always stays on axis due to limit (e.g. 1, 10, 10)

        // will reinitialize results to zeros
        if (zeroed)
        {
            summed[target_idx] = 0.0f;
        }

        int broadcasted_idx;

        for (int i = 0; i < loop_count; i++)
        {
            switch (axis)
            {
                case 0:
                    broadcasted_idx = (i * y0 * x0) + (y * x0) + x;
                    break;
                case 1:
                    broadcasted_idx = (z * y0 * x0) + (i * x0) + x;
                    break;
                case 2:
                    broadcasted_idx = (z * y0 * x0) + (y * x0) + i;
                    break;
            }

            summed[target_idx] += original[broadcasted_idx];
        }
    }
}

__global__ void sum_axis_kernel_reduction(float* summed,
    float* original, unsigned int z0, unsigned y0, unsigned x0,
    int axis, bool zeroed, int shared_memory_len)
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
            shared_block_buffer[thread_level_idx] = original[global_idx];
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

void mean_axis_ext(float* averaged,
    float* original, unsigned int z0, unsigned y0, unsigned x0,
    int axis)
{
    sum_axis_ext(averaged, original, z0, y0, x0, axis, true);

    float val;
    int new_z0, new_y0, new_x0;
    new_z0 = z0; new_y0 = y0, new_x0 = x0;
    switch (axis)
    {
        case 0:
            val = z0; new_z0 = 1;
            break;
        case 1:
            val = y0; new_y0 = 1;
            break;
        case 2:
            val = x0; new_x0 = 1;
            break;
    }

    scalar_op_3d_inplace_ext(averaged, val, 3, new_z0, new_y0, new_x0);

    //cudaDeviceSynchronize();
}