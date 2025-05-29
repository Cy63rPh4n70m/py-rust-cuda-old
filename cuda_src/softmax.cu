#include "external.cuh"
#include "array_ops.cuh"
#include "softmax.cuh"
#include <stdio.h>
#include <cfloat>

void softmax_forward_ext(
    float* input, float* exp_input, float* exp_sum,
    float* result, float* broadcast_temp, float temperature,
    unsigned int z0, unsigned y0, unsigned x0, bool zero_output
)
{
    KernelDim kernel_shape, kernel_shape1;
    int shared_memory_len;
    unsigned int z1, y1, x1;
    kernel_shape = get_kernel_dim1(z0, y0, x0, 1, 1, 128);
    shared_memory_len = 128;
    z1 = z0; y1 = y0; x1 = 1;
    kernel_shape1 = get_kernel_dim1(z1, y1, x1, 1, 1, 128);

    //float a = 10.32345f;
    //float b = 10.33456f;
    //float max = 0.0f;
    //printf("%f, %f\n", a, b);
    //printf("%i, %i\n", *reinterpret_cast<int*>(&a), *reinterpret_cast<int*>(&b));
    //exit(1);

    // static pointer to prevent reallocation
    static int* max_int_ptr = nullptr;
    static float* max_float_ptr = nullptr;
    if (max_int_ptr == nullptr)
    {
        //cudaHostAlloc((void**)&max_int_ptr, z0 * y0 * sizeof(int), cudaHostAllocMapped);
        cudaMalloc((void**)&max_int_ptr, z0 * y0 * sizeof(int));
        cudaMalloc((void**)&max_float_ptr, z0 * y0 * sizeof(float));
    }

    int min_int = -1e+8;

    fill_3d_kernel_int<<<kernel_shape1.grid_dim, kernel_shape1.block_dim>>>
    (max_int_ptr, min_int, z0, y0, 1);

    cudaDeviceSynchronize();

    //print_cuda_array_int(max_int_ptr, z0 * y0 * 1);

    float scale = 1e+6f;
    obtain_max_logit<<<kernel_shape.grid_dim, kernel_shape.block_dim>>>
    (max_int_ptr, input, scale, z0, y0, x0, true, shared_memory_len);

    int_to_float_kernel<<<kernel_shape1.grid_dim, kernel_shape1.block_dim>>>
    (max_float_ptr, max_int_ptr, scale, z0, y0, 1);

    //print_cuda_array(input, z0 * y0 * x0);
    //print_cuda_array(max_float_ptr, z0 * y0 * 1);

    cudaDeviceSynchronize();
        
    cuda_to_cuda_ext(exp_input, input, z0 * y0 * x0);
    exp_kernel<<<kernel_shape.grid_dim, kernel_shape.block_dim>>>
    (exp_input, z0, y0, x0, max_float_ptr, temperature);

    cudaDeviceSynchronize();

    // create the denominator
    sum_axis_ext(exp_sum, exp_input, z0, y0, x0, 2, true);

    cudaDeviceSynchronize();

    // divide by denominator
    cuda_to_cuda_ext(result, exp_input, z0 * y0 * x0);

    cudaDeviceSynchronize();

    broadcast_division_kernel<<<kernel_shape.grid_dim, kernel_shape.block_dim>>>
    (
        result, z0, y0, x0, exp_sum, 2
    );

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

void softmax_backward_ext(
    float* original_grads,
    float* exp_input, float* result_grads,
    float* exp_sum, float temperature,
    unsigned int z0, unsigned y0, unsigned x0, bool zero_input_grad
)
{
    KernelDim kernel_shape;
    int shared_memory_len;
    unsigned int z1, y1, x1;
    kernel_shape = get_kernel_dim1(z0, y0, x0, 1, 1, 128);
    shared_memory_len = 128;
    z1 = z0; y1 = y0; x1 = 1;

    softmax_backward_kernel<<<kernel_shape.grid_dim, kernel_shape.block_dim>>>
    (original_grads, exp_input, result_grads, exp_sum, temperature, z0, y0, x0, zero_input_grad);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

__global__ void softmax_backward_kernel(
    float* original_grads,
    float* exp_input, float* result_grads,
    float* exp_sum, float temperature,
    unsigned int z0, unsigned y0, unsigned x0, bool zero_input_grad
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (z < z0 && y < y0 && x < x0)
    {
        int result_idx = (z * y0 * x0) + (y * x0) + x;
        int exp_sum_idx = (z * y0 * 1) + (y * 1) + 0;

        // e^a * (exp_sum - e^a) / (exp_sum)^2
        if (zero_input_grad)
        {
            result_grads[result_idx] = 0.0f;
        }
        result_grads[result_idx] += original_grads[result_idx] / temperature;
        //    original_grads[result_idx] *
        //    ((exp_input[result_idx] * (exp_sum[exp_sum_idx] - exp_input[result_idx])) / 
        //        (exp_sum[exp_sum_idx] * exp_sum[exp_sum_idx]));
    }
}

__global__ void obtain_max_logit(
    int* max_value,
    float* original, float scale, unsigned int z0, unsigned y0, unsigned x0,
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

        // copy elements from array to shared block memory
        if (axis_idx < axis_dim)
        {
            shared_block_buffer[thread_level_idx] = original[global_idx];
            //printf("%d, %d, %u\n", thread_level_idx, axis_idx, axis_dim);
        }
        else
        {
            // if global index is larger, only will happen in the last block
            shared_block_buffer[thread_level_idx] = -scale;
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
                // pick the largest of the two
                if (shared_block_buffer[thread_level_idx + length] >
                        shared_block_buffer[thread_level_idx])
                {
                    //printf("%f, %f\n",shared_block_buffer[thread_level_idx], shared_block_buffer[thread_level_idx + length]);
                    shared_block_buffer[thread_level_idx] = 
                        shared_block_buffer[thread_level_idx + length];
                }
            }
            __syncthreads();

            length /= 2;
        }
        
        // atomic max to set the global max value
        if (thread_level_idx == 0)
        {
            //int float_as_int = *reinterpret_cast<int*>(&shared_block_buffer[0]);
            //printf("%i, %i\n", max_value[0], float_as_int);
            //printf("%d, %f\n", max_value[target_idx], shared_block_buffer[0]);
            atomicMax(&max_value[target_idx], (int) (shared_block_buffer[0] * scale));
            //printf("%d, %d\n", max_value[target_idx], (int) (shared_block_buffer[0] * scale));
        }
    }
}

__global__ void broadcast_division_kernel(
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
        broadcasted[flattened_broadcast_idx] /= original[flattened_original_idx];

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

__global__ void fill_3d_kernel_int(int* arr, int value, int z0, int y0, int x0)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < x0 && y < y0 && z < z0)
    {
        int flattened_idx = (z * y0 * x0) + (y * x0) + x;
        arr[flattened_idx] = value;
    }
}

__global__ void int_to_float_kernel(float* dst, int* src, float scale, int z0, int y0, int x0)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < x0 && y < y0 && z < z0)
    {
        int flattened_idx = (z * y0 * x0) + (y * x0) + x;
        dst[flattened_idx] = (float) (src[flattened_idx] / scale);
    }
}

void softmax_ce_loss_ext(float* pred, float* classes, float* loss_vals, float* grad_ptr, int z0, int y0, int x0)
{
    KernelDim kernel_shape;
    int shared_memory_len;
    unsigned int z1, y1, x1;
    kernel_shape = get_kernel_dim1(z0, y0, x0, 1, 1, 128);

    softmax_ce_loss<<<kernel_shape.grid_dim, kernel_shape.block_dim>>>
    (pred, classes, loss_vals, grad_ptr, z0, y0, x0);

    cudaDeviceSynchronize();

    //print_cuda_array(pred, (unsigned int) (z0 * y0 * x0));
    //print_cuda_array(classes, (unsigned int) (z0 * y0));
    //print_cuda_array(loss_vals, (unsigned int) (z0 * y0));
    //print_cuda_array(grad_ptr, (unsigned int) (z0 * y0 * x0));
}

__global__ void softmax_ce_loss(float* pred, float* classes, float* loss_vals, float* grad_ptr, int z0, int y0, int x0)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // loss = -log(target_pred)
    if (x < x0 && y < y0 && z < z0)
    {
        int flattened_idx = (z * y0 * x0) + (y * x0) + x;
        int flattened_loss_idx = (z * y0 * 1) + (y * 1) + 0;
        if (((int) classes[y]) == x)
        {
            loss_vals[flattened_loss_idx] = -logf(max(pred[flattened_idx], 1e-6f));
            grad_ptr[flattened_idx] = pred[flattened_idx] - 1.0f;
        }
        else
        {
            grad_ptr[flattened_idx] = pred[flattened_idx];
        }
    }
}