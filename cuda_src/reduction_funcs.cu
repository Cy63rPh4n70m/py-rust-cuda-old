#include "external.cuh"
#include "array_ops.cuh"
#include "reduction_funcs.cuh"
#include <stdio.h>
#include <cublas_v2.h>

void matmul_add_reduce_ext(
    float* A, float* B,
    unsigned int m, // output rows
    unsigned int n, // output cols
    unsigned int k, // intermediate col/row len
    unsigned int batch_size,
    float* C, float* D
)
{
    // grid dim: (rows, cols, ceil(k / 256))
    // block dim: (1, 1, 256)

    // create kernel dim, k will be organised along x axis
    KernelDim kernel_size = get_kernel_dim1(n, m, k, 1, 1, 32);

    //printf("%u, %u, %u, %u\n", m, n, k, batch_size);
    //printf("%u, %u, %u\n", kernel_size.grid_dim.z, kernel_size.grid_dim.y, kernel_size.grid_dim.x);
    //printf("%u, %u, %u\n", kernel_size.block_dim.z, kernel_size.block_dim.y, kernel_size.block_dim.x);

    batch_matmul_reduce<<<kernel_size.grid_dim, kernel_size.block_dim>>>
    (A, B, m, n, k, batch_size, C);
    cudaDeviceSynchronize();

    // add bias (array d)
    //element_op_3d_ext(C, D, 0, batch_size, m, n);
    //cudaDeviceSynchronize();
}

__global__ void batch_matmul_reduce(
    float* A, float* B,
    unsigned int m, unsigned int n, unsigned int k, unsigned int batch_size,
    float* C
)
{
    unsigned int global_z = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int global_x = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float A_shared[32];
    __shared__ float B_shared[32];

    unsigned int thread_level_idx = threadIdx.x;

    // blocks loop over batches using pointer arithmetic
    // blocks don't need to wait for others to finish
    // on the current batch of A and B
    for (int i = 0; i < batch_size; i++)
    {
        // shift to ith batch
        float* A_shifted = A + i * m * k;
        float* B_shifted = B + i * k * n;
        float* C_shifted = C + i * m * n;

        // global_z = n
        // global_y = m
        // global_x = k

        // global_x_idx = k
        // for a_shifted (m * k)
        // permute axes from kernel dim to A dim (assuming one batch)
        unsigned int A_shifted_flat_idx = idx4_to_idx1(
            0, 0, global_y, global_x, Tuple4D {1, 1, m, k}
        );

        // for b_shifted (k * n)
        unsigned int B_shifted_flat_idx = idx4_to_idx1(
            0, 0, global_x, global_z, Tuple4D {1, 1, k, n}
        );

        // copy elements from matrix A and B and sync block
        if (global_x < k)
        {
            A_shared[thread_level_idx] = A_shifted[A_shifted_flat_idx];
            B_shared[thread_level_idx] = B_shifted[B_shifted_flat_idx];
        }
        else
        {
            // zero padding for threads with global idx above the k length
            // and thus have invalid flattened idx
            A_shared[thread_level_idx] = 0.0f;
            B_shared[thread_level_idx] = 0.0f;
        }
        __syncthreads();

        // elementwise vector product
        A_shared[thread_level_idx] *= B_shared[thread_level_idx];
        __syncthreads();

        // parallel reduction sum
        //printf("test");
        for (int j = 32 / 2; j > 0; j /= 2)
        {
            if (thread_level_idx < j)
            {
                A_shared[thread_level_idx] += A_shared[thread_level_idx + j];
            }
            __syncthreads();
        }
        //printf("test1");

        // only one thread should atomic add
        if (thread_level_idx == 0)
        {
            // get flattened target index to write to in C (single batched)
            unsigned int C_shifted_flat_idx = idx4_to_idx1(
                0, 0, global_y, global_z, Tuple4D {1, 1, m, n}
            );

            // atomic add result 
            atomicAdd(&C_shifted[C_shifted_flat_idx], A_shared[0]);
            //printf("%f, %u\n", C_shifted[C_shifted_flat_idx], C_shifted_flat_idx);
        }

        __syncthreads();

        // block can continue to next batch without waiting for others
    }
}

// limit of up to 1,048,576 float values (1024 * 1024)
// temp_array summation axis must be a power of two
void mean_axis_reduction(
    float* original_array, unsigned int z0, unsigned int y0, unsigned int x0,
    float* temp_array, unsigned int z1, unsigned int y1, unsigned int x1,
    float* output, unsigned int z2, unsigned int y2, unsigned int x2, 
    unsigned int first_pass, unsigned int second_pass,
    int axis
)
{
    // the x axis of grid will be used to perform reduction
    KernelDim first_kernel_size, second_kernel_size;
    unsigned int global_kernel_z, global_kernel_y, global_kernel_x;
    unsigned int temp_kernel_z, temp_kernel_y, temp_kernel_x;
    unsigned int temp_z, temp_y, temp_x, temp_thread_len;

    // obtain the new axes if summation axis is to be done
    // along the x axis
    switch (axis)
    {
        case 0:
            global_kernel_z = x0; global_kernel_y = y0; global_kernel_x = z0;
            temp_kernel_z = x1; temp_kernel_y = y1; temp_kernel_x = z1;
            //temp_kernel_z = z1; temp_kernel_y = y1; temp_kernel_x = x1;
            temp_thread_len = z1;
            break;
        case 1:
            global_kernel_z = z0; global_kernel_y = x0; global_kernel_x = y0;
            temp_kernel_z = z1; temp_kernel_y = x1; temp_kernel_x = y1;
            //temp_kernel_z = z1; temp_kernel_y = y1; temp_kernel_x = x1;
            temp_thread_len = y1;
            break;
        case 2:
            global_kernel_z = z0; global_kernel_y = y0; global_kernel_x = x0;
            temp_kernel_z = z1; temp_kernel_y = y1; temp_kernel_x = x1;
            temp_thread_len = x1;
            break;
    }
    
    first_kernel_size = get_kernel_dim1(global_kernel_z, global_kernel_y, global_kernel_x, 1, 1, first_pass);
    second_kernel_size = get_kernel_dim1(temp_kernel_z, temp_kernel_y, temp_kernel_x, 1, 1, second_pass);

    //printf("grid dim: %u, %u, %u\n", first_kernel_size.grid_dim.z, first_kernel_size.grid_dim.y, first_kernel_size.grid_dim.x);
    //printf("block dim: %u, %u, %u\n", first_kernel_size.block_dim.z, first_kernel_size.block_dim.y, first_kernel_size.block_dim.x);
    //printf("grid dim: %u, %u, %u\n", second_kernel_size.grid_dim.z, second_kernel_size.grid_dim.y, second_kernel_size.grid_dim.x);
    //printf("block dim: %u, %u, %u\n", second_kernel_size.block_dim.z, second_kernel_size.block_dim.y, second_kernel_size.block_dim.x);
    //exit(1);

    zeroes_3d_ext(temp_array, z1, y1, x1);
    zeroes_3d_ext(output, z2, y2, x2);

    // two times reduction calls
    // first for reduction of main array
    sum_axis_reduce
    <<<first_kernel_size.grid_dim, first_kernel_size.block_dim, first_pass * sizeof(float)>>>
    (
        temp_array, z1, y1, x1,
        original_array, z0, y0, x0, axis
    );
    cudaDeviceSynchronize();
    
    // second reduction of temp_array into output
    sum_axis_reduce
    <<<second_kernel_size.grid_dim, second_kernel_size.block_dim, second_pass * sizeof(float)>>>
    (
        output, z2, y2, x2,
        temp_array, z1, y1, x1, axis
    );
    cudaDeviceSynchronize();
}

__global__ void sum_axis_reduce(
    float* dst_array, unsigned int dst_z, unsigned int dst_y, unsigned int dst_x,
    float* original_array, unsigned int array_z, unsigned int array_y, unsigned int array_x,
    int axis
)
{
    unsigned int global_kernel_z = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int global_kernel_y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int global_kernel_x = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int max_kernel_z = gridDim.z * blockDim.z;
    unsigned int max_kernel_y = gridDim.y * blockDim.y;
    unsigned int max_kernel_x = gridDim.x * blockDim.x;

    extern __shared__ float shared_memory[];

    // get flattened index of element to map from in original array
    // indexes can be overflowed
    // current grid dim is "transposed"
    // grid and global index need to be flipped
    unsigned int original_flat_idx, target_idx;
    unsigned int axis_dim;
    switch (axis)
    {
        case 0:
            original_flat_idx = idx4_to_idx1(
                0, global_kernel_x, global_kernel_y, global_kernel_z, 
                Tuple4D {1, array_z, array_y, array_x}
            );
            target_idx = idx4_to_idx1(
                0, blockIdx.x, global_kernel_y, global_kernel_z, 
                Tuple4D {1, dst_z, dst_y, dst_x}
            );
            axis_dim = array_z;
            break;

        case 1:
            original_flat_idx = idx4_to_idx1(
                0, global_kernel_z, global_kernel_x, global_kernel_y, 
                Tuple4D {1, array_z, array_y, array_x}
            );
            target_idx = idx4_to_idx1(
                0, global_kernel_z, blockIdx.x, global_kernel_y,
                Tuple4D {1, dst_z, dst_y, dst_x}
            );
            axis_dim = array_y;
            break;

        case 2:
            original_flat_idx = idx4_to_idx1(
                0, global_kernel_z, global_kernel_y, global_kernel_x, 
                Tuple4D {1, array_z, array_y, array_x}
            );
            target_idx = idx4_to_idx1(
                0, global_kernel_z, global_kernel_y, blockIdx.x, 
                Tuple4D {1, dst_z, dst_y, dst_x}
            );
            axis_dim = array_x;
            break;
    }

    unsigned int thread_idx = threadIdx.x;

    // copy elements to allocated shared memory
    if (global_kernel_x < axis_dim)
    {
        shared_memory[thread_idx] = original_array[original_flat_idx];
    }
    else
    {
        // zero padding if the global x index is larger than summation axis of
        // original array, and thus flat index is larger than array
        shared_memory[thread_idx] = 0.0f;
    }

    // synchronise threads in blocks
    __syncthreads();

    // parallel reduce
    int loop_count = (int) log2f((float) blockDim.x);
    int length = blockDim.x / 2;

    for (int i = 0; i < loop_count; i++)
    {
        if (thread_idx < length)
        {
            shared_memory[thread_idx] += shared_memory[thread_idx + length];
        }
        __syncthreads();
        length /= 2;
    }

    // write to the dst_array, target index depending on block index
    if (thread_idx == 0)
    {
        dst_array[target_idx] = shared_memory[0];
    }
}
