#include "external.cuh"
#include "matmul_tiling.cuh"
#include "array_ops.cuh"

void matmul_add_tiling_ext(
    float* A, float* B,
    unsigned int m, // output rows
    unsigned int n, // output cols
    unsigned int k, // intermediate col/row len
    unsigned int batch_size,
    float* C, float* D
)
{
    // grid dim: (batch, rows, cols)
    // block dim: (1, 16, 16)

    // create kernel dim, k will be organised along x axis
    KernelDim kernel_size = get_kernel_dim1(batch_size, m, n, 1, TILE_LEN, TILE_LEN);

    //printf("%u, %u, %u, %u\n", m, n, k, batch_size);
    //printf("%u, %u, %u\n", kernel_size.grid_dim.z, kernel_size.grid_dim.y, kernel_size.grid_dim.x);
    //printf("%u, %u, %u\n", kernel_size.block_dim.z, kernel_size.block_dim.y, kernel_size.block_dim.x);

    batch_matmul_tiled_add<<<kernel_size.grid_dim, kernel_size.block_dim>>>
    (A, B, m, n, k, batch_size, C, D);
    cudaDeviceSynchronize();

    // add bias (array d)
    //element_op_3d_ext(C, D, 0, batch_size, m, n);
    //cudaDeviceSynchronize();
}

__global__ void batch_matmul_tiled_add(
    float* A, float* B,
    unsigned int m, 
    unsigned int n, 
    unsigned int k, 
    unsigned int batch_size,
    float* C, float* D
)
{
    unsigned int global_idx_z = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int global_idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int global_idx_x = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int thread_idx_z = threadIdx.z;
    unsigned int thread_idx_y = threadIdx.y;
    unsigned int thread_idx_x = threadIdx.x;

    unsigned int block_idx_z = threadIdx.z;
    unsigned int block_idx_y = threadIdx.y;
    unsigned int block_idx_x = threadIdx.x;

    __shared__ float tile_A[TILE_LEN][TILE_LEN];
    __shared__ float tile_B[TILE_LEN][TILE_LEN];
    //__shared__ float tile_C[TILE_LEN][TILE_LEN];

    // initialize tile_C to zeroes
    //tile_C[thread_idx_y][thread_idx_x] = 0.0f;
    float sum = 0.0f;

    // tiles loop along cols of A/rows of B
    for (int i = 0; i < (k + TILE_LEN - 1) / TILE_LEN; i++)
    {
        // copy values from A and B into shared memory
        // tile A slides across columns
        // tile B slides across rows
        unsigned int A_flat_idx = idx4_to_idx1(
            0, global_idx_z, global_idx_y, i * TILE_LEN + thread_idx_x, Tuple4D {1, batch_size, m, k}
        );
        unsigned int B_flat_idx = idx4_to_idx1(
            0, global_idx_z, i * TILE_LEN + thread_idx_y, global_idx_x, Tuple4D {1, batch_size, k, n}
        );

        // check for edge cases, where tile dimension goes pass the original 
        // dimension of matrices
        if (global_idx_y < m && i * TILE_LEN + thread_idx_x < k)
        {
            tile_A[thread_idx_y][thread_idx_x] = A[A_flat_idx];
        }
        else
        {
            tile_A[thread_idx_y][thread_idx_x] = 0.0f;
        }

        if (i * TILE_LEN + thread_idx_y < k && global_idx_x < n)
        {
            tile_B[thread_idx_y][thread_idx_x] = B[B_flat_idx];
        }
        else
        {
            tile_B[thread_idx_y][thread_idx_x] = 0.0f;
        }

        // sync tiles
        __syncthreads();

        // perform matrix multiplication on 16x16 shared arrays
        for (int j = 0; j < TILE_LEN; j++)
        {
            sum += tile_A[thread_idx_y][j] * tile_B[j][thread_idx_x];
        }

        // sync tile
        __syncthreads();
    }

    // add tile_C which contain the final results
    // to C

    unsigned int C_flat_idx = idx4_to_idx1(
        0, global_idx_z, global_idx_y, global_idx_x, Tuple4D {1, batch_size, m, n}
    );

    // check for case where tile exceeds m or n
    if (global_idx_y < m && global_idx_x < n)
    {
        // add D matrix (biases)
        C[C_flat_idx] = sum + D[C_flat_idx];//tile_C[thread_idx_y][thread_idx_x] + D[C_flat_idx];
    }
    else
    {
        // ignore as its out of bounds of the result matrix
    }
}