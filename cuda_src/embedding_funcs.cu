#include "external.cuh"
#include "array_ops.cuh"
#include "embedding_funcs.cuh"
#include <stdio.h>

void embedding_forward_ext(
    float* index_vec, unsigned int seq_len, 
    float* embedding_lookup, unsigned int vocab_size, unsigned int embedding_len,
    float* result_embedding
)
{
    KernelDim kernel_dim = get_kernel_dim1(1, seq_len, embedding_len, 1, 1, 255);

    embedding_forward<<<kernel_dim.grid_dim, kernel_dim.block_dim>>>(
        index_vec, seq_len, 
        embedding_lookup, vocab_size, embedding_len,
        result_embedding
    );

    cudaDeviceSynchronize();
}

__global__ void embedding_forward(
    float* index_vec, unsigned int seq_len, 
    float* embedding_lookup, unsigned int vocab_size, unsigned int embedding_len,
    float* result_embedding
)
{
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;

    // for each index in the index vec (index of vector, not word)
    if (z < 1 && y < seq_len && x < embedding_len)
    {
        // get flattened index
        unsigned int vector_flat_idx = idx4_to_idx1(
            0, 0, 0, y, Tuple4D {1, 1, 1, seq_len}
        );

        // get the word index
        unsigned int embedding_idx = (unsigned int) index_vec[vector_flat_idx];
        //printf("%u\n", embedding_idx);

        unsigned int lookup_idx = idx4_to_idx1(
            0, 0, embedding_idx, x, Tuple4D {1, 1, vocab_size, embedding_len}
        );

        unsigned int result_idx = idx4_to_idx1(
            0, 0, y, x, Tuple4D {1, 1, seq_len, embedding_len}
        );

        result_embedding[result_idx] = embedding_lookup[lookup_idx];
    }
}

void embedding_backward_ext(
    float* index_vec, unsigned int seq_len, 
    float* embedding_lookup, unsigned int vocab_size, unsigned int embedding_len,
    float* embedding_lookup_grads, 
    float* embedding_lookup_grads_temp, 
    float* embedding_lookup_grads_count, float* prev_grads,
    float* result_grads
)
{
    KernelDim kernel_dim = get_kernel_dim1(1, seq_len, embedding_len, 1, 1, 255);
    KernelDim kernel_dim1 = get_kernel_dim1(1, vocab_size, embedding_len, 1, 1, 255);
    
    //zeroes_3d_ext(embedding_lookup_grads, 1, vocab_size, embedding_len);
    
    embedding_backward<<<kernel_dim.grid_dim, kernel_dim.block_dim>>>(
        index_vec, seq_len, 
        embedding_lookup_grads, 
        embedding_lookup_grads_count, 
        vocab_size, embedding_len,
        prev_grads
    );

    cudaDeviceSynchronize();

    // average the accumulate gradients
    /*
    ave_embedding_grad<<<kernel_dim1.grid_dim, kernel_dim1.block_dim>>>(
        embedding_lookup_grads,
        embedding_lookup_grads_temp, 
        embedding_lookup_grads_count,
        vocab_size, embedding_len
    );
    */

    // reset count and temp gradients to zero for next pass
    zeroes_3d_ext(embedding_lookup_grads_count, 1, vocab_size, embedding_len);
    //zeroes_3d_ext(embedding_lookup_grads_temp, 1, vocab_size, embedding_len);
    zeroes_3d_ext(result_grads, 1, 1, seq_len);
}

__global__ void embedding_backward(
    float* index_vec, unsigned int seq_len, 
    float* embedding_lookup_grads_temp, // (1, vocab_len, embedding)
    float* embedding_lookup_grads_count,
    unsigned int vocab_size, unsigned int embedding_len,
    float* prev_grads // (1, seq_len, embedding)
)
{
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;

    // for each index in the index vec (index of vector, not word)
    if (z < 1 && y < seq_len && x < embedding_len)
    {
        // get flattened index
        unsigned int vector_flat_idx = idx4_to_idx1(
            0, 0, 0, y, Tuple4D {1, 1, 1, seq_len}
        );

        // get the word index
        unsigned int embedding_idx = (unsigned int) index_vec[vector_flat_idx];
        unsigned int lookup_idx = idx4_to_idx1(
            0, 0, embedding_idx, x, Tuple4D {1, 1, vocab_size, embedding_len}
        );

        // index for the previous received gradients
        unsigned int prev_grad_idx = idx4_to_idx1(
            0, 0, y, x, Tuple4D {1, 1, seq_len, embedding_len}
        );

        /*
        // for online averaging
        if (!(embedding_lookup_grads_count[lookup_idx] > 0.0f))
        {
            embedding_lookup_grads_count[lookup_idx] = 1.0f;
        }
        atomicAdd(
            &embedding_lookup_grads_temp[lookup_idx], 
            (prev_grads[prev_grad_idx] - embedding_lookup_grads_temp[lookup_idx]) / 
            embedding_lookup_grads_count[lookup_idx]
        );
        //printf("%f\n", embedding_lookup_grads_temp[lookup_idx]);

        // update the count
        atomicAdd(&embedding_lookup_grads_count[lookup_idx], 1.0f);
        */

        atomicAdd(
            &embedding_lookup_grads_temp[lookup_idx], 
            prev_grads[prev_grad_idx]
        );
    }
}

__global__ void ave_embedding_grad(
    float* embedding_lookup_grads, 
    float* embedding_lookup_grads_temp, 
    float* embedding_lookup_grads_count,
    unsigned int vocab_size, unsigned int embedding_len
)
{
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;

    if (z < 1 && y < vocab_size && x < embedding_len)
    {
        unsigned int flattened_idx = idx4_to_idx1(
            0, 0, y, x, Tuple4D {1, 1, vocab_size, embedding_len}
        );
        // average gradients
        if (embedding_lookup_grads_count[flattened_idx] > 0.0f)
        {
            //printf("1: grad: %f, count: %f\n", embedding_lookup_grads_temp[flattened_idx], embedding_lookup_grads_count[flattened_idx]);
            embedding_lookup_grads_temp[flattened_idx] /= embedding_lookup_grads_count[flattened_idx];
            //printf("2: grad: %f, count: %f\n", embedding_lookup_grads_temp[flattened_idx], embedding_lookup_grads_count[flattened_idx]);
            // accumulate to actual gradient storage
            embedding_lookup_grads[flattened_idx] += embedding_lookup_grads_temp[flattened_idx];
        }
        // reset count and temp gradient
        embedding_lookup_grads_temp[flattened_idx] = 0.0f;
        embedding_lookup_grads_count[flattened_idx] = 0.0f;
    }
}

/*
void embedding_backward_ext(
    float* index_vec, unsigned int seq_len, 
    float* embedding_lookup, unsigned int vocab_size, unsigned int embedding_len,
    float* embedding_lookup_grads
)
{
    KernelDim kernel_dim = get_kernel_dim1(1, seq_len, embedding_len, 1, 1, 255);

}
*/

/*
__global__ void embedding_update(
    float* index_vec, unsigned int seq_len, 
    float* embedding_lookup_grads, // (1, vocab_len, embedding)
    unsigned int vocab_size, unsigned int embedding_len,
    float* prev_grads, // (1, seq_len, embedding)
    float lr
)
{
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;

    // for each index in the index vec (index of vector, not word)
    if (z < 1 && y < seq_len && x < embedding_len)
    {
        // get flattened index
        unsigned int vector_flat_idx = idx4_to_idx1(
            0, 0, 0, y, Tuple4D {1, 1, 1, seq_len}
        );

        // get the word index
        unsigned int embedding_idx = (unsigned int) index_vec[vector_flat_idx];

        unsigned int lookup_idx = idx4_to_idx1(
            0, 0, embedding_idx, x, Tuple4D {1, 1, vocab_size, embedding_len}
        );

        unsigned int prev_grad_idx = idx4_to_idx1(
            0, 0, y, x, Tuple4D {1, 1, seq_len, embedding_len}
        );

        atomicAdd(&embedding_lookup_grads[lookup_idx], prev_grads[prev_grad_idx]);
    }
}
*/