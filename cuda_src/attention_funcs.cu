#include "external.cuh"
#include "array_ops.cuh"
#include "attention_funcs.cuh"
#include "matmul_tiling.cuh"
#include <iostream>

void kqv_forward_ext(
    float* input_ptr, unsigned int z0, unsigned int y0, unsigned int x0,
    float* k_weight_ptr, float* k_bias_ptr, unsigned int z1, unsigned int y1, unsigned int x1, 
    float* q_weight_ptr, float* q_bias_ptr, 
    float* v_weight_ptr, float* v_bias_ptr,
    float* k_result_ptr, float* q_result_ptr, float* v_result_ptr
)
{
    KernelDim kq_kernel_size = get_kernel_dim1(z0, y0, x1, 1, TILE_LEN, TILE_LEN); // (n_heads, seq_len, new_embed)
    KernelDim v_kernel_size = get_kernel_dim1(z0, y0, x0, 1, TILE_LEN, TILE_LEN); // (n_heads, seq_len, original_embed)

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    
    static cudaStream_t stream_k, stream_q, stream_v;
    if (stream_k == nullptr)
    {
        cudaStreamCreate(&stream_k);
        cudaStreamCreate(&stream_q);
        cudaStreamCreate(&stream_v);
    }
    

    // matrix multiplication for k matrix
    batch_matmul_tiled_add<<<kq_kernel_size.grid_dim, kq_kernel_size.block_dim, 0, stream_k>>>(
        input_ptr, k_weight_ptr, 
        y0, x1, x0, z0, 
        k_result_ptr,
        k_bias_ptr
    );

    // matrix multiplication for q matrix
    batch_matmul_tiled_add<<<kq_kernel_size.grid_dim, kq_kernel_size.block_dim, 0, stream_q>>>(
        input_ptr, q_weight_ptr, 
        y1, x1, x0, z0,
        q_result_ptr,
        q_bias_ptr
    );

    // matrix multiplication for v matrix
    batch_matmul_tiled_add<<<v_kernel_size.grid_dim, v_kernel_size.block_dim, 0, stream_v>>>(
        input_ptr, v_weight_ptr,
        y0, x0, x0, z0,
        v_result_ptr,
        v_bias_ptr
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Kernel execution time (kqv_matmul): " << milliseconds / 1000.0f << " ms" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaStreamSynchronize(stream_k);
    cudaStreamSynchronize(stream_q);
    cudaStreamSynchronize(stream_v);
}

// all shared same inputs
void kqv_backward_ext(
    
    float* input_tensor,
    float* input_grads, unsigned int z0, unsigned int y0, unsigned int x0, // (3, 2, 5, 1)
    float* k_weight_grads, unsigned int z1, unsigned int y1, unsigned int x1, // (3, 1, 5, 4)
    float* k_bias_grads, unsigned int z2, unsigned int y2, unsigned int x2, // (3, 2, 5, 4) -> (3, 2, 4)
    float* k_original_grads, // (3, 2, 4)
    float* k_weight_tensor,

    float* q_weight_grads, 
    float* q_bias_grads,
    float* q_original_grads, // (3, 2, 4)
    float* q_weight_tensor,

    float* v_weight_grads, 
    float* v_bias_grads,
    float* v_original_grads, // (3, 2, 4)
    float* v_weight_tensor
)
{
    KernelDim kq_kernel_size, kq_kernel_size1, kq_kernel_size2;
    KernelDim v_kernel_size, v_kernel_size1, v_kernel_size2;

    static cudaStream_t stream0_k, stream1_k, stream2_k;
    static cudaStream_t stream0_q, stream1_q, stream2_q;
    static cudaStream_t stream0_v, stream1_v, stream2_v;

    if (stream0_k == nullptr)
    {
        cudaStreamCreate(&stream0_k);
        cudaStreamCreate(&stream1_k);
        cudaStreamCreate(&stream2_k);

        cudaStreamCreate(&stream0_q);
        cudaStreamCreate(&stream1_q);
        cudaStreamCreate(&stream2_q);

        cudaStreamCreate(&stream0_v);
        cudaStreamCreate(&stream1_v);
        cudaStreamCreate(&stream2_v);
    }

    kq_kernel_size = get_kernel_dim1(z1, y1, x1, 3, 16, 16); // weight grads
    kq_kernel_size1 = get_kernel_dim1(z0, y0, x0, 3, 16, 16); // input grads
    kq_kernel_size2 = get_kernel_dim1(z2, y2, x2, 3, 16, 16); // bias grads

    v_kernel_size = get_kernel_dim1(z1, x0, x0, 3, 16, 16); // weight grads
    v_kernel_size1 = get_kernel_dim1(z0, y0, x0, 3, 16, 16); // input grads
    v_kernel_size2 = get_kernel_dim1(z2, y0, x0, 3, 16, 16); // bias grads

    // initialize to zero
    zeroes_3d_ext(input_grads, z0, y0, x0);

    // k matrix
    calc_weight_grads_kernel
    <<<kq_kernel_size.grid_dim, kq_kernel_size.block_dim, 0, stream0_k>>>
    (
        k_weight_grads, z1, y1, x1,
        k_bias_grads, z2, y2, x2,
        k_original_grads,
        input_tensor, z0, y0, x0, false
    );

    calc_input_grads_kernel
    <<<kq_kernel_size1.grid_dim, kq_kernel_size1.block_dim, 0, stream1_k>>>
    (
        input_grads, z0, y0, x0,
        k_weight_tensor, z1, y1, x1,
        k_original_grads, z2, y2, x2, true
    );

    calc_bias_grads_kernel
    <<<kq_kernel_size2.grid_dim, kq_kernel_size2.block_dim, 0, stream2_k>>>
    (
        k_bias_grads, z2, y2, x2, k_original_grads
    );




    // q matrix
    calc_weight_grads_kernel
    <<<kq_kernel_size.grid_dim, kq_kernel_size.block_dim, 0, stream0_q>>>
    (
        q_weight_grads, z1, y1, x1,
        q_bias_grads, z2, y2, x2,
        q_original_grads,
        input_tensor, z0, y0, x0, false
    );

    calc_input_grads_kernel
    <<<kq_kernel_size1.grid_dim, kq_kernel_size1.block_dim, 0, stream1_q>>>
    (
        input_grads, z0, y0, x0,
        q_weight_tensor, z1, y1, x1,
        q_original_grads, z2, y2, x2, true
    );

    calc_bias_grads_kernel
    <<<kq_kernel_size2.grid_dim, kq_kernel_size2.block_dim, 0, stream2_q>>>
    (
        q_bias_grads, z2, y2, x2, q_original_grads
    );




    // v matrix
    calc_weight_grads_kernel
    <<<v_kernel_size.grid_dim, v_kernel_size.block_dim, 0, stream0_v>>>
    (
        v_weight_grads, z1, x0, x0,
        v_bias_grads, z2, y0, x0,
        v_original_grads,
        input_tensor, z0, y0, x0, false
    );

    calc_input_grads_kernel
    <<<v_kernel_size1.grid_dim, v_kernel_size1.block_dim, 0, stream1_v>>>
    (
        input_grads, z0, y0, x0,
        v_weight_tensor, z1, x0, x0,
        v_original_grads, z0, y0, x0, true
    );

    calc_bias_grads_kernel
    <<<v_kernel_size2.grid_dim, v_kernel_size2.block_dim, 0, stream2_v>>>
    (
        v_bias_grads, z0, y0, x0, v_original_grads
    );
    
    //cudaDeviceSynchronize();

    cudaStreamSynchronize(stream0_k);
    cudaStreamSynchronize(stream1_k);
    cudaStreamSynchronize(stream2_k);

    cudaStreamSynchronize(stream0_q);
    cudaStreamSynchronize(stream1_q);
    cudaStreamSynchronize(stream2_q);

    cudaStreamSynchronize(stream0_v);
    cudaStreamSynchronize(stream1_v);
    cudaStreamSynchronize(stream2_v);

    //cudaStreamDestroy(stream0_k);
    //cudaStreamDestroy(stream1_k);
    //cudaStreamDestroy(stream2_k);

    //cudaStreamDestroy(stream0_q);
    //cudaStreamDestroy(stream1_q);
    //cudaStreamDestroy(stream2_q);

    //cudaStreamDestroy(stream0_v);
    //cudaStreamDestroy(stream1_v);
    //cudaStreamDestroy(stream2_v);
}

void kqv_update_params_ext(
    float lr,
    float* k_weight, float* k_weight_grads, float* k_weight_vel, unsigned int z0, unsigned int y0, unsigned int x0,
    float* k_biases, float* k_biases_grads, float* k_bias_vel, unsigned int z1, unsigned int y1, unsigned int x1,
    float* q_weight, float* q_weight_grads, float* q_weight_vel, 
    float* q_biases, float* q_biases_grads, float* q_bias_vel, 
    float* v_weight, float* v_weight_grads, float* v_weight_vel, unsigned int z2, unsigned int y2, unsigned int x2,
    float* v_biases, float* v_biases_grads, float* v_bias_vel, unsigned int z3, unsigned int y3, unsigned int x3
)
{
    KernelDim kq_kernel_shape_weights = get_kernel_dim1(z0, y0, x0, 3, 16, 16);
    KernelDim kq_kernel_shape_biases = get_kernel_dim1(z1, y1, x1, 3, 16, 16);

    KernelDim v_kernel_shape_weights = get_kernel_dim1(z2, y2, x2, 3, 16, 16);
    KernelDim v_kernel_shape_biases = get_kernel_dim1(z3, y3, x3, 3, 16, 16);

    static cudaStream_t stream0_k, stream1_k;
    static cudaStream_t stream0_q, stream1_q;
    static cudaStream_t stream0_v, stream1_v;
    
    if (stream0_k == nullptr)
    {
        cudaStreamCreate(&stream0_k);
        cudaStreamCreate(&stream1_k);
        cudaStreamCreate(&stream0_q);
        cudaStreamCreate(&stream1_q);
        cudaStreamCreate(&stream0_v);
        cudaStreamCreate(&stream1_v);
    }

    gradient_desc_3d_kernel<<<kq_kernel_shape_weights.grid_dim, kq_kernel_shape_weights.block_dim, 0, stream0_k>>>
    (k_weight, k_weight_grads, k_weight_vel, lr, 0.0f, z0, y0, x0, false, 1.0f, 0, 0.0f, 0.0f, 10.0f);
    gradient_desc_3d_kernel<<<kq_kernel_shape_biases.grid_dim, kq_kernel_shape_weights.block_dim, 0, stream1_k>>>
    (k_biases, k_biases_grads, k_bias_vel, lr, 0.0f, z1, y1, x1, false, 1.0f, 0, 0.0f, 0.0f, 10.0f);

    gradient_desc_3d_kernel<<<kq_kernel_shape_weights.grid_dim, kq_kernel_shape_weights.block_dim, 0, stream0_q>>>
    (q_weight, q_weight_grads, q_weight_vel,  lr, 0.0f, z0, y0, x0, false, 1.0f, 0, 0.0f, 0.0f, 10.0f);
    gradient_desc_3d_kernel<<<kq_kernel_shape_biases.grid_dim, kq_kernel_shape_weights.block_dim, 0, stream1_q>>>
    (q_biases, q_biases_grads, q_bias_vel,  lr, 0.0f, z1, y1, x1, false, 1.0f, 0, 0.0f, 0.0f, 10.0f);

    gradient_desc_3d_kernel<<<v_kernel_shape_weights.grid_dim, v_kernel_shape_weights.block_dim, 0, stream0_v>>>
    (v_weight, v_weight_grads, v_weight_vel, lr, 0.0f, z2, y2, x2, false, 1.0f, 0, 0.0f, 0.0f, 10.0f);
    gradient_desc_3d_kernel<<<v_kernel_shape_biases.grid_dim, v_kernel_shape_weights.block_dim, 0, stream1_v>>>
    (v_biases, v_biases_grads, v_bias_vel, lr, 0.0f, z3, y3, x3, false, 1.0f, 0, 0.0f, 0.0f, 10.0f);

    cudaStreamSynchronize(stream0_k);
    cudaStreamSynchronize(stream1_k);
    cudaStreamSynchronize(stream0_q);
    cudaStreamSynchronize(stream1_q);
    cudaStreamSynchronize(stream0_v);
    cudaStreamSynchronize(stream1_v);
    //cudaStreamDestroy(stream0_k);
    //cudaStreamDestroy(stream1_k);
    //cudaStreamDestroy(stream0_q);
    //cudaStreamDestroy(stream1_q);
    //cudaStreamDestroy(stream0_v);
    //cudaStreamDestroy(stream1_v);
}