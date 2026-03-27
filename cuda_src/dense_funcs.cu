#include "dense_funcs.cuh"
#include "external.cuh"
#include "array_ops.cuh"
#include "reduction_funcs.cuh"
#include "matmul_tiling.cuh"
#include <cuda_fp16.h>
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// assumes ALL matrices are already on the GPU
void matmul_bias_ext(
    float* mat0, unsigned int z0, unsigned int y0, unsigned int x0,
    float* mat1, unsigned int z1, unsigned int y1, unsigned int x1,
    float* result_mat, float* mat2
)
{   
    KernelDim kernel_size = get_kernel_dim1(z0, y0, x1, 1, 1, 128);

    //cudaEvent_t start, stop;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);

    //cudaEventRecord(start);
    
    matmul_bias_kernel<<<kernel_size.grid_dim, kernel_size.block_dim>>>(
        mat0, z0, y0, x0,
        mat1, z1, y1, x1,
        result_mat, z0, y0, x1,
        mat2
    );

    cudaDeviceSynchronize();
    
    /*
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Kernel execution time (matmul): " << milliseconds / 1000.0f << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    */
    
    /*
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        std::cout << "Device " << device << ": " << deviceProp.name << "\n";
        std::cout << "Max number of grids (x, y, z): "
                  << deviceProp.maxGridSize[0] << " x "
                  << deviceProp.maxGridSize[1] << " x "
                  << deviceProp.maxGridSize[2] << "\n";
    }
    exit(1);
    */
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error (matmul_bias): %s\n", cudaGetErrorString(err));
    }
}

void matmul_bias_tiled_ext(
    float* mat0, unsigned int z0, unsigned int y0, unsigned int x0,
    float* mat1, unsigned int z1, unsigned int y1, unsigned int x1,
    float* result_mat, float* mat2, bool use_bias, bool zero_output
)
{   
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr)
    {
        cublasCreate(&handle);
        cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    }

    float alpha = 1.0f;
    float beta = 1.0f;

    if (use_bias && zero_output)
    {
        cudaMemcpy(result_mat, mat2, sizeof(float) * z0 * y0 * x1, cudaMemcpyDeviceToDevice);
    }
    else
    {
        //zeroes_3d_ext(result_mat, z0, y0, x1);
        beta = 0.0f;
    }

    //printf("%f, %f, %f\n", result_mat);

    cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        x1, y0, x0,
        &alpha, 
        mat1, x1, y1 * x1,
        mat0, x0, y0 * x0,
        &beta, 
        result_mat, x1, y0 * x1,
        z0
    );

    if (use_bias && !zero_output)
    {
        element_op_3d_ext(result_mat, mat2, 0, z0, y0, x1);
    }
    /**/

    /*
    matmul_add_tiling_ext(
        mat0, mat1,
        y0, x1, x0, z0,
        result_mat, mat2
    );
    */

    //cudaDeviceSynchronize();

    //element_op_3d_ext(result_mat, mat2, 0, z0, y0, x1);
    cudaDeviceSynchronize();
    
    //cudaEventRecord(stop);
    //cudaEventSynchronize(stop);
    //float milliseconds = 0;
    //cudaEventElapsedTime(&milliseconds, start, stop);

    //std::cout << "Kernel execution time (matmul_tiled): " << milliseconds / 1000.0f << " ms" << std::endl;

    //cudaEventDestroy(start);
    //cudaEventDestroy(stop);
    
    /*
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        std::cout << "Device " << device << ": " << deviceProp.name << "\n";
        std::cout << "Max number of grids (x, y, z): "
                  << deviceProp.maxGridSize[0] << " x "
                  << deviceProp.maxGridSize[1] << " x "
                  << deviceProp.maxGridSize[2] << "\n";
    }
    exit(1);
    */
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error (matmul_bias_tiled): %s\n", cudaGetErrorString(err));
        printf("input_mat_dim: %u, %u, %u\n", z0, y0, x0);
        printf("weight_mat_dim: %u, %u, %u\n", z1, y1, x1);
    }
}

void matmul_bias_back_ext(
    float* input_grads, unsigned int z0, unsigned int y0, unsigned int x0, // (3, 2, 5, 1)
    float* weight_grads, unsigned int z1, unsigned int y1, unsigned int x1, // (3, 1, 5, 4)
    float* bias_grads, unsigned int z2, unsigned int y2, unsigned int x2, // (3, 2, 5, 4) -> (3, 2, 4)
    float* original_grads, // (3, 2, 4)
    float* input_tensor,// (3, 2, 5, 1)
    float* weight_tensor,
    bool use_bias,
    bool zero_input_grad,
    bool zero_weight_grad

    //float* input_tensor_t,
    //float* weight_tensor_t,
)
{
    KernelDim kernel_size, kernel_size1, kernel_size2;
    // define enough threads that can accomodate the size for transposing
    // inputs and weights
    /*
    float largest_prod;
    if (z0 * y0 * x0 >= z1 * y1 * x1)
    {
        largest_prod = z0 * y0 * x0;
    }
    else
    {
        largest_prod = z1 * y1 * x1;
    }
    kernel_size = get_kernel_dim(largest_prod);

    // transpose the input and weights
    transpose_inputs_and_weights_kernel
    <<<kernel_size.grid_dim, kernel_size.block_dim>>>
    (
        input_tensor_t, input_tensor, z0, y0, x0,
        weight_tensor_t, weight_tensor, z1, y1, x1
    );
    */
    
    /*
    // define enough threads that can accomodate the size for matmul
    kernel_size = get_kernel_dim(z0 * y0 * x0 * x1);
    //printf("kernel size grid: %u, %u, %u\n", kernel_size.grid_dim.x, kernel_size.grid_dim.y, kernel_size.grid_dim.z);
    //printf("kernel size block: %u, %u, %u\n", kernel_size.block_dim.x, kernel_size.block_dim.y, kernel_size.block_dim.z);
    // calculate gradients for input, weights, biases
    //print_cuda_array(input_tensor, z0 * y0 * x0);
    matmul_bias_back_kernel<<<kernel_size.grid_dim, kernel_size.block_dim>>>(
        input_grads, z0, y0, x0, // (3, 2, 5, 1)
        weight_grads, z1, y1, x1, // (3, 1, 5, 4)
        bias_grads, z2, y2, x2, // (3, 2, 5, 4) -> (3, 2, 4)
        original_grads,
        input_tensor, 
        weight_tensor
    );
    /**/

    //printf("ended\n");
    /**/

    //cudaStream_t stream0, stream1, stream2;
    //cudaStreamCreate(&stream0);
    //cudaStreamCreate(&stream1);
    //cudaStreamCreate(&stream2);
    //kernel_size = get_kernel_dim1(z1, y1, x1, 1, 1, 128);
    //kernel_size1 = get_kernel_dim1(z0, y0, x0, 1, 1, 128);
    //kernel_size2 = get_kernel_dim1(z2, y2, x2, 1, 1, 128);
    //kernel_size = get_kernel_dim(z1 * y1 * x1);
    //printf("calc_weight_grad: kernel size grid: %u, %u, %u\n", kernel_size.grid_dim.x, kernel_size.grid_dim.y, kernel_size.grid_dim.z);
    //printf("calc_weight_grad: kernel size block: %u, %u, %u\n", kernel_size.block_dim.x, kernel_size.block_dim.y, kernel_size.block_dim.z);
    
    static cublasHandle_t handle0 = nullptr;
    static cublasHandle_t handle1 = nullptr;
    static cudaStream_t stream0 = nullptr;
    static cudaStream_t stream1 = nullptr;
    if (handle0 == nullptr)
    {
        cublasCreate(&handle0);
        cublasCreate(&handle1);
        cublasSetMathMode(handle0, CUBLAS_TF32_TENSOR_OP_MATH);
        cublasSetMathMode(handle1, CUBLAS_TF32_TENSOR_OP_MATH);
        cudaStreamCreate(&stream0);
        cudaStreamCreate(&stream1);
        cublasSetStream(handle0, stream0);
        cublasSetStream(handle1, stream1);
    }

    float alpha = 1.0f;
    float beta = 1.0f, weight_grad_beta = 1.0f;

    if (zero_input_grad)
    {
        beta = 0.0f;
    }

    if (zero_weight_grad)
    {
        weight_grad_beta = 0.0f;
    }

    /*
    calc_weight_grads_kernel
    <<<kernel_size.grid_dim, kernel_size.block_dim>>>
    (
        weight_grads, z1, y1, x1,
        bias_grads, z2, y2, x2,
        original_grads,
        input_tensor, z0, y0, x0, false
    );
    */

    //matmul(
    //        input_tensor, z0, y0, x0, true,
    //        original_grads, z2, y2, x2, false,
    //        y0,
    //        weight_grads, z1, y1, x1,
    //        z, y, x, add
    //    );

    /**/
    cublasSgemmStridedBatched(
        handle0,
        CUBLAS_OP_N, CUBLAS_OP_T,
        x1, y1, y0,
        &alpha, 
        original_grads, x2, y2 * x2,
        input_tensor, x0, y0 * x0,
        &weight_grad_beta, 
        weight_grads, x1, y1 * x1,
        z0
    );

    /**/

    // average gradient summation for weights (divide by the batch len assuming rows of vectors)
    //scalar_op_3d_inplace_ext(
    //    weight_grads, (float) y0, 3, z1, y1, x1
    //);

    //kernel_size = get_kernel_dim(z1 * y1 * x1);
    //printf("calc_input_grad: kernel size grid: %u, %u, %u\n", kernel_size.grid_dim.x, kernel_size.grid_dim.y, kernel_size.grid_dim.z);
    //printf("calc_input_grad: kernel size block: %u, %u, %u\n", kernel_size.block_dim.x, kernel_size.block_dim.y, kernel_size.block_dim.z);
    /*
    calc_input_grads_kernel
    <<<kernel_size1.grid_dim, kernel_size1.block_dim>>>
    (
        input_grads, z0, y0, x0,
        weight_tensor, z1, y1, x1,
        original_grads, z2, y2, x2, false
    );
    */

     //matmul(
     //       original_grads, z2, y2, x2, false,
     //       weight_tensor, z1, y1, x1, true,
     //       x2,
     //       input_grads, z0, y0, x0,
     //       z, y, x, add
     //   );

    /**/
    cublasSgemmStridedBatched(
        handle1,
        CUBLAS_OP_T, CUBLAS_OP_N,
        x0, y0, x2,
        &alpha, 
        weight_tensor, x1, y1 * x1,
        original_grads, x2, y2 * x2,
        &beta, 
        input_grads, x0, y0 * x0,
        z0
    );

    cudaDeviceSynchronize();
    /**/

    //calc_bias_grads_kernel
    //<<<kernel_size2.grid_dim, kernel_size2.block_dim>>>
    //(
    //    bias_grads, z2, y2, x2, original_grads
    //);

    // bias gradients
    if (use_bias)
    {
        element_op_3d_ext(bias_grads, original_grads, 0, z2, y2, x2);
    }

    cudaDeviceSynchronize();

    //cudaStreamSynchronize(stream0);
    //cudaStreamSynchronize(stream1);
    //cudaStreamSynchronize(stream2);
    //cudaStreamDestroy(stream0);
    //cudaStreamDestroy(stream1);
    //cudaStreamDestroy(stream2);
    /**/

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error (matmul_bias_back): %s\n", cudaGetErrorString(err));
    }
}

// use for loopings
__global__ void matmul_bias_kernel(
    float* mat0, unsigned int z0, unsigned int y0, unsigned int x0, // (3, 2, 5)
    float* mat1, unsigned int z1, unsigned int y1, unsigned int x1, // (3, 5, 4), actually (3, 1, 5, 4) for compatibility with
    //float* broadcast_buffer,// broadcast buffer shape = (3, 2, 5, 4)
    float* result_mat, unsigned int z2, unsigned int y2, unsigned int x2, // (3, 2, 4)
    float* biases
)
{
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // get flattened index based on the kernel shape
    /*
    unsigned int flattened_idx = idx4_to_idx1(
        0, z, y, x, 
        Tuple4D {
            1,
            gridDim.z * blockDim.z,
            gridDim.y * blockDim.y,
            gridDim.x * blockDim.x
        }
    );
    */
    //printf("result: %u, %u, %u\n", z2, y2, x2);

    //if (flattened_idx < z2 * y2 * x2)
    if (z < z2 && y < y2 && x < x2)
    {
        // get the 3d index based on the TARGET shape
        //Tuple4D index3 = idx1_to_idx4(flattened_idx, Tuple4D {1, z2, y2, x2});

        // matrix multiplication
        matmul(
            mat0, z0, y0, x0, false,
            mat1, z1, y1, x1, false,
            x0,
            result_mat, z2, y2, x2,
            z, y, x, false
        );

        // add bias
        add_biases(result_mat, biases, z, y, x, z2, y2, x2);
    }
}

// performs all gradient calculations in same kernel
__global__ void matmul_bias_back_kernel(
    float* input_grads, unsigned int z0, unsigned int y0, unsigned int x0, // (3, 2, 5, 1)
    float* weight_grads, unsigned int z1, unsigned int y1, unsigned int x1, // (3, 1, 5, 4)
    float* bias_grads, unsigned int z2, unsigned int y2, unsigned int x2, // (3, 2, 5, 4) -> (3, 2, 4)
    float* original_grads, // (3, 2, 4)
    float* input_tensor, // (3, 2, 5, 1)
    float* weight_tensor // (3, 1, 5, 4)
)
{
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // get flattened index based on the kernel shape
    /*
    unsigned int flattened_idx = idx4_to_idx1(
        0, z, y, x, 
        Tuple4D {
            1,
            gridDim.z * blockDim.z,
            gridDim.y * blockDim.y,
            gridDim.x * blockDim.x
        }
    );
    */
    //printf("%u, %u, %u\n", gridDim.z * blockDim.z, gridDim.y * blockDim.y, gridDim.x * blockDim.x);
    //printf("%u\n", z2 * y2 * x2);

    // calculate bias gradients (same as original gradients)
    //if (flattened_idx < z2 * y2 * x2)
    if (z < z2 && y < y2 && x < x2)
    {
        unsigned int flattened_idx = z * y2 * x2 + y * x2 + x;
        bias_grads[flattened_idx] = original_grads[flattened_idx];
    }

    // calculate weight gradients (transposed input matmul with original grads)
    //if (flattened_idx < z1 * y1 * x1)
    if (z < z1 && y < y1 && x < x1)
    {
        unsigned int flattened_idx = z * y1 * x1 + y * x1 + x;
        // get the 3d index based on the TARGET shape
        //Tuple4D index3 = idx1_to_idx4(flattened_idx, Tuple4D {1, z1, y1, x1});
        //printf("%u, %u, %u\n", z1, y1, x1);
        //printf("%u, %u, %u\n", z, y, x);

        matmul(
            input_tensor, z0, y0, x0, true,
            original_grads, z2, y2, x2, false,
            y0,
            weight_grads, z1, y1, x1,
            z, y, x, false
        );
    }

    // calculate input gradients (original grads matmul with transposed weights)
    //if (flattened_idx < z0 * y0 * x0)
    if (z < z0 && y < y0 && x < x0)
    {
        unsigned int flattened_idx = z * y0 * x0 + y * x0 + x;
        //Tuple4D index3 = idx1_to_idx4(flattened_idx, Tuple4D {1, z0, y0, x0});
        matmul(
            original_grads, z2, y2, x2, false,
            weight_tensor, z1, y1, x1, true,
            x2,
            input_grads, z0, y0, x0,
            z, y, x, false
        );
    }
}

void transpose_2d_ext(float* arr_t, float* arr, unsigned int z0, unsigned int y0, unsigned int x0)
{
    KernelDim kernel_shape = get_kernel_dim1(z0, y0, x0, 1, 1, 128);
    //print_cuda_array(arr, z0 * y0 * x0);
    //printf("\n, %u, %u, %u\n", kernel_shape.grid_dim.x, kernel_shape.grid_dim.y, kernel_shape.grid_dim.z);
    transpose_2d_kernel<<<kernel_shape.grid_dim, kernel_shape.block_dim>>>
    (arr_t, arr, z0, y0, x0);

    cudaDeviceSynchronize();
}

__global__ void transpose_2d_kernel(
    float* arr_t, float* arr, unsigned int z0, unsigned int y0, unsigned int x0
)
{
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (z < z0 && y < y0 && x < x0)
    {
        unsigned int index = idx4_to_idx1(0, z, y, x, Tuple4D {1, z0, y0, x0});
        unsigned int index_t = idx4_to_idx1(0, z, x, y, Tuple4D {1, z0, x0, y0});
        arr_t[index_t] = arr[index];
        
        //printf("%u, %u, %u | %u, %u, %u | %f, %f, %u, %u\n", z0, y0, x0, z, y, x, arr_t[index_t], arr[index], index_t, index);
    }
}

__device__ void transpose_2d(
    float* arr_t, float* arr, 
    unsigned int z, unsigned int y, unsigned int x,
    unsigned int z0, unsigned int y0, unsigned int x0
)
{
    // transpose is reversed indexes
    unsigned int arr_idx = idx4_to_idx1(0, z, y, x, Tuple4D {1, z0, y0, x0});
    unsigned int arr_t_idx = idx4_to_idx1(0, z, x, y, Tuple4D {1, z0, x0, y0});
    arr_t[arr_t_idx] = arr[arr_idx];
}

// matmul performed in (A x B) * (B x C) = (A x C)
// if (B x A) * (B x C), required to specify transpose
__device__ void matmul(
    float* mat0, unsigned int z0, unsigned int y0, unsigned int x0, bool transpose_mat0,// (3, 2, 5)
    float* mat1, unsigned int z1, unsigned int y1, unsigned int x1, bool transpose_mat1,// (3, 5, 4), actually (3, 1, 5, 4) for compatibility with
    //float* broadcast_buffer,// broadcast buffer shape = (3, 2, 5, 4)
    unsigned int val,
    float* result_mat, unsigned int z2, unsigned int y2, unsigned int x2, // (3, 2, 4)
    unsigned int z, unsigned int y, unsigned int x, // 3d index of result matrix, within result dimensions
    bool add
)
{
    float dot_prod = 0.0f;
    unsigned int mat0_flat_idx, mat1_flat_idx;

    // iterate sum along third axis of mat0, (3, 2, 5) * (3, 5, 4)
    //printf("%u\n", val);
    //printf("mat0_dim: %u, %u, %u, mat0_axis: %d\n", z0, y0, x0, mat0_axis);
    //printf("%u\n", val);
    //printf("mat1_dim: %u, %u, %u, mat1_axis %d\n", z1, y1, x1, mat1_axis);
    for (int i = 0; i < val; i++)
    {
        // iterate along specified axis
        
        switch (transpose_mat0)
        {
            case true:

                // mat0 needs to be transposed to be compatible for matrix multiplication
                // actually perform reverse transposing of index
                // reverse mapping given the proper transposed index
                mat0_flat_idx = idx4_to_idx1(0, z, i, y, Tuple4D {1, z0, y0, x0});
                //printf("mat0_axis 1: %u, %u, %u\n", z, y, i);
                break;
            case false:
                mat0_flat_idx = idx4_to_idx1(0, z, y, i, Tuple4D {1, z0, y0, x0});
                break;
        }

        // iterate along specified axis
        switch (transpose_mat1)
        {
            case true:
                mat1_flat_idx = idx4_to_idx1(0, z, x, i, Tuple4D {1, z1, y1, x1});
                //printf("mat1_axis 0: %u, %u, %u\n", z, i, x);
                break;
            case false:
                mat1_flat_idx = idx4_to_idx1(0, z, i, x, Tuple4D {1, z1, y1, x1});
                break;
        }

        dot_prod += mat0[mat0_flat_idx] * mat1[mat1_flat_idx];
    }

    // assign sum to result index
    unsigned int result_mat_flat_idx = idx4_to_idx1(0, z, y, x, Tuple4D {1, z2, y2, x2});
    //printf("%f, %f\n", result_mat[result_mat_flat_idx], dot_prod);

    if (add)
    {
        atomicAdd(&result_mat[result_mat_flat_idx], dot_prod);
    }
    else
    {
        result_mat[result_mat_flat_idx] = dot_prod;
    }
}

__device__ void add_biases(
    float* result_mat, float* biases, 
    unsigned int z, unsigned int y, unsigned int x,
    unsigned int z2, unsigned int y2, unsigned int x2
)
{
    unsigned int flat_idx = idx4_to_idx1(0, z, y, x, Tuple4D {1, z2, y2, x2});
    result_mat[flat_idx] += biases[flat_idx];
}

__device__ Tuple4D idx1_to_idx4(unsigned int flattened_idx, Tuple4D shape)
{
    unsigned int s0 = shape.z * shape.y * shape.x;
    unsigned int s1 = shape.y * shape.x;
    unsigned int s2 = shape.x;
    unsigned int s3 = 1;

    unsigned int a = flattened_idx / s0;
    unsigned int z = (flattened_idx % s0) / s1;
    unsigned int y = (flattened_idx % s1) / s2;
    unsigned int x = flattened_idx % s2;

    return Tuple4D {a, z, y, x};
}

__device__ unsigned int idx4_to_idx1(
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

__global__ void calc_input_grads_kernel(
    float* input_grads, unsigned int z0, unsigned int y0, unsigned int x0, // (3, 2, 5, 1)
    float* weight_tensor, unsigned int z1, unsigned int y1, unsigned int x1, // (3, 1, 5, 4)
    float* original_grads, unsigned int z2, unsigned int y2, unsigned int x2, // (3, 2, 4)
    bool add
)
{
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    //unsigned int flattened_idx = idx4_to_idx1(
    //    0, z, y, x, 
    //    Tuple4D {
    //        1,
    //        gridDim.z * blockDim.z,
    //        gridDim.y * blockDim.y,
    //        gridDim.x * blockDim.x
    //    }
    //);

    // calculate input gradients (original grads matmul with transposed weights)
    //if (flattened_idx < z0 * y0 * x0)
    if (z < z0 && y < y0 && x < x0)
    {
        //Tuple4D idx3d = idx1_to_idx4(flattened_idx, Tuple4D {1, z0, y0, x0});
        matmul(
            original_grads, z2, y2, x2, false,
            weight_tensor, z1, y1, x1, true,
            x2,
            input_grads, z0, y0, x0,
            z, y, x, add
        );
    }
}

__global__ void calc_weight_grads_kernel(
    float* weight_grads, unsigned int z1, unsigned int y1, unsigned int x1, // (3, 1, 5, 4)
    float* bias_grads, unsigned int z2, unsigned int y2, unsigned int x2, // (3, 2, 5, 4) -> (3, 2, 4)
    float* original_grads, // (3, 2, 4), same dimensions as bias grads
    float* input_tensor, unsigned int z0, unsigned int y0, unsigned int x0,// (3, 2, 5, 1)
    bool add
)
{
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    //unsigned int flattened_idx = idx4_to_idx1(
    //    0, z, y, x, 
    //    Tuple4D {
    //        1,
    //        gridDim.z * blockDim.z,
    //        gridDim.y * blockDim.y,
    //        gridDim.x * blockDim.x
    //    }
    //);

    // calculate weight gradients (transposed input matmul with original grads)
    if (z < z1 && y < y1 && x < x1)
    //if (flattened_idx < z1 * y1 * x1)
    {
        //Tuple4D idx3d = idx1_to_idx4(flattened_idx, Tuple4D {1, z1, y1, x1});
        matmul(
            input_tensor, z0, y0, x0, true,
            original_grads, z2, y2, x2, false,
            y0,
            weight_grads, z1, y1, x1,
            z, y, x, add
        );
    }
}

__global__ void calc_bias_grads_kernel(
    float* bias_grads, unsigned int z2, unsigned int y2, unsigned int x2, // (3, 2, 5, 4) -> (3, 2, 4)
    float* original_grads
)
{
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // calculate bias gradients (same as original gradients)
    if (z < z2 && y < y2 && x < x2)
    //if (flattened_idx < z2 * y2 * x2)
    {
        //printf("%u, %u, %u |  %u, %u, %u\n", z, y, x, z2, y2, x2);
        int flattened_idx = z * y2 * x2 + y * x2 + x;
        bias_grads[flattened_idx] = original_grads[flattened_idx];
    }
}

/*
// assumes batch matrix multiplication
// assumes ALL matrices have the same third dimension
// broadcasting of input matrix needs to be done separately
// assumes all shapes are correct
__global__ void matmul_add_kernel(
    float* mat0, unsigned int z0, unsigned int y0, unsigned int x0, // (3, 2, 5)
    float* mat1, unsigned int z1, unsigned int y1, unsigned int x1, // (3, 5, 4), actually (3, 1, 5, 4) for compatibility with
    //float* broadcast_buffer,// broadcast buffer shape = (3, 2, 5, 4)
    float* result_mat, unsigned int z2, unsigned int y2, unsigned int x2, // (3, 2, 4)
    float* mat2 // same dimension as result mat
    )
{
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (z < z2 && y < y2 && x < x2)
    {
        // initialise the result matrix to zero
        unsigned int flattened_idx = z * y2 * x2 + y * x2 + x;
        result_mat[flattened_idx] = 0.0f;
    }

    // check if thread index can fit in the maximum tensor coordinates
    // shape of the broadcast tensor
    // only include indexes that will be used for all operations
    if (z < z0 && y < y0 && x < x0 * x1)
    {
        // treat input matrix as (3, 2, 5, 1) from (3, 2, 5)
        // broadcast multiply values from input matrix to broadcast buffer (3, 2, 5, 4)
        // map original x, y, z indexes to 4d matrix

        // currently indexes are in 3d, need to 
        // convert 3d index of threads to 4d index of broadcast buffer shape
        // through flattened index
        unsigned int flattened_idx = idx4_to_idx1(0, z, y, x, Tuple4D { 1, z0, y0, x0 * x1 });

        // convert flattened to 4d given broadcast buffer shape
        Tuple4D broadcast_idx_4d = idx1_to_idx4(flattened_idx, Tuple4D {z0, y0, x0, x1});

        //printf("%d, %d, %d, %d\n", broadcast_idx_4d.a, broadcast_idx_4d.z, broadcast_idx_4d.y, broadcast_idx_4d.x);
        // obtain the flattened index of broadcast_idx_4d
        unsigned int broadcast_idx_flattened = idx4_to_idx1(
            broadcast_idx_4d.a, broadcast_idx_4d.z, broadcast_idx_4d.y, broadcast_idx_4d.x,
            Tuple4D {z0, y0, x0, x1});

        // obtain the corresponding flattened index derived from the broadcast index
        // for the input matrix (3, 2, 5, 1)
        // where it will need to elementwise multiply values
        unsigned int input_idx_flattened = idx4_to_idx1(
            broadcast_idx_4d.a, broadcast_idx_4d.z, broadcast_idx_4d.y, 0,
            Tuple4D {z0, y0, x0, 1});

        // obtain the corresponding flattened index derived from the broadcast index
        // for the weight matrix (3, 1, 5, 4)
        // where it will need to elementwise multiply values
        unsigned int weight_idx_flattened = idx4_to_idx1(
            broadcast_idx_4d.a, 0, broadcast_idx_4d.y, broadcast_idx_4d.x,
            Tuple4D {z0, 1, x0, x1});
                
        // perform broadcast/elementwise multiplication between inputs and outputs
        //broadcast_buffer[broadcast_idx_flattened] 
        //    = mat0[input_idx_flattened] * mat1[weight_idx_flattened];
        float product = mat0[input_idx_flattened] * mat1[weight_idx_flattened];
        
        // sum along the third axis of the broadcast buffer
        // get result_mat index to assign to based off the broadcast index
        // atomic add the value to flattened index of result matrix
        // 4d index still used to represent target index, third axis 
        // of 1 can be collapsed, same array length
        unsigned int result_flattened_idx = idx4_to_idx1(
            broadcast_idx_4d.a, broadcast_idx_4d.z, 0, broadcast_idx_4d.x,
            Tuple4D {z0, y0, 1, x1});

        // atomic add to first index of third axis, equivalent to summing along axis
        //atomicAdd(&result_mat[result_flattened_idx], broadcast_buffer[broadcast_idx_flattened]);
        atomicAdd(&result_mat[result_flattened_idx], product);

        // elementwise add mat2
        result_mat[result_flattened_idx] += mat2[result_flattened_idx];
    }
}

__global__ void matmul_add_back_kernel(
    float* input_grads, unsigned int z0, unsigned int y0, unsigned int x0, // (3, 2, 5, 1)
    float* weight_grads, unsigned int z1, unsigned int y1, unsigned int x1, // (3, 1, 5, 4)
    float* bias_grads, unsigned int z2, unsigned int y2, unsigned int x2, // (3, 2, 5, 4) -> (3, 2, 4)
    float* original_grads, // (3, 2, 4)
    float* input_tensor, // (3, 2, 5, 1)
    float* weight_tensor // (3, 1, 5, 4)
)
{
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    // get flattened index based on the kernel shape
    unsigned int flattened_idx = idx4_to_idx1(
        0, z, y, x, 
        Tuple4D {
            1,
            gridDim.z * blockDim.z,
            gridDim.y * blockDim.y,
            gridDim.x * blockDim.x
        }
    );

    // calculating update gradients for biases (derivative is 1)
    if (flattened_idx < z2 * y2 * x2)
    {
        //unsigned int flattened_idx = idx4_to_idx1(0, z, y, x, Tuple4D { 1, z2, y2, x2 });
        bias_grads[flattened_idx] = 1.0f * original_grads[flattened_idx];
    }
    
    // calculating update gradients for inputs and weights
    // use dimensions of broadcast intermediate tensor
    if (flattened_idx < z0 * y0 * x0 * x1)
    {
        // currently indexes are in 3d, need to 
        // convert 3d index of threads to 4d index of broadcast buffer shape
        // through flattened index
        //unsigned int flattened_idx = idx4_to_idx1(0, z, y, x, Tuple4D { 1, z0, y0, x0 * x1 });

        // convert flattened to 4d given broadcast buffer shape and the flattened index
        Tuple4D broadcast_idx_4d = idx1_to_idx4(flattened_idx, Tuple4D {z0, y0, x0, x1});

        // get flattened index of value for original gradients
        // flattened index of where to get gradient required
        // original gradients are broadcasted along the 3rd axis
        unsigned int flattened_grad_idx = idx4_to_idx1(
            broadcast_idx_4d.a, broadcast_idx_4d.z, 0, broadcast_idx_4d.x, 
            Tuple4D {z0, y0, 1, x1}
        );

        // get flattened index of value from input matrix 
        // shape of (3, 2, 5, 1)
        unsigned int flattened_input_idx = idx4_to_idx1(
            broadcast_idx_4d.a, broadcast_idx_4d.z, broadcast_idx_4d.y, 0, 
            Tuple4D {z0, y0, x0, 1}
        );

        //////////////////////////////////////////////////
        // calculate update gradients for weights
        // inputs broadcast multiply along grads
        float grad_prod;
        grad_prod = input_tensor[flattened_input_idx] * original_grads[flattened_grad_idx];

        // get flattened index of destination from weight gradient matrix 
        // shape of (3, 1, 5, 4)
        unsigned int flattened_weight_grad_idx = idx4_to_idx1(
            broadcast_idx_4d.a, 0, broadcast_idx_4d.y, broadcast_idx_4d.x, 
            Tuple4D {z0, 1, x0, x1}
        );

        // sum along second axis
        atomicAdd(&weight_grads[flattened_weight_grad_idx], grad_prod);

        //////////////////////////////////////////////////
        // calculate gradients for inputs (chained gradients to return)
        // original gradients * weights
        grad_prod = weight_tensor[flattened_weight_grad_idx] * original_grads[flattened_grad_idx];
        // sum along 4th axis, input has shape of (3, 2, 5, 1)
        atomicAdd(&input_grads[flattened_input_idx], grad_prod);
    }
}
///////////////////////////////////////////////////////////////////////////////////////
*/