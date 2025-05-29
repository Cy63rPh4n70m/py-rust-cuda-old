#include "dense_funcs.cuh"
#include "external.cuh"
#include <stdio.h>
#include <math.h>
#include <cuda_fp16.h>
#include <mma.h>

float* init_cuda_array(unsigned int length)
{
    float* result_mat;
    cudaMalloc((void**)&result_mat, sizeof(float) * length);
    cudaMemset(result_mat, 0, sizeof(float) * length);
    return result_mat;
}

half* init_cuda_array_f16(unsigned int length)
{
    half* result_mat;
    cudaMalloc((void**)&result_mat, sizeof(half) * length);
    cudaMemset(result_mat, 0, sizeof(half) * length);
    return result_mat;
}

float* init_cpu_array(unsigned int length)
{
    float* array = (float*) malloc(sizeof(float) * length);
    return array;
}

float* init_cpu_pinned_array(unsigned int length)
{
    float* array;
    cudaHostAlloc((void**)&array, sizeof(float) * length, cudaHostAllocMapped);
    return array;
}

float* cuda_ptr_from_pinned_ext(float* host)
{
    float* cuda_array;
    cudaHostGetDevicePointer((void**)&cuda_array, host, 0);
    return cuda_array;
}

void print_cuda_array(float* cuda_array, unsigned int length)
{
    float* array = (float*) malloc(sizeof(float) * length);
    cudaMemcpy(array, cuda_array, sizeof(float) * length, cudaMemcpyDeviceToHost);
    for (int i = 0; i < length; i++)
    {
        printf("%f, ", array[i]);
    }
    printf("\n");

    free(array);
    array = NULL;
}

void print_cuda_array_int(int* cuda_array, unsigned int length)
{
    int* array = (int*) malloc(sizeof(int) * length);
    cudaMemcpy(array, cuda_array, sizeof(int) * length, cudaMemcpyDeviceToHost);
    for (int i = 0; i < length; i++)
    {
        printf("%d, ", array[i]);
    }
    printf("\n");

    free(array);
    array = NULL;
}

// usually when cuda array is created from an ndarray raw pointer
// therefore input pointer isn't freed
float* to_cuda_array(float* array, unsigned int length)
{
    float* cuda_array = init_cuda_array(length);
    cudaMemcpy(cuda_array, array, sizeof(float) * length, cudaMemcpyHostToDevice);
    //free(array);
    //array = NULL;
    return cuda_array;
}

// will free the original cuda pointer
float* to_cpu_array(float* cuda_array, unsigned int length)
{
    float* cpu_array = (float*) malloc(sizeof(float) * length);
    cudaMemcpy(cpu_array, cuda_array, sizeof(float) * length, cudaMemcpyDeviceToHost);
    //cudaFree(cuda_array);
    //cuda_array = NULL;
    return cpu_array;
}

half* to_cpu_array_f16(half* cuda_array, unsigned int length)
{
    half* cpu_array = (half*) malloc(sizeof(half) * length);
    cudaMemcpy(cpu_array, cuda_array, sizeof(half) * length, cudaMemcpyDeviceToHost);
    //cudaFree(cuda_array);
    //cuda_array = NULL;
    return cpu_array;
}

void copy_host_to_cuda_array(float* dst, float* src, unsigned int length)
{
    cudaMemcpy(dst, src, sizeof(float) * length, cudaMemcpyHostToDevice);
}

void copy_host_to_cuda_array_f16(half* dst, half* src, unsigned int length)
{
    cudaMemcpy(dst, src, sizeof(half) * length, cudaMemcpyHostToDevice);
}

void copy_cuda_to_host_array(float* dst, float* src, unsigned int length)
{
    cudaMemcpy(dst, src, sizeof(float) * length, cudaMemcpyDeviceToHost);
}

void copy_host_to_host_ext(float* dst, float* src, unsigned int length)
{
    memcpy(dst, src, sizeof(float) * length);
}

void cuda_to_cuda_ext(float* dst, float* src, unsigned int length)
{
    cudaMemcpy(dst, src, sizeof(float) * length, cudaMemcpyDeviceToDevice);
}

void free_cpu_array_ext(float* array)
{
    free(array);
}

void free_cuda_array_ext(float* array)
{
    cudaFree(array);
}

KernelDim get_kernel_dim(unsigned int n_threads)
{
    // get the 3d index given number of total threads required and the shape
    Tuple4D shape3d = idx1_to_idx4_cpu(
        n_threads, 
        Tuple4D 
        {
            1, 
            MAX_GRID_Z * MAX_THREAD_Z, 
            MAX_GRID_Y * MAX_THREAD_Y, 
            MAX_GRID_X * MAX_THREAD_X
        }
    );
    //printf("------------------\n");
    //printf("n_threads: %u\n", n_threads);
    //printf("max_dim: %u, %u, %u\n", MAX_GRID_Z * MAX_THREAD_Z, MAX_GRID_Y * MAX_THREAD_Y, MAX_GRID_X * MAX_THREAD_X);

    //printf("4d shape: %u, %u, %u, %u\n", shape3d.a, shape3d.z, shape3d.y, shape3d.x);
    // account for subsequent axies having zero shape when n threads is large
    // would've resulted in very little grid blocks
    if (shape3d.a > 0)
    {
        shape3d.z = MAX_GRID_Z * MAX_THREAD_Z;
    }
    if (shape3d.z > 0)
    {
        shape3d.y = MAX_GRID_Y * MAX_THREAD_Y;
    }
    if (shape3d.y > 0)
    {
        shape3d.x = MAX_GRID_X * MAX_THREAD_X;
    }
    //printf("4d shape: %u, %u, %u, %u\n", shape3d.a, shape3d.z, shape3d.y, shape3d.x);

    // add one if shape is zero (shape in threads)
    if (shape3d.a == 0) {shape3d.a += 1;}
    if (shape3d.z == 0) {shape3d.z += 1;}
    if (shape3d.y == 0) {shape3d.y += 1;}
    if (shape3d.x == 0) {shape3d.x += 1;}

    // calculate grid dimensions based on shape3d and thread dimensions
    unsigned int grid_x = (unsigned int) ceil((float) shape3d.x / (float) MAX_THREAD_X);
    unsigned int grid_y = (unsigned int) ceil((float) shape3d.y / (float) MAX_THREAD_Y);
    unsigned int grid_z = (unsigned int) ceil((float) shape3d.z / (float) MAX_THREAD_Z);

    //printf("grid (x, y, z): %u, %u, %u\n", grid_x, grid_y, grid_z);
    //printf("block (x, y, z): %u, %u, %u\n", MAX_THREAD_X, MAX_THREAD_Y, MAX_THREAD_Z);
    //printf("------------------\n");
    //exit(1);

    dim3 gridsize(grid_x, grid_y, grid_z);
    dim3 blocksize(MAX_THREAD_X, MAX_THREAD_Y, MAX_THREAD_Z);

    return KernelDim { gridsize, blocksize };
}

KernelDim get_kernel_dim1(
    unsigned int z, unsigned int y, unsigned int x, 
    unsigned int max_thread_z, unsigned int max_thread_y, unsigned int max_thread_x
)
{
    unsigned int grid_x = (unsigned int) ceil((float) x / (float) max_thread_x);
    unsigned int grid_y = (unsigned int) ceil((float) y / (float) max_thread_y);
    unsigned int grid_z = (unsigned int) ceil((float) z / (float) max_thread_z);

    dim3 gridsize(grid_x, grid_y, grid_z);
    dim3 blocksize(max_thread_x, max_thread_y, max_thread_z);

    return KernelDim { gridsize, blocksize };
}


Tuple4D idx1_to_idx4_cpu(unsigned int length, Tuple4D shape)
{
    unsigned int s0 = shape.z * shape.y * shape.x;
    unsigned int s1 = shape.y * shape.x;
    unsigned int s2 = shape.x;
    unsigned int s3 = 1;

    unsigned int a = length / s0;
    unsigned int z = (length % s0) / s1;
    unsigned int y = (length % s1) / s2;
    unsigned int x = length % s2;

    return Tuple4D {a, z, y, x};
}

unsigned int idx4_to_idx1_cpu(
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