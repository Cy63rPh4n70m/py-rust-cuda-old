#include "dense_funcs.cuh"
#include <cuda_fp16.h>

#pragma once

#define MAX_GRID_X 2147483647U
#define MAX_GRID_Y 1U
#define MAX_GRID_Z 1U
#define MAX_THREAD_X 256U
#define MAX_THREAD_Y 1U
#define MAX_THREAD_Z 1U

typedef struct KernelDim { dim3 grid_dim, block_dim; } KernelDim;

extern "C" 
{
    __declspec(dllexport) float* to_cuda_array(float* array, unsigned int length);
    __declspec(dllexport) float* to_cpu_array(float* cuda_array, unsigned int length);
    __declspec(dllexport) half* to_cpu_array_f16(half* cuda_array, unsigned int length);
    __declspec(dllexport) float* init_cpu_array(unsigned int length);
    __declspec(dllexport) half* init_cuda_array_f16(unsigned int length);
    __declspec(dllexport) float* init_cuda_array(unsigned int length);
    //__declspec(dllexport) int* init_int_cuda_array(unsigned int length);
    __declspec(dllexport) float* init_cpu_pinned_array(unsigned int length);
    __declspec(dllexport) float* cuda_ptr_from_pinned_ext(float* host);
    __declspec(dllexport) void copy_host_to_host_ext(float* dst, float* src, unsigned int length);
    __declspec(dllexport) void copy_host_to_cuda_array(float* dst, float* src, unsigned int length);
    __declspec(dllexport) void copy_host_to_cuda_array_f16(half* dst, half* src, unsigned int length);
    __declspec(dllexport) void copy_cuda_to_host_array(float* dst, float* src, unsigned int length);
    __declspec(dllexport) void free_cpu_array_ext(float* array);
    __declspec(dllexport) void free_cuda_array_ext(void* array);
    __declspec(dllexport) void cuda_to_cuda_ext(float* dst, float* src, unsigned int length);
}

void print_cuda_array(float* cuda_array, unsigned int length);
void print_cuda_array_int(int* cuda_array, unsigned int length);
KernelDim get_kernel_dim(unsigned int n_threads);
KernelDim get_kernel_dim1(
    unsigned int z, unsigned int y, unsigned int x, 
    unsigned int max_thread_z, unsigned int max_thread_y, unsigned int max_thread_x
);
Tuple4D idx1_to_idx4_cpu(unsigned int length, Tuple4D shape);
unsigned int idx4_to_idx1_cpu(
    unsigned int a, unsigned int z, unsigned int y, unsigned int x, 
    Tuple4D shape
);