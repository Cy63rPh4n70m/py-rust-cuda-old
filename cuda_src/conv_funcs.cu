#include "conv_funcs.cuh"
#include "array_ops.cuh"
#include "external.cuh"

void conv2d_forward_ext(
    float* input_img, float* filters, float* output_img,
    unsigned int in_z, unsigned int in_y, unsigned int in_x,
    unsigned int out_z, unsigned int out_y, unsigned int out_x,
    unsigned int filter_dim, unsigned int strides, float* biases, bool zero_output
)
{
    KernelDim kernel_size = get_kernel_dim1(out_z, out_y, out_x, 1, 1, 128);

    conv2d_forward_kernel<<<kernel_size.grid_dim, kernel_size.block_dim>>>
    (
        input_img, filters, output_img,
        in_z, in_y, in_x,
        out_z, out_y, out_x,
        filter_dim, strides, biases, zero_output
    );

    cudaDeviceSynchronize();
}

__global__ void conv2d_forward_kernel(
    float* input_img, float* filters, float* output,
    unsigned int in_z, unsigned int in_y, unsigned int in_x,
    unsigned int out_z, unsigned int out_y, unsigned int out_x,
    unsigned int filter_dim, unsigned int strides, float* biases, bool zero_output
)
{
    unsigned int thread_z = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int thread_y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_z < out_z && thread_y < out_y && thread_x < out_x)
    {
        // for each output pixel, obtain the top leftmost pixel in input image
        unsigned int first_y = thread_y * strides;
        unsigned int first_x = thread_x * strides;

        float dot_product = 0.0f;

        // iterate the filter kernel on the corresponding indexes for the output pixel
        // nested loop -> input_channels -> filter_dim -> filter_dim
        unsigned int input_idx, filter_idx;
        for (unsigned int z = 0; z < in_z; z++)
        {
            for (unsigned int y = 0; y < filter_dim; y++)
            {
                for (unsigned int x = 0; x < filter_dim; x++)
                {
                    // check for out of bounds
                    //if (first_y + y < in_y && first_x + x < in_x)
                    {
                        input_idx = idx4_to_idx1(
                            0, z, first_y + y, first_x + x, 
                            Tuple4D {1, in_z, in_y, in_x}
                        );

                        // filters are in the shape (out_batch, input_batch, filter_dim, filter_dim)
                        filter_idx = idx4_to_idx1(
                            thread_z, z, y, x, 
                            Tuple4D {out_z, in_z, filter_dim, filter_dim}
                        );

                        dot_product += input_img[input_idx] * filters[filter_idx];
                    }
                }
            }
        }

        // write dot product to output
        unsigned int output_idx = idx4_to_idx1(
            0, thread_z, thread_y, thread_x, 
            Tuple4D {1, out_z, out_y, out_x}
        );
        if (zero_output)
        {
            output[output_idx] = dot_product + biases[output_idx];
        }
        else
        {
            output[output_idx] += dot_product + biases[output_idx];
        }
    }
}

void conv2d_backward_ext(
    float* input_img, float* input_grads, float* input_grads_count,
    float* filters, float* filter_grads, float* filter_grads_count,
    float* output_grads, float* bias_grads,
    unsigned int in_z, unsigned int in_y, unsigned int in_x,
    unsigned int out_z, unsigned int out_y, unsigned int out_x,
    unsigned int filter_dim, unsigned int strides
)
{
    KernelDim kernel_size = get_kernel_dim1(out_z, out_y, out_x, 1, 1, 128);

    //zeroes_3d_ext(input_grads_count, in_z, in_y, in_x);
    //zeroes_3d_ext(filter_grads_count, out_z * in_z, filter_dim, filter_dim);

    conv2d_backward_kernel<<<kernel_size.grid_dim, kernel_size.block_dim>>>
    (
        input_img, input_grads, input_grads_count,
        filters, filter_grads, filter_grads_count,
        output_grads, bias_grads,
        in_z, in_y, in_x,
        out_z, out_y, out_x,
        filter_dim, strides
    );

    cudaDeviceSynchronize();

    //element_op_3d_ext(input_grads, input_grads_count, 3, in_z, in_y, in_x);
    //element_op_3d_ext(filter_grads, filter_grads_count, 3, out_z * in_z, filter_dim, filter_dim);
}

__global__ void conv2d_backward_kernel(
    float* input_img, float* input_grads, float* input_grads_count,
    float* filters, float* filter_grads, float* filter_grads_count,
    float* output_grads, float* bias_grads,
    unsigned int in_z, unsigned int in_y, unsigned int in_x,
    unsigned int out_z, unsigned int out_y, unsigned int out_x,
    unsigned int filter_dim, unsigned int strides
)
{
    unsigned int thread_z = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int thread_y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int thread_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_z < out_z && thread_y < out_y && thread_x < out_x)
    {
        // for each output pixel, obtain the top leftmost pixel in input image
        unsigned int first_y = thread_y * strides;
        unsigned int first_x = thread_x * strides;

        // iterate the filter kernel on the corresponding indexes for the output pixel
        // nested loop -> input_channels -> filter_dim -> filter_dim
        unsigned int input_idx, filter_idx;
        
        // get index of output
        unsigned int output_idx = idx4_to_idx1(
            0, thread_z, thread_y, thread_x, 
            Tuple4D {1, out_z, out_y, out_x}
        );

        bias_grads[output_idx] += output_grads[output_idx];

        for (unsigned int z = 0; z < in_z; z++)
        {
            for (unsigned int y = 0; y < filter_dim; y++)
            {
                for (unsigned int x = 0; x < filter_dim; x++)
                {
                    // check for out of bounds
                    //if (first_y + y < in_y && first_x + x < in_x)
                    {
                        input_idx = idx4_to_idx1(
                            0, z, first_y + y, first_x + x, 
                            Tuple4D {1, in_z, in_y, in_x}
                        );

                        // filters are in the shape (out_batch, input_batch, filter_dim, filter_dim)
                        filter_idx = idx4_to_idx1(
                            thread_z, z, y, x, 
                            Tuple4D {out_z, in_z, filter_dim, filter_dim}
                        );
                        
                        // y = x * w
                        // calculate gradients for input: w
                        atomicAdd(&input_grads[input_idx], filters[filter_idx] * output_grads[output_idx]);
                        //atomicAdd(&input_grads_count[input_idx], 1.0f);
                        // calculate gradients for filter weight: x
                        atomicAdd(&filter_grads[filter_idx], input_img[input_idx] * output_grads[output_idx]);
                        //atomicAdd(&filter_grads_count[filter_idx], 1.0f);
                    }
                }
            }
        }
    }
}