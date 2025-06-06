// allow c functions that use CUDA kernels to be accessible to rust
use std::{os::raw::c_void, process::exit};
#[link(name="test/nnpackage/backend/cuda_backend", kind="dylib")]
extern "C"
{
    fn matmul_bias_tiled_ext(
        mat0: *mut f32, z0: u32, y0: u32, x0: u32,
        mat1: *mut f32, z1: u32, y1: u32, x1: u32,
        result: *mut f32, mat2: *mut f32, use_bias: bool, zero_output: bool
    );

    fn matmul_bias_back_ext(
        input_grads: *mut f32, z0: u32, y0: u32, x0: u32, // (3, 2, 5, 1)
        weight_grads: *mut f32, z1: u32, y1: u32, x1: u32, // (3, 1, 5, 4)
        bias_grads: *mut f32, z2: u32, y2: u32, x2: u32, // (3, 2, 5, 4) -> (3, 2, 4)
        original_grads: *mut f32, // (3, 2, 4)
        input_tensor: *mut f32,
        weight_tensor: *mut f32,
        use_bias: bool,
        zero_input_grad: bool,
        zero_weight_grad: bool
    );

    fn activation3d_cuda_ext(
        result: *mut f32, mat: *mut f32, z: i32, y: i32, x: i32, func_id: i32, scale: f32, zero_output: bool
    );
    fn activation3d_backward_cuda_ext(
        chained_grads: *mut f32, inputs: *mut f32, original_grad: *mut f32, 
        z: i32, y: i32, x: i32, func_id: i32, 
        scale: f32, zero_input_grad: bool
    );
    fn conv2d_forward_ext(
        input_img: *mut f32, filters: *mut f32, output_img: *mut f32,
        in_z: u32, in_y: u32, in_x: u32,
        out_z: u32, out_y: u32, out_x: u32,
        filter_dim: u32, strides: u32, biases: *mut f32, zero_output: bool
    );

    fn conv2d_backward_ext(
        input_img: *mut f32, input_grads: *mut f32, input_grads_count: *mut f32,
        filters: *mut f32, filter_grads: *mut f32, filter_grads_count: *mut f32,
        output_grads: *mut f32, bias_grads: *mut f32,
        in_z: u32, in_y: u32, in_x: u32,
        out_z: u32, out_y: u32, out_x: u32,
        filter_dim: u32, strides: u32
    );

    ////////////////////////////////////////////////////////////////////////////
    /*
    fn kqv_forward_ext(
        input_ptr: *mut f32, z0: u32, y0: u32, x0: u32,
        k_weight_ptr: *mut f32, k_bias_ptr: *mut f32, z1: u32, y1: u32, x1: u32,
        q_weight_ptr: *mut f32, q_bias_ptr: *mut f32, 
        v_weight_ptr: *mut f32, v_bias_ptr: *mut f32,
        k_result_ptr: *mut f32, q_result_ptr: *mut f32, v_result_ptr: *mut f32
    );

    fn kqv_backward_ext(
        input_tensor: *mut f32,
        input_grads: *mut f32, z0: u32, y0: u32, x0: u32, // (3, 2, 5, 1)
        k_weight_grads: *mut f32, z1: u32, y1: u32, x1: u32, // (3, 1, 5, 4)
        k_bias_grads: *mut f32, z2: u32, y2: u32, x2: u32, // (3, 2, 5, 4) -> (3, 2, 4)
        k_original_grads: *mut f32, // (3, 2, 4)
        k_weight_tensor: *mut f32,

        q_weight_grads: *mut f32, 
        q_bias_grads: *mut f32,
        q_original_grads: *mut f32, // (3, 2, 4)
        q_weight_tensor: *mut f32,

        v_weight_grads: *mut f32, 
        v_bias_grads: *mut f32,
        v_original_grads: *mut f32, // (3, 2, 4)
        v_weight_tensor: *mut f32
    );

    fn kqv_update_params_ext(
        lr: f32,
        k_weight: *mut f32, k_weight_grads: *mut f32, z0: u32, y0: u32, x0: u32,
        k_biases: *mut f32, k_biases_grads: *mut f32, z1: u32, y1: u32, x1: u32,
        q_weight: *mut f32, q_weight_grads: *mut f32,
        q_biases: *mut f32, q_biases_grads: *mut f32,
        v_weight: *mut f32, v_weight_grads: *mut f32, z2: u32, y2: u32, x2: u32,
        v_biases: *mut f32, v_biases_grads: *mut f32, z3: u32, y3: u32, x3: u32
    );
    */
    
    fn embedding_forward_ext(
        index_vec: *mut f32, seq_len: u32,
        embedding_lookup: *mut f32, vocab_size: u32, embedding_len: u32,
        result_embedding: *mut f32
    );

    fn embedding_backward_ext(
        index_vec: *mut f32, seq_len: u32,
        embedding_lookup: *mut f32, vocab_size: u32, embedding_len: u32,
        embedding_lookup_grads: *mut f32, 
        embedding_lookup_grads_temp: *mut f32, 
        embedding_lookup_grads_count: *mut f32, 
        prev_grads: *mut f32,
        result_grads: *mut f32
    );

    fn l2norm_forward_ext(
        original: *mut f32, original_pow2: *mut f32, z0: u32, y0: u32, x0: u32,
        power_sum: *mut f32, normalized: *mut f32, zeroed: bool
    );

    fn l2norm_backward_ext(
        original_grads: *mut f32, z0: u32, y0: u32, x0: u32,
        norm_ptr: *mut f32, original_inputs: *mut f32, original_outputs: *mut f32, result_grads: *mut f32,
        zeroed: bool
    );

    /*
    fn rms_norm_forward_ext(
        original: *mut f32, original_pow2: *mut f32, z0: u32, y0: u32, x0: u32,
        norm_ptr: *mut f32, scale_ptr: *mut f32, bias_ptr: *mut f32, normalized: *mut f32, zeroed: bool
    );

    fn rms_norm_backward_ext(
        original_grads: *mut f32, z0: u32, y0: u32, x0: u32,
        norm_ptr: *mut f32, original_inputs: *mut f32, result_grads: *mut f32, scale_ptr: *mut f32,
        scale_grad_ptr: *mut f32, bias_grad_ptr: *mut f32
    );
    */

    fn softmax_forward_ext(
        input: *mut f32, exp_input: *mut f32, exp_sum: *mut f32,
        result: *mut f32, broadcast_temp: *mut f32, temperature: f32,
        z0: u32, y0: u32, x0: u32, zero_output: bool
    );

    fn dropout_forward_ext(
        input: *mut f32, output: *mut f32, mask: *mut f32, states: *mut c_void,
        z0: u32, y0: u32, x0: u32, dropout_rate: f32
    );

    fn dropout_backward_ext(
        original_grads: *mut f32, 
        dropout_mask: *mut f32, result_grads: *mut f32,
        z0: u32, y0: u32, x0: u32
    );

    fn init_random_states_ext(z0: u32, y0: u32, x0: u32) -> *mut c_void;
    fn elementwise_dropout_forward_ext(
        input: *mut f32, weights: *mut f32, output: *mut f32, mask: *mut f32, states: *mut c_void,
        z0: u32, y0: u32, x0: u32, dropout_rate: f32, op: u32, activation_id: i32, act_scale: f32,
        use_dropout: bool, zero_output: bool
    );
    fn elementwise_dropout_backward_ext(
        original_grads: *mut f32,
        dropout_mask: *mut f32, 
        input: *mut f32, weights: *mut f32, weight_grads: *mut f32,
        result_grads: *mut f32, use_dropout: bool, activation_id: i32, act_scale: f32,
        z0: u32, y0: u32, x0: u32, op: u32,
        zero_input_grad: bool, zero_weight_grad: bool
    );

    ////////////////////////////////////////////////////////////////////////////

    fn element_op_3d_ext(dst: *mut f32, src: *mut f32, op: i32, z0: i32, y0: i32, x0: i32);
    fn scalar_op_3d_inplace_ext(dst: *mut f32, scalar: f32, op: i32, z0: i32, y0: i32, x0: i32);
    fn zeroes_3d_ext(dst: *mut f32, z0: i32, y0: i32, x0: i32);
    fn gradient_desc_3d_ext(
        lr: f32, l2: f32,
        weight_ptr: *mut f32, weight_ptr_grad: *mut f32, weight_velocity: *mut f32, weight_momentum: *mut f32, 
        z0: u32, y0: u32, x0: u32,
        bias_ptr: *mut f32, bias_ptr_grad: *mut f32, bias_velocity: *mut f32, bias_momentum: *mut f32, 
        z1: u32, y1: u32, x1: u32,
        only_beta: bool, non_neg: bool, batch_size: f32, optimizer_type: i32, 
        alpha: f32, beta: f32
    );
    fn transpose_2d_ext(arr_t: *mut f32, arr: *mut f32, z0: u32, y0: u32, x0: u32);
    fn broadcast_2d_to_3d_ext(
        broadcasted: *mut f32, z0: u32, y0: u32, x0: u32, original: *mut f32, axis: i32);
    fn sum_axis_ext(summed: *mut f32, original: *mut f32, z0: u32, y0: u32, x0: u32, axis: i32, zeroed: bool);

    fn softmax_ce_loss_ext(
        pred: *mut f32, classes: *mut f32, loss_vals: *mut f32, grad_ptr: *mut f32, 
        z0: i32, y0: i32, x0: i32
    );

    fn init_cuda_array(length: u32) -> *mut f32;
    fn init_cpu_pinned_array(length: u32) -> *mut f32;
    fn to_cuda_array(array: *mut f32, length: u32) -> *mut f32;
    fn to_cpu_array(cuda_array: *mut f32, length: u32) -> *mut f32;
    fn cuda_ptr_from_pinned_ext(host: *mut f32) -> *mut f32;
    fn copy_host_to_cuda_array(dst: *mut f32, src: *mut f32, length: u32);
    fn copy_host_to_host_ext(dst: *mut f32, src: *mut f32, length: u32);
    fn cuda_to_cuda_ext(dst: *mut f32, src: *mut f32, length: u32);
    fn free_cpu_array_ext(array: *mut f32);
    fn free_cuda_array_ext(array: *mut c_void);
    fn free_pinned_array_ext(array: *mut c_void);
}

pub fn matmul_add_bias_tiled(
    mat0: *mut f32, z0: u32, y0: u32, x0: u32, 
    mat1: *mut f32, z1: u32, y1: u32, x1: u32,
    result: *mut f32, mat2: *mut f32, use_bias: bool, zero_output: bool
)
{
    unsafe { matmul_bias_tiled_ext(mat0, z0, y0, x0, mat1, z1, y1, x1, result, mat2, use_bias, zero_output); }//, broadcast_buffer);
}

pub fn matmul_add_bias_back(
    input_grads: *mut f32, z0: u32, y0: u32, x0: u32, // (3, 2, 5, 1)
    weight_grads: *mut f32, z1: u32, y1: u32, x1: u32, // (3, 1, 5, 4)
    bias_grads: *mut f32, z2: u32, y2: u32, x2: u32, // (3, 2, 5, 4) -> (3, 2, 4)
    original_grads: *mut f32, // (3, 2, 4)
    input_tensor: *mut f32,
    weight_tensor: *mut f32,
    use_bias: bool,
    zero_input_grad: bool,
    zero_weight_grad: bool
)
{
    unsafe
    {
        matmul_bias_back_ext(
            input_grads, z0, y0, x0, // (3, 2, 5, 1)
            weight_grads, z1, y1, x1, // (3, 1, 5, 4)
            bias_grads, z2, y2, x2, // (3, 2, 5, 4) -> (3, 2, 4)
            original_grads, // (3, 2, 4)
            input_tensor,// (3, 2, 5, 1)
            weight_tensor, // (3, 1, 5, 4)
            use_bias,
            zero_input_grad,
            zero_weight_grad
        );
    }
}

pub fn activation3d_cuda(
    result: *mut f32, mat: *mut f32, z: u32, y: u32, x: u32, func_name: &str,
    scale: f32, zero_output: bool
)
{
    let func_id: u32;
    match func_name
    {
        "sigmoid" => func_id = 0,
        "tanh" => func_id = 1,
        "silu" => func_id = 2,
        "gelu" => func_id = 3,
        "tanh2" => func_id = 4,
        "softplus" => func_id = 5,
        "linear" => func_id = 6,
        _ => {println!("Invalid activation {:?}", func_name); exit(1)},
    };

    unsafe { activation3d_cuda_ext(result, mat, z as i32, y as i32, x as i32, func_id as i32, scale, zero_output); }
}

pub fn activation3d_cuda_backward(
    chained_grads: *mut f32, inputs: *mut f32, original_grad: *mut f32, 
    z: u32, y: u32, x: u32, func_name: &str,
    scale: f32, zero_input_grad: bool
)
{
    let func_id: u32;
    match func_name
    {
        "sigmoid" => func_id = 0,
        "tanh" => func_id = 1,
        "silu" => func_id = 2,
        "gelu" => func_id = 3,
        "tanh2" => func_id = 4,
        "softplus" => func_id = 5,
        "linear" => func_id = 6,
        _ => {println!("Invalid activation {:?}", func_name); exit(1)},
    };

    unsafe 
    { 
        activation3d_backward_cuda_ext(
            chained_grads, inputs, original_grad, 
            z as i32, y as i32, x as i32, func_id as i32,
            scale, zero_input_grad
        ); 
    }
}

pub fn conv2d_forward(
    input_img: *mut f32, filters: *mut f32, output_img: *mut f32,
    in_z: u32, in_y: u32, in_x: u32,
    out_z: u32, out_y: u32, out_x: u32,
    filter_dim: u32, strides: u32, biases: *mut f32, zero_output: bool
)
{
    unsafe
    {
        conv2d_forward_ext(
            input_img, filters, output_img,
            in_z, in_y, in_x,
            out_z, out_y, out_x,
            filter_dim, strides, biases, zero_output
        );
    }
}

pub fn conv2d_backward(
    input_img: *mut f32, input_grads: *mut f32, input_grads_count: *mut f32,
    filters: *mut f32, filter_grads: *mut f32, filter_grads_count: *mut f32,
    output_grads: *mut f32, bias_grads: *mut f32,
    in_z: u32, in_y: u32, in_x: u32,
    out_z: u32, out_y: u32, out_x: u32,
    filter_dim: u32, strides: u32
)
{
    unsafe
    {
        conv2d_backward_ext(
            input_img, input_grads, input_grads_count,
            filters, filter_grads, filter_grads_count,
            output_grads, bias_grads,
            in_z, in_y, in_x,
            out_z, out_y, out_x,
            filter_dim, strides
        );
    }
}


/////////////////////////////////////////////////////////////////////
/*
pub fn kqv_forward(
    input_ptr: *mut f32, z0: usize, y0: usize, x0: usize,
    k_weight_ptr: *mut f32, k_bias_ptr: *mut f32, z1: usize, y1: usize, x1: usize,
    q_weight_ptr: *mut f32, q_bias_ptr: *mut f32, 
    v_weight_ptr: *mut f32, v_bias_ptr: *mut f32,
    k_result_ptr: *mut f32, q_result_ptr: *mut f32, v_result_ptr: *mut f32
)
{
    unsafe
    {
        kqv_forward_ext(
            input_ptr, z0 as u32, y0 as u32, x0 as u32,
            k_weight_ptr, k_bias_ptr, z1 as u32, y1 as u32, x1 as u32,
            q_weight_ptr, q_bias_ptr, 
            v_weight_ptr, v_bias_ptr,
            k_result_ptr, q_result_ptr, v_result_ptr
        );
    }
}

pub fn kqv_backward(
    input_tensor: *mut f32,
    input_grads: *mut f32, z0: usize, y0: usize, x0: usize, // (3, 2, 5, 1)
    k_weight_grads: *mut f32, z1: usize, y1: usize, x1: usize, // (3, 1, 5, 4)
    k_bias_grads: *mut f32, z2: usize, y2: usize, x2: usize, // (3, 2, 5, 4) -> (3, 2, 4)
    k_original_grads: *mut f32, // (3, 2, 4)
    k_weight_tensor: *mut f32,

    q_weight_grads: *mut f32, 
    q_bias_grads: *mut f32,
    q_original_grads: *mut f32, // (3, 2, 4)
    q_weight_tensor: *mut f32,

    v_weight_grads: *mut f32, 
    v_bias_grads: *mut f32,
    v_original_grads: *mut f32, // (3, 2, 4)
    v_weight_tensor: *mut f32
)
{
    unsafe
    {
        kqv_backward_ext(
            input_tensor,
            input_grads, z0 as u32, y0 as u32, x0 as u32, // (3, 2, 5, 1)
            k_weight_grads, z1 as u32, y1 as u32, x1 as u32, // (3, 1, 5, 4)
            k_bias_grads, z2 as u32, y2 as u32, x2 as u32, // (3, 2, 5, 4) -> (3, 2, 4)
            k_original_grads, // (3, 2, 4)
            k_weight_tensor,
        
            q_weight_grads, 
            q_bias_grads,
            q_original_grads, // (3, 2, 4)
            q_weight_tensor,
        
            v_weight_grads, 
            v_bias_grads,
            v_original_grads, // (3, 2, 4)
            v_weight_tensor
        );
    }
}

pub fn kqv_update_params(
    lr: f32,
    k_weight: *mut f32, k_weight_grads: *mut f32, z0: usize, y0: usize, x0: usize,
    k_biases: *mut f32, k_biases_grads: *mut f32, z1: usize, y1: usize, x1: usize,
    q_weight: *mut f32, q_weight_grads: *mut f32,
    q_biases: *mut f32, q_biases_grads: *mut f32,
    v_weight: *mut f32, v_weight_grads: *mut f32, z2: usize, y2: usize, x2: usize,
    v_biases: *mut f32, v_biases_grads: *mut f32, z3: usize, y3: usize, x3: usize
)
{
    unsafe
    {
        kqv_update_params_ext(
            lr,
            k_weight, k_weight_grads, z0 as u32, y0 as u32, x0 as u32,
            k_biases, k_biases_grads, z1 as u32, y1 as u32, x1 as u32,
            q_weight, q_weight_grads,
            q_biases, q_biases_grads,
            v_weight, v_weight_grads, z2 as u32, y2 as u32, x2 as u32,
            v_biases, v_biases_grads, z3 as u32, y3 as u32, x3 as u32
        );
    }
}
*/

pub fn embedding_forward(
    index_vec: *mut f32, seq_len: usize,
    embedding_lookup: *mut f32, vocab_size: usize, embedding_len: usize,
    result_embedding: *mut f32
)
{
    unsafe {
        embedding_forward_ext(
            index_vec, seq_len as u32,
            embedding_lookup, vocab_size as u32, embedding_len as u32,
            result_embedding
        );
    }
}

pub fn embedding_backward(
    index_vec: *mut f32, seq_len: usize,
    embedding_lookup: *mut f32, vocab_size: usize, embedding_len: usize,
    embedding_lookup_grads: *mut f32, 
    embedding_lookup_grads_temp: *mut f32, 
    embedding_lookup_grads_count: *mut f32, 
    prev_grads: *mut f32,
    result_grads: *mut f32
)
{
    unsafe {
        embedding_backward_ext(
            index_vec, seq_len as u32,
            embedding_lookup, vocab_size as u32, embedding_len as u32,
            embedding_lookup_grads, 
            embedding_lookup_grads_temp, 
            embedding_lookup_grads_count, 
            prev_grads,
            result_grads
        );
    }
}

pub fn l2norm_forward(
    original: *mut f32, original_pow2: *mut f32, z0: usize, y0: usize, x0: usize,
    norm_ptr: *mut f32, normalized: *mut f32, zeroed: bool
)
{
    unsafe
    {
        l2norm_forward_ext(
            original, original_pow2, z0 as u32, y0 as u32, x0 as u32,
            norm_ptr, normalized, zeroed
        );
    }
}

pub fn l2norm_backward(
    original_grads: *mut f32, z0: usize, y0: usize, x0: usize,
    norm_ptr: *mut f32, original_inputs: *mut f32, original_outputs: *mut f32,
    result_grads: *mut f32, zeroed: bool
)
{
    unsafe
    {
        l2norm_backward_ext(
            original_grads, z0 as u32, y0 as u32, x0 as u32,
            norm_ptr, original_inputs, original_outputs, result_grads,
            zeroed
        );
    }
}
/*
pub fn rms_norm_forward(
    original: *mut f32, original_pow2: *mut f32, z0: usize, y0: usize, x0: usize,
    norm_ptr: *mut f32, scale_ptr: *mut f32, bias_ptr: *mut f32, normalized: *mut f32, zeroed: bool
)
{
    unsafe
    {
        rms_norm_forward_ext(
            original, original_pow2, z0 as u32, y0 as u32, x0 as u32,
            norm_ptr, scale_ptr, bias_ptr, normalized, zeroed
        );
    }
}

pub fn rms_norm_backward(
    original_grads: *mut f32, z0: usize, y0: usize, x0: usize,
    norm_ptr: *mut f32, original_inputs: *mut f32, result_grads: *mut f32, scale_ptr: *mut f32,
    scale_ptr_grad: *mut f32, bias_grad_ptr: *mut f32
)
{
    unsafe
    {
        rms_norm_backward_ext(
            original_grads, z0 as u32, y0 as u32, x0 as u32,
            norm_ptr, original_inputs, result_grads, scale_ptr,
            scale_ptr_grad, bias_grad_ptr
        );
    }
}
*/

pub fn softmax_forward(
    input: *mut f32, exp_input: *mut f32, exp_sum: *mut f32,
    result: *mut f32, broadcast_temp: *mut f32, temperature: f32,
    z0: usize, y0: usize, x0: usize, zero_output: bool
)
{
    unsafe
    {
        softmax_forward_ext(
            input, exp_input, exp_sum,
            result, broadcast_temp, temperature,
            z0 as u32, y0 as u32, x0 as u32, zero_output
        );
    }
}

pub fn dropout_forward(
    input: *mut f32, output: *mut f32, mask: *mut f32, states: *mut c_void,
    z0: usize, y0: usize, x0: usize, dropout_rate: f32
)
{
    unsafe
    {
        dropout_forward_ext(
            input, output, mask, states,
            z0 as u32, y0 as u32, x0 as u32, dropout_rate
        );
    }
}

pub fn dropout_backward(
    original_grads: *mut f32, 
    dropout_mask: *mut f32, result_grads: *mut f32,
    z0: usize, y0: usize, x0: usize
)
{
    unsafe
    {
        dropout_backward_ext(
            original_grads, 
            dropout_mask, result_grads,
            z0 as u32, y0 as u32, x0 as u32
        );
    }
}

pub fn init_random_states(z0: usize, y0: usize, x0: usize) -> *mut c_void
{
    unsafe
    {
        return init_random_states_ext(z0 as u32, y0 as u32, x0 as u32);
    }
}

pub fn elementwise_dropout_forward(
    input: *mut f32, weights: *mut f32, output: *mut f32, mask: *mut f32, states: *mut c_void,
    z0: usize, y0: usize, x0: usize, dropout_rate: f32, op: u32, activation_id: i32, act_scale: f32,
    use_dropout: bool, zero_output: bool
)
{
    unsafe
    {
        elementwise_dropout_forward_ext(
            input, weights, output, mask, states,
            z0 as u32, y0 as u32, x0 as u32, dropout_rate, op, activation_id, act_scale,
            use_dropout, zero_output
        );
    }
}

pub fn elementwise_dropout_backward(
    original_grads: *mut f32,
    dropout_mask: *mut f32, 
    input: *mut f32, weights: *mut f32, weight_grads: *mut f32,
    result_grads: *mut f32, use_dropout: bool, activation_id: i32, act_scale: f32,
    z0: usize, y0: usize, x0: usize, op: u32,
    zero_input_grad: bool, zero_weight_grad: bool
    
)
{
    unsafe
    {
        elementwise_dropout_backward_ext(
            original_grads,
            dropout_mask, 
            input, weights, weight_grads,
            result_grads, use_dropout, activation_id, act_scale,
            z0 as u32, y0 as u32, x0 as u32, op,
            zero_input_grad, zero_weight_grad
        );
    }
}

/////////////////////////////////////////////////////////////////////

pub fn element_op_3d_inplace(
    dst: *mut f32, src: *mut f32, op: i32, z0: usize, y0: usize, x0: usize
)
{
    unsafe { element_op_3d_ext(dst, src, op, z0 as i32, y0 as i32, x0 as i32) };
}

pub fn scalar_op_3d_inplace(
    dst: *mut f32, scalar: f32, op: i32, z0: usize, y0: usize, x0: usize
)
{
    unsafe { scalar_op_3d_inplace_ext(dst, scalar, op, z0 as i32, y0 as i32, x0 as i32); }
}

pub fn zeroes_3d_inplace(dst: *mut f32, z0: usize, y0: usize, x0: usize)
{
    unsafe { zeroes_3d_ext(dst, z0 as i32, y0 as i32, x0 as i32); }
}

pub fn gradient_desc_3d(
    lr: f32, l2: f32,
    weight_ptr: *mut f32, weight_ptr_grad: *mut f32, weight_velocity: *mut f32, weight_momentum: *mut f32, 
    z0: usize, y0: usize, x0: usize,
    bias_ptr: *mut f32, bias_ptr_grad: *mut f32, bias_velocity: *mut f32, bias_momentum: *mut f32,
    z1: usize, y1: usize, x1: usize,
    only_beta: bool, non_neg: bool, batch_size: f32, optimizer_type: i32, 
    alpha: f32, beta: f32
)
{
    unsafe { 
        gradient_desc_3d_ext(
            lr, l2,
            weight_ptr, weight_ptr_grad, weight_velocity, weight_momentum, 
            z0 as u32, y0 as u32, x0 as u32,
            bias_ptr, bias_ptr_grad, bias_velocity, bias_momentum, 
            z1 as u32, y1 as u32, x1 as u32,
            only_beta, non_neg, batch_size, optimizer_type, 
            alpha, beta
        );
    };
}

pub fn transpose_2d(arr_t: *mut f32, arr: *mut f32, z0: usize, y0: usize, x0: usize)
{
    unsafe { transpose_2d_ext(arr_t, arr, z0 as u32, y0 as u32, x0 as u32); }
}

pub fn broadcast_2d_to_3d(
    broadcasted: *mut f32, z0: usize, y0: usize, x0: usize, 
    original: *mut f32, axis: i32
)
{
    unsafe { 
        broadcast_2d_to_3d_ext(
            broadcasted, z0 as u32, y0 as u32, x0 as u32, 
            original, axis
        ); 
    }
}

pub fn sum_axis(summed: *mut f32, original: *mut f32, z0: usize, y0: usize, x0: usize, axis: i32, zeroed: bool)
{
    unsafe { sum_axis_ext(summed, original, z0 as u32, y0 as u32, x0 as u32, axis, zeroed) };
}

pub fn softmax_ce_loss(
    pred: *mut f32, classes: *mut f32, loss_vals: *mut f32, grad_ptr: *mut f32, 
    z0: i32, y0: i32, x0: i32
)
{
    unsafe 
    {
        softmax_ce_loss_ext(
            pred, classes, loss_vals, grad_ptr, 
            z0, y0, x0
        );
    }
}

pub fn new_cuda_array(
    length: u32, 
) -> *mut f32
{
    unsafe { return init_cuda_array(length) };
}

pub fn new_cpu_pinned_array(
    length: u32
) -> *mut f32
{
    unsafe { return init_cpu_pinned_array(length) };
}

pub fn cuda_ptr_from_pinned(host: *mut f32) -> *mut f32
{
    unsafe {return cuda_ptr_from_pinned_ext(host);}
}

pub fn to_cuda(
    array: *mut f32, length: u32) -> *mut f32
{
    let cuda_array_ptr: *mut f32;
    unsafe
    {
        cuda_array_ptr = to_cuda_array(array, length);
    }

    return cuda_array_ptr;
}

pub unsafe fn to_cpu(
    cuda_array: *mut f32, length: u32) -> *mut f32
{
    return to_cpu_array(cuda_array, length);
}

pub fn copy_host_to_cuda(dst: *mut f32, src: *mut f32, length: u32)
{
    unsafe {copy_host_to_cuda_array(dst, src, length);}
}

pub fn copy_host_to_host(dst: *mut f32, src: *mut f32, shape: &[usize])
{
    let mut length: usize = 1;
    for i in 0..shape.len()
    {
        length *= shape[i];
    }
    unsafe { copy_host_to_host_ext(dst, src, length as u32); }
}

pub fn copy_cuda_to_cuda(dst: *mut f32, src: *mut f32, shape: &[usize])
{
    let mut length: usize = 1;
    for i in 0..shape.len()
    {
        length *= shape[i];
    }
    unsafe { cuda_to_cuda_ext(dst, src, length as u32) };
}

pub unsafe fn free_cpu_array(array: *mut f32)
{
    free_cpu_array_ext(array);
}

pub fn free_cuda_array(array: *mut c_void)
{
    unsafe { free_cuda_array_ext(array); }
}

pub fn free_pinned_array(array: *mut c_void)
{
    unsafe { free_pinned_array_ext(array); }
}