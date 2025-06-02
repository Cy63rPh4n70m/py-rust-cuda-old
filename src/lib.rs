#![allow(dead_code)]
#![allow(non_camel_case_types)]

use std::{ffi::{c_char, CStr}, os::raw::c_void, process::exit};

//use io_functions::{load_model, save_model};
use layers::{
    activation_cuda::ActivationCuda, broadcast_cuda::BroadcastCuda, 
    cls_cuda::CLSCuda, conv2d_cuda::Conv2dCuda, dense_cuda::DenseCuda, 
    dropout_cuda::DropoutCuda, elementwise_cuda::ElementwiseCuda, 
    embedding_cuda::Embedding2DCuda, l2norm_cuda::L2NormCuda, 
    softmax_cuda::SoftmaxCuda, sum_cuda::SumCuda, transpose_cuda::BatchTransposeCuda};
    
use math_functions::{get_loss_deriv_from_str, get_loss_from_str};
use ndarray::{ArrayD, IxDyn};
use neuralnet::{NeuralNet, TraversePtrs};
use storage::{ae_buf::AutoencoderBuf, storage_seq_buf::ReplayBuf};
use types::{LossFn, LossFnDeriv};

mod neuralnet;
mod math_functions;
mod types;
mod batch;
mod shaping;
//mod io_functions;
mod layers;
mod storage;
mod random_name_gen;
mod heap_dict;
mod parallel;
mod cuda_bridge;
mod pointer_ops;

#[no_mangle]
pub unsafe extern "C" fn create_model() -> *mut c_void
{
    std::env::set_var("RUST_BACKTRACE", "full");
    let nn_model: NeuralNet = NeuralNet::new();
    return Box::into_raw(Box::new(nn_model)) as *mut c_void;
}

// CUDA layers
#[no_mangle]
pub unsafe extern "C" fn add_dense_cuda_layer(
    vp: *mut c_void, n_in: usize, n_out: usize, batch: usize, rows: usize, use_bias: bool, id: *mut c_char
)
{
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    let name: &str = CStr::from_ptr(id).to_str().unwrap();
    (*nn).add_cuda_layer(
        name.to_string(), 
        Box::new(DenseCuda::new(
            n_in, n_out, batch, rows, use_bias, name
        )
    ));
}

#[no_mangle]
pub unsafe extern "C" fn add_activation_cuda_layer(
    vp: *mut c_void, batch: usize, rows: usize, cols: usize, activation: *mut c_char, scale: f32, id: *mut c_char
)
{
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    let id: &str = CStr::from_ptr(id).to_str().unwrap();
    let activation_str: &str = CStr::from_ptr(activation).to_str().unwrap();
    (*nn).add_cuda_layer(
        id.to_string(), 
        Box::new(ActivationCuda::new(activation_str, batch, rows, cols, scale))
    );
}

#[no_mangle]
pub unsafe extern "C" fn add_l2norm_cuda_layer(
    vp: *mut c_void, batch: usize, rows: usize, cols: usize, id: *mut c_char
)
{
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    let name: String = CStr::from_ptr(id).to_str().unwrap().to_string();
    (*nn).add_cuda_layer(
        name,
        Box::new(L2NormCuda::new(batch, rows, cols))
    );
}

#[no_mangle]
pub unsafe extern "C" fn add_embedding_2d_cuda_layer(
    vp: *mut c_void, vocab_size: usize, embedding_len: usize, seq_len: usize, id: *mut c_char
)
{
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    let name: String = CStr::from_ptr(id).to_str().unwrap().to_string();
    (*nn).add_cuda_layer(name, Box::new(
        Embedding2DCuda::new(
            vocab_size, embedding_len, seq_len
        )
    ));
}

#[no_mangle]
pub unsafe extern "C" fn add_conv2d_cuda_layer(
    vp: *mut c_void, n_filters: usize, filter_dim: usize, strides: usize, 
    batch: usize, rows: usize, cols: usize, flatten: bool, id: *mut c_char
)
{
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    let name: String = CStr::from_ptr(id).to_str().unwrap().to_string();
    (*nn).add_cuda_layer(
        name,
        Box::new(
            Conv2dCuda::new(
                n_filters, filter_dim, strides, batch, rows, cols,
                flatten, "conv2d"
            )
        )
    );
}

#[no_mangle]
pub unsafe extern "C" fn add_softmax_cuda_layer(
    vp: *mut c_void, batch: usize, rows: usize, cols: usize, temperature: f32, id: *mut c_char
)
{
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    let name: String = CStr::from_ptr(id).to_str().unwrap().to_string();
    (*nn).add_cuda_layer(name, Box::new(SoftmaxCuda::new(batch, rows, cols, temperature)));
}

#[no_mangle]
pub unsafe extern "C" fn add_dropout_cuda_layer(
    vp: *mut c_void, batch: usize, rows: usize, cols: usize, dropout_rate: f32, id: *mut c_char
)
{
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    let name: String = CStr::from_ptr(id).to_str().unwrap().to_string();
    (*nn).add_cuda_layer(name, Box::new(DropoutCuda::new(batch, rows, cols, dropout_rate, "dropout")));
}

#[no_mangle]
pub unsafe extern "C" fn add_elementwise_cuda_layer(
    vp: *mut c_void, batch: usize, rows: usize, cols: usize, scale_range: f32, 
    op: u32, dropout_rate: f32, id: *mut c_char, activation_str: *mut c_char, act_scale: f32
)
{
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    let name: String = CStr::from_ptr(id).to_str().unwrap().to_string();
    let activation_str: &str = CStr::from_ptr(activation_str).to_str().unwrap();

    let func_id: i32;
    match activation_str
    {
        "sigmoid" => func_id = 0,
        "tanh" => func_id = 1,
        "silu" => func_id = 2,
        "gelu" => func_id = 3,
        "tanh2" => func_id = 4,
        "softplus" => func_id = 5,
        "linear" => func_id = 6,
        _ => {println!("Invalid activation {:?}", activation_str); exit(1)},
    };

    (*nn).add_cuda_layer(name, Box::new(
        ElementwiseCuda::new(
            batch, rows, cols, scale_range,
            op, dropout_rate, func_id, act_scale
        )
    ));
}

#[no_mangle]
pub unsafe extern "C" fn add_broadcast_cuda_layer(
    vp: *mut c_void, out_batch: usize, out_rows: usize, out_cols: usize,
    axis: i32, id: *mut c_char
)
{
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    let name: String = CStr::from_ptr(id).to_str().unwrap().to_string();
    (*nn).add_cuda_layer(name, Box::new(BroadcastCuda::new(out_batch, out_rows, out_cols, axis)));
}

#[no_mangle]
pub unsafe extern "C" fn add_sum_cuda_layer(
    vp: *mut c_void, in_batch: usize, in_rows: usize, in_cols: usize,
    axis: i32, id: *mut c_char
)
{
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    let name: String = CStr::from_ptr(id).to_str().unwrap().to_string();
    (*nn).add_cuda_layer(name, Box::new(SumCuda::new(in_batch, in_rows, in_cols, axis)));
}

#[no_mangle]
pub unsafe extern "C" fn add_transpose_cuda_layer(
    vp: *mut c_void, batch: usize, rows: usize, cols: usize, id: *mut c_char
)
{
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    let name: String = CStr::from_ptr(id).to_str().unwrap().to_string();
    (*nn).add_cuda_layer(name, Box::new(BatchTransposeCuda::new(batch, rows, cols)));
}

#[no_mangle]
pub unsafe extern "C" fn add_cls_cuda_layer(
    vp: *mut c_void, batch: usize, rows: usize, cols: usize, token_idx: usize, id: *mut c_char
)
{
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    let name: String = CStr::from_ptr(id).to_str().unwrap().to_string();
    (*nn).add_cuda_layer(name, Box::new(CLSCuda::new(batch, rows, cols, token_idx)));
}

/////////////////////////////////////////////////////////////////////
#[no_mangle]
pub unsafe extern "C" fn pass_to_input(
    vp: *mut c_void, name: *mut c_char, array_ptr: *mut f32, array_len: usize
) -> *mut c_void
{
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    let name: String = CStr::from_ptr(name).to_str().unwrap().to_string();
    let traverse_ptr: *mut TraversePtrs = (*nn).pass_to_input(name, array_ptr, array_len);
    //println!("{:?}", traverse_ptr_str);
    return traverse_ptr as *mut c_void;
}

#[no_mangle]
pub unsafe extern "C" fn pass_to_output(
    vp: *mut c_void, name: *mut c_char, traverse_v_ptr: *mut c_void, array_len: usize
) -> *mut f32
{
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    let name: String = CStr::from_ptr(name).to_str().unwrap().to_string();
    let traverse_ptr: *mut TraversePtrs = traverse_v_ptr as *mut TraversePtrs;
    let array_output: *mut f32 = (*nn).pass_to_output(name, traverse_ptr, array_len);
    return array_output;
}

#[no_mangle]
pub unsafe extern "C" fn ce_loss(
    vp: *mut c_void, output_name: *mut c_char, target_classes: *mut f32,
    batch: usize, rows: usize, cols: usize
) -> *mut f32
{
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    let output_name: String = CStr::from_ptr(output_name).to_str().unwrap().to_string();
    let loss_vals: *mut f32 = (*nn).ce_loss_fn(output_name, target_classes, batch, rows, cols);

    return loss_vals;
}

#[no_mangle]
pub unsafe extern "C" fn pass_to_output_grad(
    vp: *mut c_void, name: *mut c_char, array_ptr: *mut f32, array_len: usize
)
{
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    let name: String = CStr::from_ptr(name).to_str().unwrap().to_string();
    (*nn).pass_to_output_grad(name, array_ptr, array_len);
}

#[no_mangle]
pub unsafe extern "C" fn forward(
    vp: *mut c_void, layer_id: *mut c_char, trav_in_v_ptr: *mut c_void, trav_weight_v_ptr: *mut c_void
) -> *mut c_void
{   
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    
    let trav_in_ptr: *mut TraversePtrs = trav_in_v_ptr as *mut TraversePtrs;
    let trav_weight_ptr: *mut TraversePtrs = trav_weight_v_ptr as *mut TraversePtrs;
    
    let layer_id: String = CStr::from_ptr(layer_id).to_str().unwrap().to_string();
    let traverse_ptr: *mut TraversePtrs = (*nn).forward(
        layer_id, trav_in_ptr, trav_weight_ptr
    );
    return traverse_ptr as *mut c_void;
}

// assuming output is always 1 dimensional
#[no_mangle]
pub unsafe extern "C" fn backward(
    vp: *mut c_void
)
{
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    (*nn).backward();
}

#[no_mangle]
pub unsafe extern "C" fn update_params(
    vp: *mut c_void, layer_id: *mut c_char, optimizer_type: i32, lr: f32, l2: f32, alpha: f32, beta: f32
)
{
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    let name: &str = CStr::from_ptr(layer_id).to_str().unwrap();
    (*nn).update_params(name, optimizer_type, lr, l2, alpha, beta);
}

#[no_mangle]
pub unsafe extern "C" fn set_dropout(
    vp: *mut c_void, use_dropout: bool
)
{
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    (*nn).set_dropout(use_dropout)
}

#[no_mangle]
pub unsafe extern "C" fn free_array(array_ptr: *mut f32, len: usize)
{
    let output_vec: Vec<f32> = Vec::from_raw_parts(array_ptr, len, len);
    std::mem::drop(output_vec);
}

#[no_mangle]
pub unsafe extern "C" fn free_nn(ptr: *mut c_void)
{
    let nn: *mut NeuralNet = ptr as *mut NeuralNet;
    let nn_boxed: Box<NeuralNet> = Box::from_raw(nn);
    std::mem::drop(nn_boxed);
}

#[no_mangle]
pub unsafe extern "C" fn loss(
    loss_fn: *mut c_char, array1: *mut f32, array2: *mut f32, flattened_len: usize) -> f32
{
    let loss_fn_str: &str = CStr::from_ptr(loss_fn).to_str().unwrap();

    let loss_function: LossFn = get_loss_from_str(loss_fn_str).unwrap();

    let vec1_flattened: Vec<f32> = std::slice::from_raw_parts_mut(array1, flattened_len).to_vec();
    let vec2_flattened: Vec<f32> = std::slice::from_raw_parts_mut(array2, flattened_len).to_vec();

    let array1: ArrayD<f32> = ArrayD::from_shape_vec(IxDyn(&[flattened_len]), vec1_flattened).unwrap();
    let array2: ArrayD<f32> = ArrayD::from_shape_vec(IxDyn(&[flattened_len]), vec2_flattened).unwrap();

    let loss_value: f32 = loss_function(array1, array2);

    return loss_value;
}

#[no_mangle]
pub unsafe extern "C" fn loss_grads(
    loss_fn: *mut c_char, array1: *mut f32, array2: *mut f32, flattened_len: usize) -> *mut f32
{
    let loss_fn_str: &str = CStr::from_ptr(loss_fn).to_str().unwrap();

    let loss_func_deriv: LossFnDeriv = get_loss_deriv_from_str(loss_fn_str).unwrap();

    //let output_shape: Vec<usize> = std::slice::from_raw_parts_mut(array_shape, array_shape_len).to_vec();
    let vec1_flattened: Vec<f32> = std::slice::from_raw_parts_mut(array1, flattened_len).to_vec();
    let vec2_flattened: Vec<f32> = std::slice::from_raw_parts_mut(array2, flattened_len).to_vec();

    let array1: ArrayD<f32> = ArrayD::from_shape_vec(IxDyn(&[flattened_len]), vec1_flattened).unwrap();
    let array2: ArrayD<f32> = ArrayD::from_shape_vec(IxDyn(&[flattened_len]), vec2_flattened).unwrap();

    let grads: ArrayD<f32> = loss_func_deriv(array1, array2);

    let mut grads_vec: Vec<f32> = grads.into_raw_vec();

    let grads_vec_ptr: *mut f32 = grads_vec.as_mut_ptr();

    std::mem::forget(grads_vec);
    return grads_vec_ptr;
}

#[no_mangle]
pub unsafe extern "C" fn details(ptr: *mut c_void)
{
    let nn: *mut NeuralNet = ptr as *mut NeuralNet;
    (*nn).details();
}

/*
#[no_mangle]
pub unsafe extern "C" fn save(ptr: *mut c_void, path: *mut c_char)
{
    let nn: *mut NeuralNet = ptr as *mut NeuralNet;
    let path: &str = CStr::from_ptr(path).to_str().unwrap();
    //save_model(path, &mut *nn, checkpoint);
}

#[no_mangle]
pub unsafe extern "C" fn load(path: *mut c_char) -> *mut c_void
{
    std::env::set_var("RUST_BACKTRACE", "full");
    let path: &str = CStr::from_ptr(path).to_str().unwrap();
    //let nn_struct: NeuralNet = load_model(path);

    let nn: *mut NeuralNet = Box::into_raw(Box::new(nn_struct));

    return nn as *mut c_void
}
*/

//////////////////////////////////////////////////////////////////
// for storage buffer

#[no_mangle]
pub unsafe extern "C" fn create_replay_buf(max_buffer_len: u64, max_online_queue_len: u64) -> *mut c_void
{
    let seq_buf: ReplayBuf = ReplayBuf::new(
        max_buffer_len as usize, max_online_queue_len as usize
    );
    return Box::into_raw(Box::new(seq_buf)) as *mut c_void;
}

#[no_mangle]
pub unsafe extern "C" fn add_state(
    seq_buf_v: *mut c_void,
    input: *mut f32, input_len: usize, 
    input_shape: *mut usize, input_shape_len: usize,
    output: *mut f32, output_len: usize, 
    output_shape: *mut usize, output_shape_len: usize,
    reward: f32, choice_type: f32
)
{
    let seq_buf: *mut ReplayBuf = seq_buf_v as *mut ReplayBuf;

    // convert the input pointer into a ndarray type
    let input_vec: Vec<f32> = std::slice::from_raw_parts_mut(input, input_len).to_vec();
    let input_shape_vec: &mut [usize] = std::slice::from_raw_parts_mut(input_shape, input_shape_len);
    let input_array: ArrayD<f32> = 
        ArrayD::from_shape_vec(IxDyn(input_shape_vec), input_vec).unwrap();
    
    // convert the output pointer into a ndarray type
    let output_vec: Vec<f32> = std::slice::from_raw_parts_mut(output, output_len).to_vec();
    let output_shape_vec: &mut [usize] = std::slice::from_raw_parts_mut(output_shape, output_shape_len);
    let output_array: ArrayD<f32> = 
        ArrayD::from_shape_vec(IxDyn(output_shape_vec), output_vec).unwrap();
    
    // pass input, corresponding output and reward to online buffer
    (*seq_buf).add_state(input_array, output_array, reward, choice_type);
}

#[no_mangle]
pub unsafe extern "C" fn get_state_input_at(seq_buf_v: *mut c_void, idx: usize) -> *mut f32
{
    let seq_buf: *mut ReplayBuf = seq_buf_v as *mut ReplayBuf;

    let array: ArrayD<f32> = (*seq_buf).get_state_input_at(idx);
    let mut array1d: Vec<f32> = array.into_raw_vec();
    let array_1d_ptr: *mut f32 = array1d.as_mut_ptr();

    std::mem::forget(array1d);

    return array_1d_ptr;
}

#[no_mangle]
pub unsafe extern "C" fn get_state_output_at(seq_buf_v: *mut c_void, idx: usize) -> *mut f32
{
    let seq_buf: *mut ReplayBuf = seq_buf_v as *mut ReplayBuf;

    let array: ArrayD<f32> = (*seq_buf).get_state_output_at(idx);
    let mut array1d: Vec<f32> = array.into_raw_vec();
    let array_1d_ptr: *mut f32 = array1d.as_mut_ptr();
    
    std::mem::forget(array1d);

    return array_1d_ptr;
}

#[no_mangle]
pub unsafe extern "C" fn reset_online_queue(seq_buf_v: *mut c_void)
{
    let seq_buf: *mut ReplayBuf = seq_buf_v as *mut ReplayBuf;
    (*seq_buf).reset_online_queue();
}

#[no_mangle]
pub unsafe extern "C" fn get_ave_rating(seq_buf_v: *mut c_void) -> f32
{
    let seq_buf: *mut ReplayBuf = seq_buf_v as *mut ReplayBuf;
    let ave: f32 = (*seq_buf).get_ave_rating();
    return ave;
}

#[no_mangle]
pub unsafe extern "C" fn get_count(seq_buf_v: *mut c_void) -> usize
{
    let seq_buf: *mut ReplayBuf = seq_buf_v as *mut ReplayBuf;
    let count: usize = (*seq_buf).get_count();
    return count;
}

//////////////////////////////////////////////////////////////////
// for ae buffer

#[no_mangle]
pub unsafe extern "C" fn create_ae_buf(max_len: u64, encoder_vp: *mut c_void, decoder_vp: *mut c_void) -> *mut c_void
{
    let encoder_ptr: *mut NeuralNet = encoder_vp as *mut NeuralNet;
    let decoder_ptr: *mut NeuralNet = decoder_vp as *mut NeuralNet;
    let ae_buf: AutoencoderBuf = AutoencoderBuf::new(
        max_len as usize, encoder_ptr, decoder_ptr
    );
    return Box::into_raw(Box::new(ae_buf)) as *mut c_void;
}

#[no_mangle]
pub unsafe extern "C" fn add_to_autoencoder_buf(
    ae_buf_v: *mut c_void,
    input: *mut f32, input_len: usize, input_shape: *mut usize, input_shape_len: usize)
{
    let ae_buf: *mut AutoencoderBuf = ae_buf_v as *mut AutoencoderBuf;

    // convert the input pointer into a ndarray type
    let input_vec: Vec<f32> = std::slice::from_raw_parts_mut(input, input_len).to_vec();
    let input_shape_vec: &mut [usize] = std::slice::from_raw_parts_mut(input_shape, input_shape_len);
    let input_array: ArrayD<f32> = 
        ArrayD::from_shape_vec(IxDyn(input_shape_vec), input_vec).unwrap();
    
    // pass input, corresponding output and reward to online buffer
    (*ae_buf).add_new_array(input_array);
}