//! Main file of the Rust backend Machine Learning project
//! 
//! Contains functions that can be called by the Python frontend to
//! facilitate data exchange and method calls for neural nets

use std::{ffi::{c_char, CStr}, os::raw::c_void, process::exit};

use layers::{
    activation_cuda::ActivationCuda, broadcast_cuda::BroadcastCuda, 
    cls_cuda::CLSCuda, conv2d_cuda::Conv2dCuda, dense_cuda::DenseCuda, 
    dropout_cuda::DropoutCuda, elementwise_cuda::ElementwiseCuda, 
    embedding_cuda::Embedding2DCuda, l2norm_cuda::L2NormCuda, 
    softmax_cuda::SoftmaxCuda, sum_cuda::SumCuda, transpose_cuda::BatchTransposeCuda};
    
use math_functions::{get_loss_deriv_from_str, get_loss_from_str};
use ndarray::{ArrayD, IxDyn};
use neuralnet::{NeuralNet, TraversePtrs};
use types::{LossFn, LossFnDeriv};

mod neuralnet;
mod math_functions;
mod types;
mod layers;
mod cuda_bridge;
mod pointer_ops;

// create a neural net and returns it as a void pointer to Python frontend
#[no_mangle]
pub unsafe extern "C" fn create_model() -> *mut c_void
{
    std::env::set_var("RUST_BACKTRACE", "full");
    let nn_model: NeuralNet = NeuralNet::new();
    return Box::into_raw(Box::new(nn_model)) as *mut c_void;
}

// ---------------------------------------------------------------
// Functions to create layers in neural nets given the void pointer 
// (called by Python frontend), all return layer names 
// to the Python frontend, with a number added in front of them to prevent
// duplication
// Layer names returned are important for forward and gradient descent calls

#[no_mangle]
pub unsafe extern "C" fn add_dense_cuda_layer(
    vp: *mut c_void, n_in: usize, n_out: usize, batch: usize, rows: usize, use_bias: bool, id: *mut c_char
) -> *mut c_char
{
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    let name: &str = CStr::from_ptr(id).to_str().unwrap();
    let n_layers: usize = (*nn).all_cuda_layers.len();
    let new_name: String = n_layers.to_string() + "_" + name;
    (*nn).add_cuda_layer(
        new_name.clone(), 
        Box::new(DenseCuda::new(
            n_in, n_out, batch, rows, use_bias, name
        )
    ));

    let mut new_name_terminated: String = new_name + "\0";
    return new_name_terminated.as_mut_ptr() as *mut c_char;
}

#[no_mangle]
pub unsafe extern "C" fn add_activation_cuda_layer(
    vp: *mut c_void, batch: usize, rows: usize, cols: usize, activation: *mut c_char, scale: f32, id: *mut c_char
) -> *mut c_char
{
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    let name: &str = CStr::from_ptr(id).to_str().unwrap();
    let activation_str: &str = CStr::from_ptr(activation).to_str().unwrap();

    let n_layers: usize = (*nn).all_cuda_layers.len();
    let new_name: String = n_layers.to_string() + "_" + name;
    (*nn).add_cuda_layer(
        new_name.clone(), 
        Box::new(ActivationCuda::new(activation_str, batch, rows, cols, scale))
    );

    let mut new_name_terminated: String = new_name + "\0";
    return new_name_terminated.as_mut_ptr() as *mut c_char;
}

#[no_mangle]
pub unsafe extern "C" fn add_l2norm_cuda_layer(
    vp: *mut c_void, batch: usize, rows: usize, cols: usize, id: *mut c_char
) -> *mut c_char
{
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    let name: &str = CStr::from_ptr(id).to_str().unwrap();
    let n_layers: usize = (*nn).all_cuda_layers.len();
    let new_name: String = n_layers.to_string() + "_" + name;
    (*nn).add_cuda_layer(
        new_name.clone(),
        Box::new(L2NormCuda::new(batch, rows, cols))
    );

    let mut new_name_terminated: String = new_name + "\0";
    return new_name_terminated.as_mut_ptr() as *mut c_char;
}

#[no_mangle]
pub unsafe extern "C" fn add_embedding_2d_cuda_layer(
    vp: *mut c_void, vocab_size: usize, embedding_len: usize, seq_len: usize, id: *mut c_char
) -> *mut c_char
{
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    let name: &str = CStr::from_ptr(id).to_str().unwrap();
    let n_layers: usize = (*nn).all_cuda_layers.len();
    let new_name: String = n_layers.to_string() + "_" + name;
    (*nn).add_cuda_layer(
        new_name.clone(),
        Box::new(
        Embedding2DCuda::new(
            vocab_size, embedding_len, seq_len
        )
    ));

    let mut new_name_terminated: String = new_name + "\0";
    return new_name_terminated.as_mut_ptr() as *mut c_char;
}

#[no_mangle]
pub unsafe extern "C" fn add_conv2d_cuda_layer(
    vp: *mut c_void, n_filters: usize, filter_dim: usize, strides: usize, 
    batch: usize, rows: usize, cols: usize, id: *mut c_char
) -> *mut c_char
{
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    let name: &str = CStr::from_ptr(id).to_str().unwrap();
    let n_layers: usize = (*nn).all_cuda_layers.len();
    let new_name: String = n_layers.to_string() + "_" + name;
    (*nn).add_cuda_layer(
        new_name.clone(),
        Box::new(
            Conv2dCuda::new(
                n_filters, filter_dim, strides, batch, rows, cols,
                "conv2d"
            )
        )
    );

    let mut new_name_terminated: String = new_name + "\0";
    return new_name_terminated.as_mut_ptr() as *mut c_char;
}

#[no_mangle]
pub unsafe extern "C" fn add_softmax_cuda_layer(
    vp: *mut c_void, batch: usize, rows: usize, cols: usize, temperature: f32, id: *mut c_char
) -> *mut c_char
{
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    let name: &str = CStr::from_ptr(id).to_str().unwrap();
    let n_layers: usize = (*nn).all_cuda_layers.len();
    let new_name: String = n_layers.to_string() + "_" + name;
    (*nn).add_cuda_layer(
        new_name.clone(),
        Box::new(SoftmaxCuda::new(batch, rows, cols, temperature)
    ));

    let mut new_name_terminated: String = new_name + "\0";
    return new_name_terminated.as_mut_ptr() as *mut c_char;
}

#[no_mangle]
pub unsafe extern "C" fn add_dropout_cuda_layer(
    vp: *mut c_void, batch: usize, rows: usize, cols: usize, dropout_rate: f32, id: *mut c_char
) -> *mut c_char
{
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    let name: &str = CStr::from_ptr(id).to_str().unwrap();
    let n_layers: usize = (*nn).all_cuda_layers.len();
    let new_name: String = n_layers.to_string() + "_" + name;
    (*nn).add_cuda_layer(
        new_name.clone(),
        Box::new(DropoutCuda::new(batch, rows, cols, dropout_rate, "dropout"))
    );

    let mut new_name_terminated: String = new_name + "\0";
    return new_name_terminated.as_mut_ptr() as *mut c_char;
}

#[no_mangle]
pub unsafe extern "C" fn add_elementwise_cuda_layer(
    vp: *mut c_void, batch: usize, rows: usize, cols: usize, scale_range: f32, 
    op: u32, dropout_rate: f32, id: *mut c_char, activation_str: *mut c_char, act_scale: f32
) -> *mut c_char
{
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    let name: &str = CStr::from_ptr(id).to_str().unwrap();
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

    let n_layers: usize = (*nn).all_cuda_layers.len();
    let new_name: String = n_layers.to_string() + "_" + name;
    (*nn).add_cuda_layer(
        new_name.clone(),Box::new(
        ElementwiseCuda::new(
            batch, rows, cols, scale_range,
            op, dropout_rate, func_id, act_scale
        )
    ));

    let mut new_name_terminated: String = new_name + "\0";
    return new_name_terminated.as_mut_ptr() as *mut c_char;
}

#[no_mangle]
pub unsafe extern "C" fn add_broadcast_cuda_layer(
    vp: *mut c_void, out_batch: usize, out_rows: usize, out_cols: usize,
    axis: i32, id: *mut c_char
) -> *mut c_char
{
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    let name: &str = CStr::from_ptr(id).to_str().unwrap();
    let n_layers: usize = (*nn).all_cuda_layers.len();
    let new_name: String = n_layers.to_string() + "_" + name;
    (*nn).add_cuda_layer(
        new_name.clone(),
        Box::new(BroadcastCuda::new(out_batch, out_rows, out_cols, axis)
    ));

    let mut new_name_terminated: String = new_name + "\0";
    return new_name_terminated.as_mut_ptr() as *mut c_char;
}

#[no_mangle]
pub unsafe extern "C" fn add_sum_cuda_layer(
    vp: *mut c_void, in_batch: usize, in_rows: usize, in_cols: usize,
    axis: i32, id: *mut c_char
) -> *mut c_char
{
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    let name: &str = CStr::from_ptr(id).to_str().unwrap();
    let n_layers: usize = (*nn).all_cuda_layers.len();
    let new_name: String = n_layers.to_string() + "_" + name;
    (*nn).add_cuda_layer(
        new_name.clone(),
        Box::new(SumCuda::new(in_batch, in_rows, in_cols, axis)
    ));

    let mut new_name_terminated: String = new_name + "\0";
    return new_name_terminated.as_mut_ptr() as *mut c_char;
}

#[no_mangle]
pub unsafe extern "C" fn add_transpose_cuda_layer(
    vp: *mut c_void, batch: usize, rows: usize, cols: usize, id: *mut c_char
) -> *mut c_char
{
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    let name: &str = CStr::from_ptr(id).to_str().unwrap();
    let n_layers: usize = (*nn).all_cuda_layers.len();
    let new_name: String = n_layers.to_string() + "_" + name;
    (*nn).add_cuda_layer(
        new_name.clone(),
        Box::new(BatchTransposeCuda::new(batch, rows, cols)
    ));

    let mut new_name_terminated: String = new_name + "\0";
    return new_name_terminated.as_mut_ptr() as *mut c_char;
}

#[no_mangle]
pub unsafe extern "C" fn add_cls_cuda_layer(
    vp: *mut c_void, batch: usize, rows: usize, cols: usize, token_idx: usize, id: *mut c_char
) -> *mut c_char
{
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    let name: &str = CStr::from_ptr(id).to_str().unwrap();
    let n_layers: usize = (*nn).all_cuda_layers.len();
    let new_name: String = n_layers.to_string() + "_" + name;
    (*nn).add_cuda_layer(
        new_name.clone(),
        Box::new(CLSCuda::new(batch, rows, cols, token_idx)
    ));

    let mut new_name_terminated: String = new_name + "\0";
    return new_name_terminated.as_mut_ptr() as *mut c_char;
}

// ---------------------------------------------------------------

// Accepts numpy array from Python frontend, returns traverse pointer
// casted to void pointer
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

// Accepts traverse pointer casted to void pointer from Python frontend and 
// return float pointer which is converted to a numpy array
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

// accepts target classes as an array of integers casted to float 
// (array length equals to softmax rows) and returns loss values as float pointer
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

// accepts numpy array of gradients
#[no_mangle]
pub unsafe extern "C" fn pass_to_output_grad(
    vp: *mut c_void, name: *mut c_char, array_ptr: *mut f32, array_len: usize
)
{
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    let name: String = CStr::from_ptr(name).to_str().unwrap().to_string();
    (*nn).pass_to_output_grad(name, array_ptr, array_len);
}

// forward function to be called by Python frontend
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

// Python frontend calls backpropagation method
#[no_mangle]
pub unsafe extern "C" fn backward(
    vp: *mut c_void
)
{
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    (*nn).backward();
}

// Called by Python frontend to perform gradient descent on a specified layer
// given ID
#[no_mangle]
pub unsafe extern "C" fn update_params(
    vp: *mut c_void, layer_id: *mut c_char, optimizer_type: i32, lr: f32, l2: f32, alpha: f32, beta: f32
)
{
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    let name: &str = CStr::from_ptr(layer_id).to_str().unwrap();
    (*nn).update_params(name, optimizer_type, lr, l2, alpha, beta);
}

// Set whether dropout should be used during forward and backward pass
#[no_mangle]
pub unsafe extern "C" fn set_dropout(
    vp: *mut c_void, use_dropout: bool
)
{
    let nn: *mut NeuralNet = vp as *mut NeuralNet;
    (*nn).set_dropout(use_dropout)
}

// Host side function to calculate the loss between two numpy arrays
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

// Host side function to calculate the loss gradient between two numpy arrays
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

// Function to call by Python frontend to print model details
#[no_mangle]
pub unsafe extern "C" fn details(ptr: *mut c_void)
{
    let nn: *mut NeuralNet = ptr as *mut NeuralNet;
    (*nn).details();
}

// Save to JSON function for Python frontend
#[no_mangle]
pub unsafe extern "C" fn save(ptr: *mut c_void, path: *mut c_char)
{
    let nn: *mut NeuralNet = ptr as *mut NeuralNet;
    let path: &str = CStr::from_ptr(path).to_str().unwrap();
    (*nn).save(path);
}

// Load from JSON function for Python frontend
#[no_mangle]
pub unsafe extern "C" fn load(ptr: *mut c_void, path: *mut c_char)
{
    let path: &str = CStr::from_ptr(path).to_str().unwrap();
    let nn: *mut NeuralNet = ptr as *mut NeuralNet;
    (*nn).load(path);
}

// Free memory function for Python frontend
#[no_mangle]
pub unsafe extern "C" fn delete(ptr: *mut c_void)
{
    let nn: *mut NeuralNet = ptr as *mut NeuralNet;
    (*nn).delete();

    // free the neural net object itself
    let _ = Box::from_raw(nn);
}

// Free raw Vec<f32> pointer
#[no_mangle]
pub unsafe extern "C" fn free_raw_vector(ptr: *mut f32, length: usize)
{
    // raw pointer is consumed by vector type, automatically dropped/freed
    // when function ends
    let _vector: Vec<f32> = Vec::from_raw_parts(ptr, length, length);
}