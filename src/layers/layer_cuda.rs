use std::os::raw::c_void;

use crate::neuralnet::TraversePtrs;

trait LayerCuda
{
    fn forward(
        &mut self, inputs: *mut c_void, weights: *mut c_void, use_dropout: bool
    ) -> *mut c_void;

    fn backward(&mut self);
    fn update_params(&mut self);
    fn details(&mut self);
    fn get_param_count(&self) -> f32;
    fn move_ptrs_to_arrays(&mut self);
}

// composition structs to reduce code repetition
pub struct IOPtrs
{
    pub in_shape: (usize, usize, usize),
    pub out_shape: (usize, usize, usize),

    pub output_ptr: *mut f32,
    pub output_grad_ptr: *mut f32,

    pub input_ptr: *mut f32,
    pub input_grad_ptr: *mut f32,

    pub output_traverse_ptr: *mut TraversePtrs,
}

pub struct ParameterPtrs
{
    pub biases_ptr: *mut f32,
    pub bias_grad_ptr: *mut f32,
    pub bias_vel_ptr: *mut f32,
    pub bias_moment_ptr: *mut f32,

    pub weight_ptr: *mut f32,
    pub weight_grad_ptr: *mut f32,
    pub weight_vel_ptr: *mut f32,
    pub weight_moment_ptr: *mut f32,
}

pub struct AllocationStatus
{
    pub weight_ptr_allocated: bool, 
    pub bias_ptr_allocated: bool, 
    pub weight_array_allocated: bool,
    pub bias_array_allocated: bool,
    
    pub use_bias: bool,
    
    pub zero_output: bool,
    pub zero_input_grad: bool,
    pub zero_weight_grad: bool
}

pub struct MiscData
{
    pub backward_count: Box<f32>,
    pub backward_count_weight_prev: Box<f32>,
    pub backward_count_in_prev: Box<f32>,

    pub batch_size: f32,
    pub count: u128,
}

pub struct WeightTensors
{
    pub weight: Vec<f32>,
    pub biases: Vec<f32>,
}