use std::{os::raw::c_void, process::Stdio};

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

    pub input_ptr: *mut f32,
    pub input_grad_ptr: *mut f32,

    pub output_ptr: *mut f32,
    pub output_grad_ptr: *mut f32,

    pub output_traverse_ptr: Option<Box<TraversePtrs>>,
    pub backward_count: Option<Box<f32>>,
    pub backward_count_weight_prev: Option<Box<f32>>,
    pub backward_count_in_prev: Option<Box<f32>>,
}
impl IOPtrs
{
    pub fn new(in_shape: (usize, usize, usize), out_shape: (usize, usize, usize)) -> Self
    {
        return Self
        {
            in_shape, out_shape,
            input_ptr: std::ptr::null_mut(),
            input_grad_ptr: std::ptr::null_mut(),
            output_ptr: std::ptr::null_mut(),
            output_grad_ptr: std::ptr::null_mut(),
            output_traverse_ptr: None,
            backward_count: None,
            backward_count_weight_prev: None,
            backward_count_in_prev: None,
        }
    }
}


pub struct ParameterPtrs
{
    pub weight_ptr: *mut f32,
    pub weight_grad_ptr: *mut f32,
    pub weight_vel_ptr: *mut f32,
    pub weight_moment_ptr: *mut f32,

    pub biases_ptr: *mut f32,
    pub bias_grad_ptr: *mut f32,
    pub bias_vel_ptr: *mut f32,
    pub bias_moment_ptr: *mut f32,
}
impl ParameterPtrs
{
    pub fn new() -> Self
    {
        return Self
        {
            weight_ptr: std::ptr::null_mut(),
            weight_grad_ptr: std::ptr::null_mut(),
            weight_vel_ptr: std::ptr::null_mut(),
            weight_moment_ptr: std::ptr::null_mut(),
        
            biases_ptr: std::ptr::null_mut(),
            bias_grad_ptr: std::ptr::null_mut(),
            bias_vel_ptr: std::ptr::null_mut(),
            bias_moment_ptr: std::ptr::null_mut(),
        }
    }
}

pub struct AllocationStatus
{
    pub ptrs_allocated: bool, 
    pub arrays_allocated: bool,
        
    pub zero_output: bool,
    pub zero_input_grad: bool,
    pub zero_weight_grad: bool
}
impl AllocationStatus
{
    pub fn new() -> Self
    {
        return Self
        {
            ptrs_allocated: false,
            arrays_allocated: false,

            zero_output: true,
            zero_input_grad: false,
            zero_weight_grad: false
        }
    }
}

pub struct WeightTensors
{
    pub weight: Vec<f32>,
    pub biases: Vec<f32>,
}
impl WeightTensors
{
    pub fn new() -> Self
    {
        return Self
        {
            weight: Vec::new(),
            biases: Vec::new()
        }
    }
}