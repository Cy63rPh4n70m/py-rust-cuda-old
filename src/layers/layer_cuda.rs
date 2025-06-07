use std::collections::HashMap;

use crate::neuralnet::TraversePtrs;

// LayerCuda trait is important for all layer implementations
// contains methods that all layers must implement, simulate inheritance
pub trait LayerCuda
{
    fn forward(
        &mut self, inputs: *mut TraversePtrs, weights: *mut TraversePtrs, use_dropout: bool
    ) -> *mut TraversePtrs;
    fn backward(&mut self, use_dropout: bool);
    fn update_params(&mut self, optimizer_type: i32, lr: f32, l2: f32, alpha: f32, beta: f32);
    fn details(&self);

    // default implementations
    fn free_detached_ptrs(&self) {}
    fn get_param_count(&self) -> usize { return 0_usize; }
    fn move_ptrs_to_arrays(&mut self) {}
    fn get_weights_hashmap(&mut self) -> Option<HashMap<&str, Vec<f32>>> { return None; }
    fn load_weights_from_hashmap(&mut self, _hashmap: &HashMap<&str, Vec<f32>>) {}
}

// ----------------------------------------------------------
// composition structs are used to reduce code repetition
// used along with LayerCuda trait to simulate inheritance

// - contains pointers involved with transferring data between layers
// - includes input/output shape for layers
pub struct IOPtrs
{
    pub in_shape: (usize, usize, usize),
    pub out_shape: (usize, usize, usize),

    pub input_ptr: *mut f32,
    pub input_grad_ptr: *mut f32,

    pub output_ptr: *mut f32,
    pub output_grad_ptr: *mut f32,

    pub output_traverse_ptr: *mut TraversePtrs,
    pub backward_count: *mut usize,
    pub backward_count_weight_prev: *mut usize,
    pub backward_count_in_prev: *mut usize,
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
            output_traverse_ptr: std::ptr::null_mut(),
            backward_count: std::ptr::null_mut(),
            backward_count_weight_prev: std::ptr::null_mut(),
            backward_count_in_prev: std::ptr::null_mut(),
        }
    }
}

// includes the pointers for layer weights (if applicable), 
// along with pointers required during training of the weights
// using AdamW optimizer
pub struct ParameterPtrs
{
    pub weight_ptr: *mut f32,
    pub weight_grad_ptr: *mut f32,
    pub weight_vel_ptr: *mut f32,
    pub weight_moment_ptr: *mut f32,

    pub weight_ptr_detached: bool,

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

            weight_ptr_detached: true,
        
            biases_ptr: std::ptr::null_mut(),
            bias_grad_ptr: std::ptr::null_mut(),
            bias_vel_ptr: std::ptr::null_mut(),
            bias_moment_ptr: std::ptr::null_mut(),
        }
    }
}

// keeps track of whether pointers has been allocated in layers to prevent
// repeated memory allocations during forward and backward passes
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

// stores weights in vector types, required for saving model state
// to JSON and vice-versa
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