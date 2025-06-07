use crate::{
    cuda_bridge::{activation3d_cuda, activation3d_cuda_backward}, 
    neuralnet::TraversePtrs, 
    pointer_ops::{counter_is_zero, increment_counter, init_trav_in_ptrs, set_zero_counter}
};

use super::layer_cuda::{AllocationStatus, IOPtrs, LayerCuda};

/// Activation layers play an important role in introducing
/// nonlinearity in a model, increases model expressiveness,
/// no trainable parameters
pub struct ActivationCuda
{
    pub io_ptrs: IOPtrs,
    pub allocation_status: AllocationStatus,

    pub scale: f32,
    pub activation_str: String,
    pub batch_size: f32,
}
impl ActivationCuda
{
    pub fn new(activation_str: &str, batch: usize, rows: usize, cols: usize, scale: f32) -> Self
    {
        return Self {
            io_ptrs: IOPtrs::new((batch, rows, cols), (batch, rows, cols)),
            allocation_status: AllocationStatus::new(),
            activation_str: activation_str.to_string(),
            scale,
            batch_size: 0.0,
        }
    }
}

impl LayerCuda for ActivationCuda
{
    fn forward(&mut self, trav_ptr_in: *mut TraversePtrs, _trav_ptr_weight: *mut TraversePtrs, _use_dropout: bool) -> *mut TraversePtrs
    {       
        if !self.allocation_status.ptrs_allocated
        {
            init_trav_in_ptrs(
                &trav_ptr_in, &mut self.io_ptrs.backward_count,
                &mut self.io_ptrs.backward_count_in_prev, 
                &mut self.io_ptrs.input_ptr, &mut self.io_ptrs.input_grad_ptr, 
                &mut self.io_ptrs.output_ptr, &mut self.io_ptrs.output_grad_ptr, 
                &mut self.io_ptrs.output_traverse_ptr, 
                (self.io_ptrs.in_shape.0 * self.io_ptrs.in_shape.1 * self.io_ptrs.in_shape.2) as usize
            );

            self.allocation_status.ptrs_allocated = true;
            self.allocation_status.arrays_allocated = true;
        }

        activation3d_cuda(
            self.io_ptrs.output_ptr, self.io_ptrs.input_ptr, 
            self.io_ptrs.in_shape.0 as u32, self.io_ptrs.in_shape.1 as u32, self.io_ptrs.in_shape.2 as u32, 
            &self.activation_str, self.scale, self.allocation_status.zero_output
        );
        set_zero_counter(self.io_ptrs.backward_count);

        return self.io_ptrs.output_traverse_ptr;
    }

    fn backward(&mut self, _use_dropout: bool)
    {
        if !counter_is_zero(self.io_ptrs.backward_count_in_prev)
        {
            self.allocation_status.zero_input_grad = false;
        }
        activation3d_cuda_backward(
            self.io_ptrs.input_grad_ptr, self.io_ptrs.input_ptr, 
            self.io_ptrs.output_grad_ptr, 
            self.io_ptrs.in_shape.0 as u32, self.io_ptrs.in_shape.1 as u32, 
            self.io_ptrs.in_shape.2 as u32, 
            &self.activation_str, self.scale, self.allocation_status.zero_input_grad
        );
        increment_counter(self.io_ptrs.backward_count_in_prev);

        self.batch_size += 1.0;
    }

    fn update_params(&mut self, _optimizer_type: i32, _lr: f32, _l2: f32, _alpha: f32, _beta: f32)
    { 
        self.batch_size = 0.0;
    }

    fn details(&self)
    {
        println!("Layer type: ACTIVATION");
        println!("Activation function: {:?}", self.activation_str);
        println!("Input ptr: {:?} | Input grad ptr: {:?}", self.io_ptrs.input_ptr, self.io_ptrs.input_grad_ptr);
        println!("Output ptr: {:?} | Output grad ptr: {:?}", self.io_ptrs.output_ptr, self.io_ptrs.output_grad_ptr);
        println!("Shape: {:?}", self.io_ptrs.in_shape);
    }
}