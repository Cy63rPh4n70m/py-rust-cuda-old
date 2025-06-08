use std::os::raw::c_void;

use crate::{
    cuda_bridge::{copy_cuda_to_cuda, dropout_backward, dropout_forward, free_cuda_array, init_random_states, new_cuda_array}, 
    neuralnet::TraversePtrs, 
    pointer_ops::{increment_counter, init_trav_in_ptrs, set_zero_counter}};

use super::layer_cuda::{AllocationStatus, IOPtrs, LayerCuda};

/// Performs random dropout on input array,
/// useful for mitigating overfitting during model training
pub struct DropoutCuda
{
    pub name: String,
    pub io_ptrs: IOPtrs,
    pub allocation_status: AllocationStatus,

    pub dropout_mask_ptr: *mut f32,
    pub rand_state_v_ptr: *mut c_void,
    pub dropout_rate: f32,
    pub batch_size: f32,
}

// implement constructor
impl DropoutCuda
{
    pub fn new(batch: usize, rows: usize, cols: usize, dropout_rate: f32, name: &str) -> Self
    {
        return Self
        {
            name: name.to_string(),
            dropout_rate,

            io_ptrs: IOPtrs::new((batch, rows, cols), (batch, rows, cols)),
            allocation_status: AllocationStatus::new(),
            
            dropout_mask_ptr: std::ptr::null_mut(),
            rand_state_v_ptr: std::ptr::null_mut(),
            batch_size: 0.0,
        }
    }
}

// trait implementation
impl LayerCuda for DropoutCuda
{
    fn forward(&mut self, trav_ptr_in: *mut TraversePtrs, _trav_ptr_weight: *mut TraversePtrs, use_dropout: bool) -> *mut TraversePtrs
    {
        let batch: usize = self.io_ptrs.in_shape.0;
        let rows: usize = self.io_ptrs.in_shape.1;
        let cols: usize = self.io_ptrs.in_shape.2;

        if !self.allocation_status.ptrs_allocated
        {   
            // initialize dropout mask pointer
            self.dropout_mask_ptr = new_cuda_array((batch * rows * cols) as u32);

            // initialize the random states pointer
            self.rand_state_v_ptr = init_random_states(batch, rows, cols);
            
            // connect the output pointers of the previous layer with the 
            // current layer's input pointers
            init_trav_in_ptrs(
                &trav_ptr_in, &mut self.io_ptrs.backward_count,
                &mut self.io_ptrs.backward_count_in_prev, 
                &mut self.io_ptrs.input_ptr, &mut self.io_ptrs.input_grad_ptr, 
                &mut self.io_ptrs.output_ptr, &mut self.io_ptrs.output_grad_ptr, 
                &mut self.io_ptrs.output_traverse_ptr, 
                (self.io_ptrs.out_shape.0 * self.io_ptrs.out_shape.1 * self.io_ptrs.out_shape.2) as usize
            );

            self.allocation_status.ptrs_allocated = true;
            self.allocation_status.arrays_allocated = true;
        }
        
        if use_dropout
        {
            // CUDA function for dropout, multiplies input with randomized binary dropout
            // to zero inputs
            dropout_forward(
                self.io_ptrs.input_ptr, self.io_ptrs.output_ptr, self.dropout_mask_ptr, 
                self.rand_state_v_ptr,
                self.io_ptrs.in_shape.0, self.io_ptrs.in_shape.1, 
                self.io_ptrs.in_shape.2, 
                self.dropout_rate
            );
        }
        else
        {
            // just copy array memory if dropout disabled
            copy_cuda_to_cuda(
                self.io_ptrs.output_ptr, self.io_ptrs.input_ptr, 
                &[self.io_ptrs.in_shape.0, self.io_ptrs.in_shape.1, self.io_ptrs.in_shape.2]
            );
        }

        // important for zeroing gradients during backward pass
        set_zero_counter(self.io_ptrs.backward_count);

        return self.io_ptrs.output_traverse_ptr;

    }

    fn backward(&mut self, use_dropout: bool)
    {
        if use_dropout
        {
            // CUDA function for backpropagation through dropout, same as
            // elementwise multiplication gradient calculation
            dropout_backward(
                self.io_ptrs.output_grad_ptr, self.dropout_mask_ptr, 
                self.io_ptrs.input_grad_ptr,
                self.io_ptrs.in_shape.0, self.io_ptrs.in_shape.1, 
                self.io_ptrs.in_shape.2
            );

            self.batch_size += 1.0;
        }
        else
        {
            // just copy gradients if disabled dropout
            copy_cuda_to_cuda(
                self.io_ptrs.input_grad_ptr, self.io_ptrs.output_grad_ptr, 
                &[self.io_ptrs.in_shape.0, self.io_ptrs.in_shape.1, self.io_ptrs.in_shape.2]);
        }

        // increment counter to tell other layers connected to the same previous layer
        // to accumulate the gradient instead of zeroing it first
        increment_counter(self.io_ptrs.backward_count_in_prev);
    }

    fn update_params(&mut self, _optimizer_type: i32, _lr: f32, _l2: f32, _alpha: f32, _beta: f32)
    {    
        self.batch_size = 0.0;
    }

    fn details(&self)
    {
        // print all details of layer (e.g. IO shape, weights, etc)
        println!("Layer type: DROPOUT | Layer name: {:?}", self.name);
        println!("Shape: {:?}", self.io_ptrs.in_shape);
        println!("Dropout rate: {:?}", self.dropout_rate);
        println!("Input ptr: {:?} | Input grad ptr: {:?}", self.io_ptrs.input_ptr, self.io_ptrs.input_grad_ptr);
        println!("Output ptr: {:?} | Output grad ptr: {:?}", self.io_ptrs.output_ptr, self.io_ptrs.output_grad_ptr);
    }

    fn free_detached_ptrs(&self) 
    {
        free_cuda_array(self.rand_state_v_ptr);
        free_cuda_array(self.dropout_mask_ptr as *mut c_void);
    }
}