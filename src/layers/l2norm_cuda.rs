use std::ffi::c_void;

use crate::{
    cuda_bridge::{free_cuda_array, l2norm_backward, l2norm_forward, new_cuda_array}, 
    neuralnet::TraversePtrs, 
    pointer_ops::{counter_is_zero, increment_counter, init_trav_in_ptrs, set_zero_counter}};

use super::layer_cuda::{AllocationStatus, IOPtrs, LayerCuda};

/// Perform L2 or Euclidean norm for input arrays, normalizes  
/// rows to a length of 1 and remove magnitude while maintaining directional info
pub struct L2NormCuda
{
    pub io_ptrs: IOPtrs,
    pub allocation_status: AllocationStatus,

    pub input_pow2_ptr: *mut f32,
    pub power_sum: *mut f32,
    pub batch_size: f32,
}

// implement constructor
impl L2NormCuda
{
    pub fn new(batch: usize, rows: usize, cols: usize) -> Self
    {
        return Self
        {
            input_pow2_ptr: std::ptr::null_mut(),
            power_sum: std::ptr::null_mut(),
            io_ptrs: IOPtrs::new((batch, rows, cols),(batch, rows, cols)),
            allocation_status: AllocationStatus::new(),
            batch_size: 0.0,
        }
    }
}

// trait implementation
impl LayerCuda for L2NormCuda
{
    fn forward(&mut self, trav_ptr_in: *mut TraversePtrs, _trav_ptr_weight: *mut TraversePtrs, _use_dropout: bool) -> *mut TraversePtrs
    {
        let batch: usize = self.io_ptrs.in_shape.0;
        let rows: usize = self.io_ptrs.in_shape.1;
        let cols: usize = self.io_ptrs.in_shape.2;

        if !self.allocation_status.ptrs_allocated
        {   
            // contains the squared values of all values in the input
            self.input_pow2_ptr = new_cuda_array((batch * rows * cols) as u32);
            
            // contains the sum of the squared values in each row, the L2Norm values
            self.power_sum = new_cuda_array((batch * rows * 1) as u32);

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

        // CUDA function to normalize input tensor using L2Norm
        l2norm_forward(
            self.io_ptrs.input_ptr, self.input_pow2_ptr, 
            batch, rows, cols, 
            self.power_sum, self.io_ptrs.output_ptr, true
        );

        // important for zeroing gradients during backward pass
        set_zero_counter(self.io_ptrs.backward_count);

        return self.io_ptrs.output_traverse_ptr;
    }

    fn backward(&mut self, _use_dropout: bool)
    {
        // only zeros input gradients when the counter in the previous layer is zero
        if !counter_is_zero(self.io_ptrs.backward_count_in_prev)
        {
            self.allocation_status.zero_input_grad = false;
        }

        // CUDA function to perform backpropagation through L2Norm
        l2norm_backward(
            self.io_ptrs.output_grad_ptr, 
            self.io_ptrs.in_shape.0, self.io_ptrs.in_shape.1, self.io_ptrs.in_shape.2, 
            self.power_sum, self.io_ptrs.input_ptr, self.io_ptrs.output_ptr, 
            self.io_ptrs.input_grad_ptr, self.allocation_status.zero_input_grad
        );
        self.batch_size += 1.0;

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
        println!("Layer type: L2Norm");
        println!("Input Shape: {:?}", self.io_ptrs.in_shape);
        println!("Input ptr: {:?} | Input grad ptr: {:?}", self.io_ptrs.input_ptr, self.io_ptrs.input_grad_ptr);
        println!("Output ptr: {:?} | Output grad ptr: {:?}", self.io_ptrs.output_ptr, self.io_ptrs.output_grad_ptr);
    }

    fn free_detached_ptrs(&self) 
    {
        free_cuda_array(self.input_pow2_ptr as *mut c_void);
        free_cuda_array(self.power_sum as *mut c_void);
    }
}