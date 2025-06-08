
use std::ffi::c_void;

use crate::{
    cuda_bridge::{copy_cuda_to_cuda, element_op_3d_inplace, free_cuda_array, new_cuda_array, scalar_op_3d_inplace, softmax_forward}, 
    neuralnet::TraversePtrs, 
    pointer_ops::{counter_is_zero, increment_counter, init_trav_in_ptrs, set_zero_counter}};

use super::layer_cuda::{AllocationStatus, IOPtrs, LayerCuda};

/// Softmax layer, used as the output layer for probabilistic
/// output classification
pub struct SoftmaxCuda
{
    pub io_ptrs: IOPtrs,
    pub allocation_status: AllocationStatus,

    pub input_exp_ptr: *mut f32,
    pub exp_sum_ptr: *mut f32,
    pub broadcast_temp_ptr: *mut f32,
    
    pub temperature: f32,
    pub input_grad_temp: *mut f32,

    pub batch_size: f32,
}

// implement constructor
impl SoftmaxCuda
{
    pub fn new(batch: usize, rows: usize, cols: usize, temperature: f32) -> Self
    {
        return Self
        {   
            io_ptrs: IOPtrs::new((batch, rows, cols), (batch, rows, cols)),
            allocation_status: AllocationStatus::new(),

            input_exp_ptr: std::ptr::null_mut(),
            exp_sum_ptr: std::ptr::null_mut(),
            input_grad_temp: std::ptr::null_mut(),
            broadcast_temp_ptr: std::ptr::null_mut(),

            temperature,
            batch_size: 0.0
        }
    }
}

// trait implementation
impl LayerCuda for SoftmaxCuda
{
    fn forward(&mut self, trav_ptr_in: *mut TraversePtrs, _trav_ptr_weight: *mut TraversePtrs, _use_dropout: bool) -> *mut TraversePtrs
    {
        let batch: usize = self.io_ptrs.in_shape.0;
        let rows: usize = self.io_ptrs.in_shape.1;
        let cols: usize = self.io_ptrs.in_shape.2;

        let flattened_shape: u32 = (batch * rows * cols) as u32;

        if !self.allocation_status.ptrs_allocated
        {   
            // stores the natural exponential of all values in input array 
            self.input_exp_ptr = new_cuda_array(flattened_shape);

            // stores of the sum of natural exponential values
            self.exp_sum_ptr = new_cuda_array((batch * rows * 1) as u32);

            // potentially unnecessary
            self.broadcast_temp_ptr = new_cuda_array(flattened_shape);
            self.input_grad_temp = new_cuda_array(flattened_shape);

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

        // CUDA function to calculate softmax on input array
        softmax_forward(
            self.io_ptrs.input_ptr, self.input_exp_ptr, self.exp_sum_ptr,
            self.io_ptrs.output_ptr, self.broadcast_temp_ptr, self.temperature,
            batch, rows, cols, self.allocation_status.zero_output
        );

        // important for zeroing gradients during backward pass
        set_zero_counter(self.io_ptrs.backward_count);
        return self.io_ptrs.output_traverse_ptr;
    }
    
    fn backward(&mut self, _use_dropout: bool)
    {
        // potentially unnecessary, to be removed
        copy_cuda_to_cuda(self.input_grad_temp, self.io_ptrs.output_grad_ptr, &[self.io_ptrs.in_shape.0, self.io_ptrs.in_shape.1, self.io_ptrs.in_shape.2]);
        
        // for temperature
        scalar_op_3d_inplace(
            self.input_grad_temp, self.temperature, 3, 
            self.io_ptrs.in_shape.0, self.io_ptrs.in_shape.1, self.io_ptrs.in_shape.2
        );

        // only zeros input gradients when the counter in the previous layer is zero
        if counter_is_zero(self.io_ptrs.backward_count_in_prev)
        {
            // overwrite the input gradients instead of accumulation
            copy_cuda_to_cuda(
                self.io_ptrs.input_grad_ptr, self.input_grad_temp, 
                &[self.io_ptrs.in_shape.0, self.io_ptrs.in_shape.1, self.io_ptrs.in_shape.2]
            );
        }
        else
        {
            // accumulate input gradients as another layer has written to it previously
            element_op_3d_inplace(
                self.io_ptrs.input_grad_ptr, self.input_grad_temp, 0, 
                self.io_ptrs.in_shape.0, self.io_ptrs.in_shape.1, self.io_ptrs.in_shape.2
            );
        }

        // increment counter to tell other layers connected to the same previous layer
        // to accumulate the gradient instead of zeroing it first
        increment_counter(self.io_ptrs.backward_count_in_prev);
        self.batch_size += 1.0;
    }

    fn update_params(&mut self, _optimizer_type: i32, _lr: f32, _l2: f32, _alpha: f32, _beta: f32)
    {    
        self.batch_size = 0.0;
    }

    fn details(&self)
    {
        // print all details of layer (e.g. IO shape, weights, etc)
        println!("Layer type: Softmax");
        println!("Input ptr: {:?} | Input grad ptr: {:?}", self.io_ptrs.input_ptr, self.io_ptrs.input_grad_ptr);
        println!("Output ptr: {:?} | Output grad ptr: {:?}", self.io_ptrs.output_ptr, self.io_ptrs.output_grad_ptr);
        println!("Shape: {:?}", self.io_ptrs.in_shape);
    }

    fn free_detached_ptrs(&self) 
    {
        free_cuda_array(self.input_exp_ptr as *mut c_void);
        free_cuda_array(self.exp_sum_ptr as *mut c_void);
        free_cuda_array(self.broadcast_temp_ptr as *mut c_void);
        free_cuda_array(self.input_grad_temp as *mut c_void);
    }
}