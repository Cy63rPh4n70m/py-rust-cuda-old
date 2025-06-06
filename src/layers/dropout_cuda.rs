use std::os::raw::c_void;

use crate::{
    cuda_bridge::{copy_cuda_to_cuda, dropout_backward, dropout_forward, free_cuda_array, init_random_states, new_cuda_array}, 
    neuralnet::TraversePtrs, 
    pointer_ops::{increment_counter, init_trav_in_ptrs, set_zero_counter}};

use super::layer_cuda::{AllocationStatus, IOPtrs, LayerCuda};

pub struct DropoutCuda
{
    pub name: String,
    pub io_ptrs: IOPtrs,
    pub allocation_status: AllocationStatus,

    pub dropout_mask_ptr: *mut f32,
    pub rand_state_v_ptr: *mut c_void,
    pub dropout_rate: f32,
    pub batch_size: f32,
    pub count: u128,
}
impl DropoutCuda
{
    // weight matrix initialize during first ever run
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
            count: 0,
        }
    }
}

impl LayerCuda for DropoutCuda
{
    fn forward(&mut self, trav_ptr_in: *mut TraversePtrs, _trav_ptr_weight: *mut TraversePtrs, use_dropout: bool) -> *mut TraversePtrs
    {
        let batch: usize = self.io_ptrs.in_shape.0;
        let rows: usize = self.io_ptrs.in_shape.1;
        let cols: usize = self.io_ptrs.in_shape.2;

        if !self.allocation_status.ptrs_allocated
        {   
            self.dropout_mask_ptr = new_cuda_array((batch * rows * cols) as u32);

            // initialize the random states pointer
            self.rand_state_v_ptr = init_random_states(batch, rows, cols);
            
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
            dropout_forward(
                self.io_ptrs.input_ptr, self.io_ptrs.output_ptr, self.dropout_mask_ptr, 
                self.rand_state_v_ptr,
                self.io_ptrs.in_shape.0, self.io_ptrs.in_shape.1, 
                self.io_ptrs.in_shape.2, 
                self.dropout_rate
            );

            //println!("{:?}", cuda_ptr_to_array(input_ptr, &[batch, rows, cols]));
            //println!("{:?}", cuda_ptr_to_array(result_ptr, &[batch, rows, cols]));
            //exit(1);
        }
        else
        {
            copy_cuda_to_cuda(
                self.io_ptrs.output_ptr, self.io_ptrs.input_ptr, 
                &[self.io_ptrs.in_shape.0, self.io_ptrs.in_shape.1, self.io_ptrs.in_shape.2]
            );
        }

        set_zero_counter(self.io_ptrs.backward_count);

        return self.io_ptrs.output_traverse_ptr;

        //println!("{:?}", cuda_ptr_to_array(input.get_ptr(), input.get_shape()));
        //println!("-----------------------------");
        //exit(1);
        // previous pointer will be recorded in previous layer

    }

    fn backward(&mut self, use_dropout: bool)
    {
        if use_dropout
        {
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
            copy_cuda_to_cuda(
                self.io_ptrs.input_grad_ptr, self.io_ptrs.output_grad_ptr, 
                &[self.io_ptrs.in_shape.0, self.io_ptrs.in_shape.1, self.io_ptrs.in_shape.2]);
        }

        //println!("backward {:?}", cuda_ptr_to_array(input_grad_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
        //exit(1);

        increment_counter(self.io_ptrs.backward_count);
        //let end = start.elapsed();
        //println!("backward: {:.6}", end.as_secs_f64());

        //println!("\noriginal_grads: {:?}", cuda_ptr_to_array(ptr.get_ptr(), ptr.get_shape()));
        //println!("\nchained_gradients: {:?}", cuda_ptr_to_array(ptr.get_ptr(), ptr.get_shape()));
        //println!("\nweight_gradients: {:?}", cuda_ptr_to_array(weight_grad_ptr, &[self.in_shape.0, self.in_shape.2, self.out_shape.2]));
        //println!("\nbias_gradients: {:?}", cuda_ptr_to_array(bias_grad_ptr, &[self.out_shape.0, self.out_shape.1, self.out_shape.2]));
        //println!("=========================================================");
        //exit(1);
        // calculate summed respect to bias
        // calculate bias gradients
    }

    fn update_params(&mut self, _optimizer_type: i32, _lr: f32, _l2: f32, _alpha: f32, _beta: f32)
    {    
        self.batch_size = 0.0;
    }

    fn details(&self)
    {
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