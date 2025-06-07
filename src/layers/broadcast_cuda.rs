
use crate::{
    cuda_bridge::{broadcast_2d_to_3d, sum_axis}, 
    neuralnet::TraversePtrs, 
    pointer_ops::{counter_is_zero, increment_counter, init_trav_in_ptrs, set_zero_counter}};

use super::layer_cuda::{AllocationStatus, IOPtrs, LayerCuda};

pub struct BroadcastCuda
{
    pub io_ptrs: IOPtrs,
    pub allocation_status: AllocationStatus,

    pub axis: i32,
    pub batch_size: f32,
}
impl BroadcastCuda
{
    // weight matrix initialize during first ever run
    pub fn new(
        out_batch: usize, out_rows: usize, out_cols: usize, 
        axis: i32
    ) -> Self
    {
        let out_shape: (usize, usize, usize) = (out_batch, out_rows, out_cols);
        let mut in_shape: (usize, usize, usize) = out_shape.clone();
        match axis
        {
            0 => in_shape.0 = 1,
            1 => in_shape.1 = 1,
            2 => in_shape.2 = 1,
            _ => println!("Error: axis {} not allowed", axis)
        }

        return Self
        {
            io_ptrs: IOPtrs::new(in_shape, out_shape),
            allocation_status: AllocationStatus::new(),
            
            axis,

            batch_size: 0.0,
        }
    }

}

impl LayerCuda for BroadcastCuda
{
    // supports batch matrix multiplication unlike cpu
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
                (self.io_ptrs.out_shape.0 * self.io_ptrs.out_shape.1 * self.io_ptrs.out_shape.2) as usize
            );

            self.allocation_status.ptrs_allocated = true;
            self.allocation_status.arrays_allocated = true;
        }

        broadcast_2d_to_3d(
            self.io_ptrs.output_ptr, 
            self.io_ptrs.out_shape.0, self.io_ptrs.out_shape.1, self.io_ptrs.out_shape.2, 
            self.io_ptrs.input_ptr, self.axis
        );

        set_zero_counter(self.io_ptrs.backward_count);
        //println!("input: {:?}\n", cuda_ptr_to_array(input_ptr, &[1, 1, self.shape.2]));
        //println!("results: {:?}\n", cuda_ptr_to_array(result_ptr, &[self.shape.0, self.shape.1, self.shape.2]));

        return self.io_ptrs.output_traverse_ptr;
    }

    fn backward(&mut self, _use_dropout: bool)
    {
        if counter_is_zero(self.io_ptrs.backward_count_in_prev)
        {
            self.allocation_status.zero_input_grad = true;
        }

        sum_axis(
            self.io_ptrs.input_grad_ptr, self.io_ptrs.output_grad_ptr, 
            self.io_ptrs.out_shape.0, self.io_ptrs.out_shape.1, self.io_ptrs.out_shape.2, 
            self.axis, self.allocation_status.zero_input_grad
        );

        increment_counter(self.io_ptrs.backward_count_in_prev);
        self.batch_size += 1.0;
        
        /*
        println!("\noriginal_grads: {:?}", cuda_ptr_to_array(original_grads, &[self.shape.0, self.shape.1, self.shape.2]));
        //ptr.set_ptr(input_grad_ptr, vec![self.in_shape.0, self.in_shape.1, self.in_shape.2]);
        if self.axis == 0
        {
            println!("\nchained_gradients: {:?}", cuda_ptr_to_array(input_grad_ptr, &[1, self.shape.1, self.shape.2]));
        }
        else if self.axis == 1
        {
            println!("\nchained_gradients: {:?}", cuda_ptr_to_array(input_grad_ptr, &[self.shape.0, 1, self.shape.2]));
        }
        else if self.axis == 2
        {
            println!("\nchained_gradients: {:?}", cuda_ptr_to_array(input_grad_ptr, &[self.shape.0, self.shape.1, 1]));
        }
        */
        //println!("\nweight_gradients: {:?}", cuda_ptr_to_array(weight_grad_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
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
        println!("Layer type: BROADCAST");
        println!("Input ptr: {:?} | Input grad ptr: {:?}", self.io_ptrs.input_ptr, self.io_ptrs.input_grad_ptr);
        println!("Output ptr: {:?} | Output grad ptr: {:?}", self.io_ptrs.output_ptr, self.io_ptrs.output_grad_ptr);
        println!("Output shape: {:?}", self.io_ptrs.out_shape);
        println!("Axis: {}", self.axis);
    }
}