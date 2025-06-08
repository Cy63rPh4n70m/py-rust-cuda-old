use crate::{
    cuda_bridge::{broadcast_2d_to_3d, sum_axis}, 
    neuralnet::TraversePtrs, 
    pointer_ops::{counter_is_zero, increment_counter, init_trav_in_ptrs, set_zero_counter}};

use super::layer_cuda::{AllocationStatus, IOPtrs, LayerCuda};

/// performs summing along axis of an array
pub struct SumCuda
{
    pub io_ptrs: IOPtrs,
    pub allocation_status: AllocationStatus,

    pub axis: i32,
    pub batch_size: f32,
}

// implement constructor
impl SumCuda
{
    pub fn new(
        in_batch: usize, in_rows: usize, in_cols: usize, 
        axis: i32
    ) -> Self
    {
        let in_shape: (usize, usize, usize) = (in_batch, in_rows, in_cols);
        let mut out_shape: (usize, usize, usize) = in_shape.clone();
        match axis
        {
            0 => out_shape.0 = 1,
            1 => out_shape.1 = 1,
            2 => out_shape.2 = 1,
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

// trait implementation
impl LayerCuda for SumCuda
{
    fn forward(&mut self, trav_ptr_in: *mut TraversePtrs, _trav_ptr_weight: *mut TraversePtrs, _use_dropout: bool) -> *mut TraversePtrs
    {
        if !self.allocation_status.ptrs_allocated
        {
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

        // CUDA axis sum
        sum_axis(
            self.io_ptrs.output_ptr, self.io_ptrs.input_ptr, 
            self.io_ptrs.in_shape.0, self.io_ptrs.in_shape.1, self.io_ptrs.in_shape.2, 
            self.axis, self.allocation_status.zero_output
        );

        // important for zeroing gradients during backward pass
        set_zero_counter(self.io_ptrs.backward_count);

        return self.io_ptrs.output_traverse_ptr;

    }

    fn backward(&mut self, _use_dropout: bool)
    {
        // only zero input gradients when previous layer's
        // backward counter is zero
        if !counter_is_zero(self.io_ptrs.backward_count_in_prev)
        {
            self.allocation_status.zero_input_grad = false;
        }

        // opposite of summing is broadcasting
        broadcast_2d_to_3d(
            self.io_ptrs.input_grad_ptr, 
            self.io_ptrs.in_shape.0, self.io_ptrs.in_shape.1, self.io_ptrs.in_shape.2, 
            self.io_ptrs.output_grad_ptr, self.axis
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
        println!("Layer type: SUM");
        println!("Input ptr: {:?} | Input grad ptr: {:?}", self.io_ptrs.input_ptr, self.io_ptrs.input_grad_ptr);
        println!("Output ptr: {:?} | Output grad ptr: {:?}", self.io_ptrs.output_ptr, self.io_ptrs.output_grad_ptr);
        println!("Input shape: {:?}", self.io_ptrs.in_shape);
        println!("Axis: {:?}", self.axis);
    }
}