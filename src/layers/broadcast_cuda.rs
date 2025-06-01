
use crate::{cuda_bridge::{broadcast_2d_to_3d, sum_axis}, pointer_ops::{counter_is_zero, increment_counter, init_layer_connections, set_zero_counter, string_to_ptr}};

use super::layer_cuda::{AllocationStatus, IOPtrs};

pub struct BroadcastCuda
{
    pub io_ptrs: IOPtrs,
    pub allocation_status: AllocationStatus,

    pub axis: i32,
    pub batch_size: f32,
    pub count: u128,
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
            count: 0,
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
                (self.io_ptrs.out_shape.0 * self.io_ptrs.out_shape.2 * self.io_ptrs.out_shape.1) as usize
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

        increment_counter(self.io_ptrs.backward_count);
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

    pub fn update_params(&mut self)
    {   
        /*
        let weight_ptr: *mut f32 = string_to_ptr(self.io_ptrs.get("weight").unwrap());
        let weight_grad_ptr: *mut f32 = string_to_ptr(self.io_ptrs.get("weight_grad").unwrap());
        let weight_vel_ptr: *mut f32 = string_to_ptr(&self.weight_vel_ptr);

        //scalar_op_3d_inplace(weight_grad_ptr, self.lr, 2, self.shape.0, self.shape.2, self.shape.2);
        //scalar_op_3d_inplace(bias_grad_ptr, self.lr, 2, self.shape.0, self.shape.1, self.shape.2);
        //element_op_3d_inplace(weight_ptr, weight_grad_ptr, 1, self.shape.0, self.shape.2, self.shape.2);
        //element_op_3d_inplace(bias_ptr, bias_grad_ptr, 1, self.shape.0, self.shape.1, self.shape.2);
        gradient_desc_3d_dense(
            self.lr, self.l2,
            weight_ptr, weight_grad_ptr, weight_vel_ptr,self.shape.0, self.shape.1, self.shape.2,
            weight_ptr, weight_grad_ptr, weight_vel_ptr, self.shape.0, self.shape.1, self.shape.2,
            true, true, self.batch_size
        );

        //println!("{:?}", cuda_ptr_to_array(weight_grad_ptr, &[self.shape.0, self.shape.2, self.shape.2]))
        //self.weights -= &(self.lr * (&self.weight_gradients + self.l2 * &self.weights));
        //self.biases -= &(self.lr * &self.bias_gradients);
        */
        self.batch_size = 0.0;

    }

    pub fn zero_io(&mut self, io_ptr_name: &String)
    {
        //let io_ptr: *mut f32 = string_to_ptr(self.io_ptrs.get(io_ptr_name).unwrap());

        if io_ptr_name.contains("input")
        {
            //zeroes_3d_inplace(io_ptr, self.in_shape.0, self.in_shape.1, self.in_shape.2);
            self.zero_input_grad = true;
        }
        
        if io_ptr_name.contains("output")
        {
            //zeroes_3d_inplace(io_ptr, self.shape.0, self.shape.1, self.shape.2);
            self.zero_output = true;
        }
    }

    pub fn details(&self)
    {
        println!("Layer type: BROADCAST");
        println!("Input ptr: {:?} | Input grad ptr: {:?}", self.input_ptr, self.input_grad_ptr);
        println!("Output ptr: {:?} | Output grad ptr: {:?}", self.output_ptr, self.output_grad_ptr);
        println!("Output shape: {:?}", self.shape);
        println!("Axis: {}", self.axis);
    }

    pub fn get_param_count(&self) -> usize
    {
        return 0_usize;
    }

    pub fn move_ptrs_to_arrays(&mut self)
    {
        //self.weights = cuda_ptr_to_array(
        //    string_to_ptr(self.io_ptrs.get("weight").unwrap()), 
        //    &[self.shape.0, self.shape.1, self.shape.2]
        //);
        //self.biases = cuda_ptr_to_array(string_to_ptr(&self.biases_ptr), &[self.shape.0, self.shape.1, self.shape.2]);
        //free_cuda_array(string_to_ptr(&self.weight_ptr));
        //free_cuda_array(string_to_ptr(&self.biases_ptr));
        //free_cuda_array(string_to_ptr(&self.input_ptr));
        //free_cuda_array(string_to_ptr(&self.result_ptr));
        //free_cuda_array(string_to_ptr(&self.input_grads_ptr));
        //free_cuda_array(string_to_ptr(&self.output_grads_ptr));
        //free_cuda_array(string_to_ptr(&self.weight_gradients_ptr));
        //free_cuda_array(string_to_ptr(&self.bias_gradients_ptr));
        self.in_out_ptrs_allocated = false;
    }

    pub fn set_ptrs_allocated(&mut self)
    {
        self.in_out_ptrs_allocated = true;
    }
}