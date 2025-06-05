use crate::{
    cuda_bridge::{l2norm_backward, l2norm_forward, new_cuda_array}, 
    neuralnet::TraversePtrs, 
    pointer_ops::{counter_is_zero, increment_counter, init_trav_in_ptrs, set_zero_counter}};

use super::layer_cuda::{AllocationStatus, IOPtrs, LayerCuda};

pub struct L2NormCuda
{
    pub io_ptrs: IOPtrs,
    pub allocation_status: AllocationStatus,

    pub input_pow2_ptr: *mut f32,
    pub power_sum: *mut f32,
    pub batch_size: f32,
}
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

impl LayerCuda for L2NormCuda
{
    // supports batch matrix multiplication unlike cpu
    fn forward(&mut self, trav_ptr_in: *mut TraversePtrs, _trav_ptr_weight: *mut TraversePtrs, _use_dropout: bool) -> *mut TraversePtrs
    {
        let batch: usize = self.io_ptrs.in_shape.0;
        let rows: usize = self.io_ptrs.in_shape.1;
        let cols: usize = self.io_ptrs.in_shape.2;

        if !self.allocation_status.ptrs_allocated
        {   
            // initialise input pointer, set the input as the result pointer from previous layer
            // tensor struct at this stage will contain the result ptr of the previous layer
            ////////////////////////////////////////////////////////////////
            self.input_pow2_ptr = new_cuda_array((batch * rows * cols) as u32);
            
            // initialise the result tensor/pointer
            self.power_sum = new_cuda_array((batch * rows * 1) as u32);

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

        l2norm_forward(
            self.io_ptrs.input_ptr, self.input_pow2_ptr, 
            batch, rows, cols, 
            self.power_sum, self.io_ptrs.output_ptr, true
        );

        //println!("{:?}", cuda_ptr_to_array(input_ptr, &[batch, rows, cols]));
        //println!("{:?}", cuda_ptr_to_array(power_sum, &[batch, rows, 1]));
        //println!("{:?}", cuda_ptr_to_array(result_ptr, &[batch, rows, cols]));
        //exit(1);

        set_zero_counter(self.io_ptrs.backward_count);

        return self.io_ptrs.output_traverse_ptr;
    }

    fn backward(&mut self, _use_dropout: bool)
    {
        if counter_is_zero(self.io_ptrs.backward_count_in_prev)
        {
            self.allocation_status.zero_input_grad = true;
        }

        l2norm_backward(
            self.io_ptrs.output_grad_ptr, 
            self.io_ptrs.in_shape.0, self.io_ptrs.in_shape.1, self.io_ptrs.in_shape.2, 
            self.power_sum, self.io_ptrs.input_ptr, self.io_ptrs.output_ptr, 
            self.io_ptrs.input_grad_ptr, self.allocation_status.zero_input_grad
        );
        self.batch_size += 1.0;
        increment_counter(self.io_ptrs.backward_count);

        //println!("{:?}", cuda_ptr_to_array(original_grads, &[self.shape.0, self.shape.1, self.shape.2]));
        //println!("{:?}", cuda_ptr_to_array(input_grad_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
        
        //exit(1);
        //let end = start.elapsed();
        //println!("backward: {:.6}", end.as_secs_f64());

        //println!("\noriginal_grads: {:?}", cuda_ptr_to_array(ptr.get_ptr(), ptr.get_shape()));
        //println!("\nchained_gradients: {:?}", cuda_ptr_to_array(ptr.get_ptr(), ptr.get_shape()));
        //exit(1);
        //println!("\nweight_gradients: {:?}", cuda_ptr_to_array(weight_grad_ptr, &[self.in_shape.0, self.in_shape.2, self.out_shape.2]));
        //println!("\nbias_gradients: {:?}", cuda_ptr_to_array(bias_grad_ptr, &[self.out_shape.0, self.out_shape.1, self.out_shape.2]));
        //println!("=========================================================");
        //exit(1);      
    }
    
    fn update_params(&mut self, _optimizer_type: i32, _lr: f32, _l2: f32, _alpha: f32, _beta: f32)
    {    
        self.batch_size = 0.0;
    }

    fn details(&self)
    {
        println!("Layer type: L2Norm");
        println!("Input Shape: {:?}", self.io_ptrs.in_shape);
        println!("Input ptr: {:?} | Input grad ptr: {:?}", self.io_ptrs.input_ptr, self.io_ptrs.input_grad_ptr);
        println!("Output ptr: {:?} | Output grad ptr: {:?}", self.io_ptrs.output_ptr, self.io_ptrs.output_grad_ptr);
    }
}