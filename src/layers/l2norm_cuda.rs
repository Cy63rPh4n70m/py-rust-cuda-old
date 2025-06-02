use crate::{cuda_bridge::{l2norm_backward, l2norm_forward}, pointer_ops::{counter_is_zero, increment_counter, init_layer_connections, new_cuda_ptr_str, set_zero_counter, string_to_ptr}};

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

    pub fn backward(&mut self)
    {
        let input_ptr: *mut f32 = string_to_ptr(&self.input_ptr);
        let output_ptr: *mut f32 = string_to_ptr(&self.output_ptr);
        let power_sum: *mut f32 = string_to_ptr(&self.power_sum);
        let input_grad_ptr: *mut f32 = string_to_ptr(&self.input_grads_ptr);
        let original_grads: *mut f32 = string_to_ptr(&self.output_grads_ptr);

        //println!("=========================================================");
        //println!("input_array: {:?}", cuda_ptr_to_array(input_ptr, &[self.in_shape.0, self.in_shape.1, self.in_shape.2]));
        //println!("\npower_sum: {:?}", cuda_ptr_to_array(power_sum, &[self.in_shape.0, self.in_shape.2, 1]));
        //println!("\nmatmul result: {:?}", cuda_ptr_to_array(result_ptr, &[self.out_shape.0, self.out_shape.1, self.out_shape.2]));

        //let start: Instant = Instant::now();

        if counter_is_zero(&self.backward_count_prev)
        {
            self.zero_input_grad = true;
        }

        l2norm_backward(
            original_grads, 
            self.shape.0, self.shape.1, self.shape.2, 
            power_sum, input_ptr, output_ptr, 
            input_grad_ptr, self.zero_input_grad
        );
        self.batch_size += 1.0;
        increment_counter(&self.backward_count);

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
        // calculate summed respect to bias
        // calculate bias gradients

        /**/
        // calculate summed respect to bias
        // calculate bias gradients
        //self.bias_gradients += &(1.0 * &loss_r_summed); // bias derivative is 1.0

        /*
        // reshape
        let mut shape: Vec<usize> = loss_r_summed.shape().to_vec();
        shape.insert(shape.len() - 1, self.in_shape.1);
        let grads: ArrayViewD<f32> = loss_r_summed.broadcast(shape).unwrap();
        
        // calculate update gradients for weights (multiply with the reshaped inputs)

        let weight_grads: ArrayD<f32> = (&grads * &self.input).sum_axis(Axis(0));
        //self.weight_gradients += &weight_grads;

        // calculate update gradients for input (multiply with weights)
        let return_grads: ArrayD<f32> = (&grads * &self.weights).sum_axis(Axis(2));
        */
    }
    
    pub fn update_params(&mut self)
    {   
        self.batch_size = 0.0;

        //println!("{:?}", cuda_ptr_to_array(weight_grad_ptr, &[self.in_shape.0, self.in_shape.2, self.out_shape.2]))
        //self.weights -= &(self.lr * (&self.weight_gradients + self.l2 * &self.weights));
        //self.biases -= &(self.lr * &self.bias_gradients);
    }


    pub fn details(&self)
    {
        println!("Layer type: L2Norm");
        println!("Input Shape: {:?}", self.shape);
        println!("Input ptr: {:?} | Input grad ptr: {:?}", &self.input_ptr, &self.input_grads_ptr);
        println!("Output ptr: {:?} | Output grad ptr: {:?}", &self.output_ptr, &self.output_grads_ptr);
    }

    pub fn get_param_count(&self) -> usize
    {
        return 0_usize;
    }

    pub fn move_ptrs_to_arrays(&mut self)
    {
        self.ptrs_allocated = false;
    }

    pub fn set_ptrs_allocated(&mut self)
    {
        self.ptrs_allocated = true;
    }
}