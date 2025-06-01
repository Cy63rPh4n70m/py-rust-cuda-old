use serde::{Deserialize, Serialize};

use crate::{cuda_bridge::transpose_2d, pointer_ops::{increment_counter, init_layer_connections, set_zero_counter, string_to_ptr}};

#[derive(Serialize, Deserialize)]
pub struct BatchTransposeCuda
{
    pub io_ptrs: IOPtrs,
    pub allocation_states: AllocationStatus,

    pub batch_size: f32,
    pub count: u128,
}
impl BatchTransposeCuda
{
    // weight matrix initialize during first ever run
    pub fn new(
        batch: usize, rows: usize, cols: usize
    ) -> Self
    {
        return Self
        {
            io_ptrs: IOPtrs::new((batch, rows, cols), (batch, cols, rows)),
            allocation_states: AllocationStatus::new(),
            batch_size: 0.0,
            count: 0,
        }
    }

    // supports batch matrix multiplication unlike cpu
    fn forward(&mut self, trav_ptr_in: *mut TraversePtrs, _trav_ptr_weight: *mut TraversePtrs, _use_dropout: bool) -> *mut TraversePtrs
    {

        if !self.allocation_states.ptrs_allocated
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

        transpose_2d(
            self.io_ptrs.output_ptr, self.io_ptrs.input_ptr, 
            self.io_ptrs.in_shape.0, self.io_ptrs.in_shape.1, self.io_ptrs.in_shape.2
        );

        set_zero_counter(&self.backward_count);

        //println!("input: {:?}\n", cuda_ptr_to_array(input_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
        //println!("results: {:?}\n", cuda_ptr_to_array(result_ptr, &[self.shape.0, self.shape.2, self.shape.1]));

        return self.output_traverse_ptr.clone();
    }

    pub fn backward(&mut self)
    {
        //let input_ptr: *mut f32 = string_to_ptr(self.io_ptrs.get("input").unwrap());
        //let input_t_ptr: *mut f32 = string_to_ptr(&self.input_t_ptr);
        //let weight_ptr: *mut f32 = string_to_ptr(self.io_ptrs.get("weight").unwrap());
        //let weight_t_ptr: *mut f32 = string_to_ptr(&self.weight_t_ptr);
        let input_grad_ptr: *mut f32 = string_to_ptr(&self.input_grad_ptr);
        //let weight_grad_ptr: *mut f32 = string_to_ptr(self.io_ptrs.get("weight_grad").unwrap());
        //let weight_grad_temp_ptr: *mut f32 = string_to_ptr(&self.weight_gradients_temp_ptr);
        let original_grads: *mut f32 = string_to_ptr(&self.output_grad_ptr);

        //let mask_ptr: *mut f32 = string_to_ptr(&self.dropout_mask_ptr);

        //println!("=========================================================");
        //println!("input_array: {:?}", cuda_ptr_to_array(input_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
        //println!("\nweights: {:?}", cuda_ptr_to_array(weight_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
        //println!("\nmatmul result: {:?}", cuda_ptr_to_array(result_ptr, &[self.shape.0, self.shape.1, self.shape.2]));

        // calculate gradient for bias
        //copy_cuda_to_cuda(
        //    bias_grad_ptr, original_grads, 
        //    &[self.shape.0, self.shape.1, self.shape.2]
        //);

        /*
        // calculate gradient for weights (broadcast multiplying original_grads along axis)
        element_op_3d_ret(
            weight_grad_temp_ptr, 
            original_grads, input_ptr, 2, 
            self.shape.0, self.shape.1, self.shape.2
        );
        element_op_3d_inplace(weight_grad_ptr, weight_grad_temp_ptr, 0, self.shape.0, self.shape.1, self.shape.2);

        //activation3d_cuda_backward(
        //    weight_grad_ptr, weight_shifted_ptr, weight_act_grad_ptr, 
        //    self.shape.0 as u32, self.shape.1 as u32, self.shape.2 as u32, 
        //    "softplus"
        //);
        
        // calculate gradient for input (elementwise multiplication followed by summation along axis)
        element_op_3d_ret(
            input_grad_ptr, 
            original_grads, weight_ptr, 2, 
            self.shape.0, self.shape.1, self.shape.2
        );
        */

        transpose_2d(input_grad_ptr, original_grads, self.shape.0, self.shape.2, self.shape.1);

        increment_counter(&self.backward_count);
        self.batch_size += 1.0;
        
        //let end = start.elapsed();
        //println!("backward: {:.6}", end.as_secs_f64());

        
        //println!("original_grads: {:?}\n", cuda_ptr_to_array(&[self.shape.0, self.shape.1, self.shape.2]));
        //ptr.set_ptr(input_grad_ptr, vec![self.shape.0, self.shape.1, self.shape.2]);
        //println!("input_gradients: {:?}\n", cuda_ptr_to_array(ptr.get_ptr(), ptr.get_shape()));
        //println!("weight_gradients: {:?}\n", cuda_ptr_to_array(input_grad_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
        //println!("mask: {:?}\n", cuda_ptr_to_array(mask_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
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
        shape.insert(shape.len() - 1, self.shape.1);
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
        println!("Layer type: BATCH_TRANSPOSE");
        println!("Input ptr: {:?} | Input grad ptr: {:?}", self.input_ptr, self.input_grad_ptr);
        println!("Output ptr: {:?} | Output grad ptr: {:?}", self.output_ptr, self.output_grad_ptr);
        println!("Input shape: {:?}", self.shape);
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