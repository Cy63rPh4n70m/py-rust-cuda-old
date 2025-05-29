use serde::{Deserialize, Serialize};

use crate::{cuda_bridge::{l2norm_backward, l2norm_forward}, pointer_ops::{counter_is_zero, increment_counter, init_layer_connections, new_cuda_ptr_str, set_zero_counter, string_to_ptr}};

#[derive(Serialize, Deserialize)]
pub struct L2NormCuda
{
    pub shape: (usize, usize, usize),

    pub input_ptr: String,
    pub input_pow2_ptr: String,
    pub power_sum: String,
    pub output_ptr: String,
    pub input_grads_ptr: String,
    pub output_grads_ptr: String,
    
    pub output_traverse_ptr: String,
    
    pub backward_count: String,
    pub backward_count_prev: String,

    pub scale_ptr: String,
    pub scale_vec: Vec<f32>,
    
    pub batch_size: f32,
    pub zero_input_grad: bool,

    pub ptrs_allocated: bool, 
}
impl L2NormCuda
{
    // weight matrix initialize during first ever run
    pub fn new(batch: usize, rows: usize, cols: usize) -> Self
    {
        return Self
        {
            input_ptr: "".to_string(),
            input_pow2_ptr: "".to_string(),
            power_sum: "".to_string(),
            input_grads_ptr: "".to_string(),
            output_grads_ptr: "".to_string(),
            output_ptr: "".to_string(),
            shape: (batch, rows, cols),
            output_traverse_ptr: "".to_string(),

            backward_count: String::from("none"),
            backward_count_prev: String::from("none"),

            scale_ptr: "".to_string(),
            scale_vec: Vec::new(),

            batch_size: 0.0,
            zero_input_grad: false,
            ptrs_allocated: false, 
        }
    }

    // supports batch matrix multiplication unlike cpu
    pub fn forward(&mut self, str_ptr_in: String) -> String
    {
        let batch: usize = self.shape.0;
        let rows: usize = self.shape.1;
        let cols: usize = self.shape.2;

        if !self.ptrs_allocated
        {   
            // initialise input pointer, set the input as the result pointer from previous layer
            // tensor struct at this stage will contain the result ptr of the previous layer
            ////////////////////////////////////////////////////////////////
            self.input_pow2_ptr = new_cuda_ptr_str(&[batch, rows, cols]);
            
            // initialise the result tensor/pointer
            self.power_sum = new_cuda_ptr_str(&[batch, rows, 1]);

            init_layer_connections(
                &mut self.backward_count, &str_ptr_in, 
                &mut self.backward_count_prev, &mut self.input_ptr, 
                &mut self.input_grads_ptr, &mut self.output_ptr, 
                &mut self.output_grads_ptr, &mut self.output_traverse_ptr, 
                &[self.shape.0, self.shape.1, self.shape.2]
            );
            
            self.ptrs_allocated = true;
        }

        //let broadcast_buf: String = new_cuda_ptr_str(batch * rows * cols * self.n_out);

        // convert strings to pointers
        let input_ptr: *mut f32 = string_to_ptr(&self.input_ptr);
        let input_pow2_ptr: *mut f32 = string_to_ptr(&self.input_pow2_ptr);
        let power_sum: *mut f32 = string_to_ptr(&self.power_sum);
        let result_ptr: *mut f32 = string_to_ptr(&self.output_ptr);

        // data does not need to be copied as result ptr from previous layer
        // is set as the input

        // copy data to the input pointer, prevent reallocation
        //copy_cuda_to_cuda(
        //    input_ptr, 
        //    input.get_ptr(), 
        //    &[batch, rows, cols]
        //);

        // parallel perform matrix multiplication
        // and sum with bias tensor
        // result pointer updated
        //let start: Instant = Instant::now();
        //if self.use_tiled || !self.use_tiled
        //{
        //println!("{:?}", cuda_ptr_to_array(input_ptr, &[batch, rows, cols]));
        l2norm_forward(
            input_ptr, input_pow2_ptr, 
            batch, rows, cols, 
            power_sum, result_ptr, true
        );

        //println!("{:?}", cuda_ptr_to_array(input_ptr, &[batch, rows, cols]));
        //println!("{:?}", cuda_ptr_to_array(power_sum, &[batch, rows, 1]));
        //println!("{:?}", cuda_ptr_to_array(result_ptr, &[batch, rows, cols]));
        //exit(1);

        set_zero_counter(&self.backward_count);

        return self.output_traverse_ptr.clone();
        //println!("{:?}", cuda_ptr_to_array(power_sum, &[batch, rows, 1]));
        //println!("{:?}", cuda_ptr_to_array(result_ptr, &[batch, rows, cols]));
        //}
        //else
        //{
        //    matmul_add_bias(
        //        input_ptr, batch as u32, rows as u32, cols as u32, 
        //        weight_ptr, batch as u32, cols as u32, self.n_out as u32,
        //        result_ptr, bias_ptr
        //    );
        //}
        //let end = start.elapsed();
        //println!("dense: {:.6}", end.as_secs_f64());

        //println!("-----------------------------");
        //println!("{:?}", cuda_ptr_to_array(input_ptr, &[batch, rows, cols]));
        //println!("{:?}", cuda_ptr_to_array(weight_ptr, &[batch, cols, self.n_out]));
        // overwrite the current pointer with result ptr, to be COPIED to input of next layer
        // current pointer is already recorded by previous layer, don't free

        //println!("{:?}", cuda_ptr_to_array(input.get_ptr(), input.get_shape()));
        //println!("-----------------------------");
        //exit(1);
        // previous pointer will be recorded in previous layer

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