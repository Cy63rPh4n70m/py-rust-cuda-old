use std::os::raw::c_void;

use serde::{Deserialize, Serialize};

use crate::{cuda_bridge::{copy_cuda_to_cuda, dropout_backward, dropout_forward, init_random_states}, pointer_ops::{increment_counter, init_layer_connections, new_cuda_ptr_str, ptr_to_string_void, set_zero_counter, string_to_ptr, string_to_ptr_void}};

#[derive(Serialize, Deserialize)]
pub struct DropoutCuda
{
    pub name: String,
    pub shape: (usize, usize, usize),

    pub dropout_mask_ptr: String,
    pub rand_state_v_ptr: String,

    pub input_ptr: String,
    pub input_grad_ptr: String,
    pub output_ptr: String,
    pub output_grad_ptr: String,

    pub dropout_rate: f32,

    pub output_traverse_ptr: String,
    
    pub backward_count: String,
    pub backward_count_prev: String,

    pub batch_size: f32,
    pub count: u128,

    pub mask_ptr_allocated: bool, 
}
impl DropoutCuda
{
    // weight matrix initialize during first ever run
    pub fn new(batch: usize, rows: usize, cols: usize, dropout_rate: f32, name: &str) -> Self
    {
        return Self
        {
            name: name.to_string(),
            //input_ptr: "".to_string(),
            dropout_rate,
            
            dropout_mask_ptr: "".to_string(),

            //input_grads_ptr: String::from("NONE"),
            //output_grads_ptr: String::from("NONE"),
            rand_state_v_ptr: String::from("NONE"),

            input_ptr: "".to_string(),
            input_grad_ptr: "".to_string(),
            output_ptr: "".to_string(),
            output_grad_ptr: "".to_string(),

            output_traverse_ptr: "".to_string(),

            backward_count: String::from("none"),
            backward_count_prev: String::from("none"),

            //broadcast_array: ArrayD::zeros(IxDyn(&[0])),
            //broadcast_array_ptr: "".to_string(),
            shape: (batch, rows, cols),
            batch_size: 0.0,
            count: 0,
            mask_ptr_allocated: false,
        }
    }
    // supports batch matrix multiplication unlike cpu
    pub fn forward(&mut self, str_ptr_in: String, use_dropout: bool) -> String
    {
        //println!("{:?}", cuda_ptr_to_array(input.get_ptr(), input_shape));
        let batch: usize = self.shape.0;
        let rows: usize = self.shape.1;
        let cols: usize = self.shape.2;

        //println!("{:?}, {:?}, {:?}, {:?}", input.get_ptr(), batch, rows, cols);

        if !self.mask_ptr_allocated
        {   
            self.dropout_mask_ptr = new_cuda_ptr_str(&[batch, rows, cols]);

            // initialise input pointer, set the input as the result pointer from previous layer
            // tensor struct at this stage will contain the result ptr of the previous layer
            ////////////////////////////////////////////////////////////////
            //self.input_ptr = input.get_ptr_as_str();
            //self.input_t_ptr = new_cuda_ptr_str(&[batch, cols, rows]);
            //////////////////////////////////////////////////////////////////

            // initialise the result tensor/pointer
            //self.result_ptr = new_cuda_ptr_str(&[batch, rows, cols]);
            //self.result_ptr_t = new_cuda_ptr_str(&[batch, self.n_out, rows]);
            
            //self.input_grads_ptr = new_cuda_ptr_str(&[batch, rows, cols]);
            //self.output_grads_ptr = new_cuda_ptr_str(&[batch, rows, cols]);

            //self.shape = (batch, rows, cols);

            // initialize the random states pointer
            self.rand_state_v_ptr = ptr_to_string_void(init_random_states(batch, rows, cols));
            
            init_layer_connections(
                &mut self.backward_count, &str_ptr_in, 
                &mut self.backward_count_prev, &mut self.input_ptr, 
                &mut self.input_grad_ptr, &mut self.output_ptr, 
                &mut self.output_grad_ptr, &mut self.output_traverse_ptr, 
                &[self.shape.0, self.shape.1, self.shape.2]
            );

            self.mask_ptr_allocated = true;
        }

        //let broadcast_buf: String = new_cuda_ptr_str(batch * rows * cols * self.n_out);

        // convert strings to pointers
        let input_ptr: *mut f32 = string_to_ptr(&self.input_ptr);
        let mask_ptr: *mut f32 = string_to_ptr(&self.dropout_mask_ptr);
        let rand_states_ptr: *mut c_void = string_to_ptr_void(&self.rand_state_v_ptr);
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
        //println!("{:?}", cuda_ptr_to_array(mask_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
        if use_dropout
        {
            dropout_forward(
                input_ptr, result_ptr, mask_ptr, rand_states_ptr,
                self.shape.0, self.shape.1, self.shape.2, self.dropout_rate
            );

            //println!("{:?}", cuda_ptr_to_array(input_ptr, &[batch, rows, cols]));
            //println!("{:?}", cuda_ptr_to_array(result_ptr, &[batch, rows, cols]));
            //exit(1);
            // overwrite the current pointer with result ptr, to be COPIED to input of next layer
            // current pointer is already recorded by previous layer, don't free
            //input.set_ptr(result_ptr, vec![batch, rows, cols]);
        }
        else
        {
            copy_cuda_to_cuda(
                result_ptr, input_ptr, 
                &[self.shape.0, self.shape.1, self.shape.2]
            );
        }

        set_zero_counter(&self.backward_count);

        //println!("{:?}", self.output_traverse_ptr);

        return self.output_traverse_ptr.clone();

        //println!("{:?}", cuda_ptr_to_array(input.get_ptr(), input.get_shape()));
        //println!("-----------------------------");
        //exit(1);
        // previous pointer will be recorded in previous layer

    }

    pub fn backward(&mut self, use_dropout: bool)
    {
        //let input_ptr: *mut f32 = string_to_ptr(self.io_ptrs.get("input").unwrap());
        //let input_t_ptr: *mut f32 = string_to_ptr(&self.input_t_ptr);
        let dropout_mask_ptr: *mut f32 = string_to_ptr(&self.dropout_mask_ptr);
        //let weight_t_ptr: *mut f32 = string_to_ptr(&self.weight_t_ptr);
        let input_grad_ptr: *mut f32 = string_to_ptr(&self.input_grad_ptr);
        //let result_ptr: *mut f32 = string_to_ptr(self.io_ptrs.get("output").unwrap());
        let original_grads: *mut f32 = string_to_ptr(&self.output_grad_ptr);

        //println!("=========================================================");
        //println!("input_array: {:?}", cuda_ptr_to_array(input_ptr, &[self.in_shape.0, self.in_shape.1, self.in_shape.2]));
        //println!("\nweights: {:?}", cuda_ptr_to_array(dropout_mask_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
        //println!("\nmatmul result: {:?}", cuda_ptr_to_array(result_ptr, &[self.out_shape.0, self.out_shape.1, self.out_shape.2]));

        //let start: Instant = Instant::now();
        if use_dropout
        {
            dropout_backward(
                original_grads, dropout_mask_ptr, input_grad_ptr,
                self.shape.0, self.shape.1, self.shape.2
            );

            self.batch_size += 1.0;
            //ptr.set_ptr(input_grad_ptr, vec![self.shape.0, self.shape.1, self.shape.2]);
        }
        else
        {
            copy_cuda_to_cuda(input_grad_ptr, original_grads, &[self.shape.0, self.shape.1, self.shape.2]);
        }

        //println!("backward {:?}", cuda_ptr_to_array(input_grad_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
        //exit(1);

        increment_counter(&self.backward_count);
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
        println!("Layer type: DROPOUT | Layer name: {:?}", self.name);
        println!("Shape: {:?}", self.shape);
        println!("Dropout rate: {:?}", self.dropout_rate);
        println!("Input ptr: {:?} | Input grad ptr: {:?}", self.input_ptr, self.input_grad_ptr);
        println!("Output ptr: {:?} | Output grad ptr: {:?}", self.output_ptr, self.output_grad_ptr);
    }

    pub fn get_param_count(&self) -> usize
    {
        return 0_usize
    }

    pub fn move_ptrs_to_arrays(&mut self)
    {
        self.mask_ptr_allocated = false;
    }

    pub fn set_ptrs_allocated(&mut self)
    {
        self.mask_ptr_allocated = true;
    }
}