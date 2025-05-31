use std::{os::raw::c_void, process::exit};

use ndarray::{ArrayD, IxDyn};
use rand::Rng;

use crate::{cuda_bridge::{elementwise_dropout_backward, elementwise_dropout_forward, gradient_desc_3d, init_random_states}, pointer_ops::{array_to_cuda_ptr_str, counter_is_zero, cuda_ptr_to_array, get_traverse_str_ptr, increment_counter, init_layer_connections, new_cuda_ptr_str, ptr_to_string, ptr_to_string_void, set_zero_counter, string_to_ptr, string_to_ptr_void}};

use super::layer_cuda::{AllocationStatus, IOPtrs, ParameterPtrs, WeightTensors};

pub struct ElementwiseCuda
{
    pub io_ptrs: IOPtrs,
    pub parameter_ptrs: ParameterPtrs,
    pub allocation_status: AllocationStatus,
    pub weight_tensors: WeightTensors,
    
    pub dropout_mask_ptr: *mut f32,
    pub rand_state_v_ptr: *mut c_void,

    pub dropout_rate: f32,

    pub op: u32,
    pub range: f32
    pub activation_fn_id: i32,
    pub activation_scale: f32,

    pub batch_size: f32,
    pub count: u128,
}
impl ElementwiseCuda
{
    // weight matrix initialize during first ever run
    pub fn new(
        batch: usize, rows: usize, cols: usize, 
        range: f32, op: u32, dropout_rate: f32, activation_fn_id: i32, 
        activation_scale: f32
    ) -> Self
    {
        return Self
        {
            io_ptrs: IOPtrs::new((batch, rows, cols), (batch, rows, cols)),
            parameter_ptrs: ParameterPtrs::new(),
            allocation_status: AllocationStatus::new(),
            weight_tensors: WeightTensors::new(),

            dropout_mask_ptr: std::ptr::null_mut(),
            rand_state_v_ptr: std::ptr::null_mut(),

            dropout_rate,
            op,
            activation_fn_id,
            activation_scale,

            range,
            //out_shape,
            batch_size: 0.0,
            count: 0,
        }
    }

    // supports batch matrix multiplication unlike cpu
    pub fn forward(&mut self, str_ptr_in: String, str_ptr_weight: String, use_dropout: bool) -> String
    {
        ////println!("{:?}", cuda_ptr_to_array(input.get_ptr(), input_shape));

        ////println!("{:?}, {:?}, {:?}, {:?}", input.get_ptr(), batch, rows, cols);

        if !self.in_out_ptrs_allocated
        {   
            ////////////////////////////////////////////////////////////////
            //self.input_t_ptr = new_cuda_ptr_str(&[batch, cols, rows]);
            //////////////////////////////////////////////////////////////////
            

            // initialise the result tensor/pointer
            //self.result_ptr_t = new_cuda_ptr_str(&[batch, self.n_out, rows]);
            
            self.rand_state_v_ptr = ptr_to_string_void(init_random_states(self.shape.0, self.shape.1, self.shape.2));
            self.dropout_mask_ptr = new_cuda_ptr_str(&[self.shape.0, self.shape.1, self.shape.2]);
            self.weight_velocity_ptr = new_cuda_ptr_str(&[self.shape.0, self.shape.1, self.shape.2]);
            self.weight_momentum_ptr = new_cuda_ptr_str(&[self.shape.0, self.shape.1, self.shape.2]);
            //self.out_shape = (batch, rows, self.n_out);

            self.in_out_ptrs_allocated = true;
            
            if str_ptr_weight == "none" 
            {
                if !self.weight_array_allocated
                {
                    // initialise weights and weight pointer
                    //let range: f32 = (6.0 / (cols + rows) as f32).sqrt();
                    self.weights = ArrayD::from_shape_fn(
                        IxDyn(&[self.shape.0, self.shape.1, self.shape.2]), 
                        |_| rand::thread_rng().gen_range(-self.range..self.range)
                    );

                    self.weight_array_allocated = true;
                }

                //self.weights = Array3::from_shape_fn(
                //    (batch, cols, self.n_out), 
                //    |(i, j, k)|
                //    {
                //        (i * cols * self.n_out + j * self.n_out + k) as f32
                //    }
                //).into_dyn() / (batch * cols * self.n_out) as f32;
                self.weight_ptr = array_to_cuda_ptr_str(&mut self.weights);
                ////println!("{:?}", cuda_ptr_to_array(string_to_ptr(weight_ptr), &[self.shape.0, self.shape.1, self.shape.2]));
                self.weight_grad_ptr = new_cuda_ptr_str(&[self.shape.0, self.shape.1, self.shape.2]);
            }
            else
            {
                let (weight_traverse_ptr, grad_weight_traverse_ptr,
                    backward_count_weight_prev) = 
                    get_traverse_str_ptr(&str_ptr_weight);

                self.backward_count_weight_prev = backward_count_weight_prev;
                self.weight_ptr = ptr_to_string(weight_traverse_ptr);
                self.weight_grad_ptr = ptr_to_string(grad_weight_traverse_ptr);
            }

            self.weight_ptr_allocated = true;

            init_layer_connections(
                &mut self.backward_count, &str_ptr_in, 
                &mut self.backward_count_in_prev, &mut self.input_ptr, 
                &mut self.input_grad_ptr, &mut self.output_ptr, 
                &mut self.output_grad_ptr, &mut self.output_traverse_ptr, 
                &[self.shape.0, self.shape.1, self.shape.2]
            );
        }

        //let broadcast_buf: String = new_cuda_ptr_str(batch * rows * cols * self.n_out);

        // convert strings to pointers
        let input_ptr: *mut f32 = string_to_ptr(&self.input_ptr);
        let weight_ptr: *mut f32 = string_to_ptr(&self.weight_ptr);
        let mask_ptr: *mut f32 = string_to_ptr(&self.dropout_mask_ptr);
        let rand_states: *mut c_void = string_to_ptr_void(&self.rand_state_v_ptr);
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
        //copy_cuda_to_cuda(weight_shifted_ptr, weight_ptr, &[self.shape.0, self.shape.1, self.shape.2]);
        //scalar_op_3d_inplace(weight_shifted_ptr, self.weight_shift, 1, self.shape.0, self.shape.1, self.shape.2);
        //activation3d_cuda(
        //    weight_act_ptr, weight_shifted_ptr, 
        //    input_shape[0] as u32, input_shape[1] as u32, input_shape[2] as u32, 
        //    "softplus"
        //);
        
        elementwise_dropout_forward(
            input_ptr, weight_ptr, result_ptr, mask_ptr, rand_states,
            self.shape.0, self.shape.1, self.shape.2, self.dropout_rate, self.op,
            self.activation_fn_id, self.activation_scale, 
            use_dropout, self.zero_output
        );

        set_zero_counter(&self.backward_count);
        //println!("input: {:?}", cuda_ptr_to_array(input_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
        //println!("weights: {:?}", cuda_ptr_to_array(weight_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
        //println!("results: {:?}\n", cuda_ptr_to_array(result_ptr, &[self.shape.0, self.shape.1, self.shape.2]));

        return self.output_traverse_ptr.clone();

        //element_op_3d_ret(result_ptr, input_ptr, weight_ptr, 2, self.shape.0, self.shape.1, self.shape.2);

        //element_op_3d_inplace(
        //    result_ptr, bias_ptr, 
        //    0, 
        //    self.shape.0, self.shape.1, self.shape.2
        //);

        ////println!("-----------------------------");
        ////println!("input: {:?}\n", cuda_ptr_to_array(input_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
        ////println!("mask: {:?}\n", cuda_ptr_to_array(mask_ptr, &[batch, rows, cols]));
        ////println!("weights: {:?}\n", cuda_ptr_to_array(weight_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
        // overwrite the current pointer with result ptr, to be COPIED to input of next layer
        // current pointer is already recorded by previous layer, don't free
        //input.set_ptr(result_ptr, vec![self.shape.0, self.shape.1, self.shape.2]);

        ////println!("results: {:?}\n", cuda_ptr_to_array(result_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
        ////println!("-----------------------------");
        //exit(1);
        // previous pointer will be recorded in previous layer

    }

    pub fn backward(&mut self, use_dropout: bool)
    {
        let input_ptr: *mut f32 = string_to_ptr(&self.input_ptr);
        //let input_t_ptr: *mut f32 = string_to_ptr(&self.input_t_ptr);
        let weight_ptr: *mut f32 = string_to_ptr(&self.weight_ptr);
        //let weight_t_ptr: *mut f32 = string_to_ptr(&self.weight_t_ptr);
        let input_grad_ptr: *mut f32 = string_to_ptr(&self.input_grad_ptr);
        let weight_grad_ptr: *mut f32 = string_to_ptr(&self.weight_grad_ptr);
        //let weight_grad_temp_ptr: *mut f32 = string_to_ptr(&self.weight_gradients_temp_ptr);
        let original_grads: *mut f32 = string_to_ptr(&self.output_grad_ptr);

        let mask_ptr: *mut f32 = string_to_ptr(&self.dropout_mask_ptr);

        ////println!("=========================================================");
        ////println!("input_array: {:?}", cuda_ptr_to_array(input_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
        ////println!("\nweights: {:?}", cuda_ptr_to_array(weight_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
        ////println!("\nmatmul result: {:?}", cuda_ptr_to_array(result_ptr, &[self.shape.0, self.shape.1, self.shape.2]));

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
        ////println!("A");
        if counter_is_zero(&self.backward_count_in_prev)
        {
            self.zero_input_grad = true;
        }
        
        ////println!("{:?}", self.backward_count_weight_prev);
        if counter_is_zero(&self.backward_count_weight_prev)
        {
            self.zero_weight_grad = true;
        }

        elementwise_dropout_backward(
            original_grads, mask_ptr, 
            input_ptr, weight_ptr, weight_grad_ptr, 
            input_grad_ptr, use_dropout, self.activation_fn_id, self.activation_scale,
            self.shape.0, self.shape.1, self.shape.2, self.op,
            self.zero_input_grad, self.zero_weight_grad
        );
        
        increment_counter(&self.backward_count);

        self.batch_size += 1.0;
        ////println!("B");
        
        //let end = start.elapsed();
        ////println!("backward: {:.6}", end.as_secs_f64());

        
        ////println!("\noriginal_grads: {:?}", cuda_ptr_to_array(original_grads, &[self.shape.0, self.shape.1, self.shape.2]));
        //ptr.set_ptr(input_grad_ptr, vec![self.in_shape.0, self.in_shape.1, self.in_shape.2]);
        ////println!("\nchained_gradients: {:?}", cuda_ptr_to_array(input_grad_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
        ////println!("\nweight_gradients: {:?}", cuda_ptr_to_array(weight_grad_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
        ////println!("=========================================================");
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

    pub fn update_params(&mut self, optimizer_type: i32, lr: f32, l2: f32, alpha: f32, beta: f32)
    {   
        let weight_ptr: *mut f32 = string_to_ptr(&self.weight_ptr);
        let weight_grad_ptr: *mut f32 = string_to_ptr(&self.weight_grad_ptr);
        let weight_velocity_ptr: *mut f32 = string_to_ptr(&self.weight_velocity_ptr);
        let weight_momentum_ptr: *mut f32 = string_to_ptr(&self.weight_momentum_ptr);

        //scalar_op_3d_inplace(weight_grad_ptr, self.lr, 2, self.shape.0, self.shape.2, self.out_shape.2);
        //scalar_op_3d_inplace(bias_grad_ptr, self.lr, 2, self.out_shape.0, self.out_shape.1, self.out_shape.2);
        //element_op_3d_inplace(weight_ptr, weight_grad_ptr, 1, self.shape.0, self.shape.2, self.out_shape.2);
        //element_op_3d_inplace(bias_ptr, bias_grad_ptr, 1, self.out_shape.0, self.out_shape.1, self.out_shape.2);
        gradient_desc_3d(
            lr, l2,
            weight_ptr, weight_grad_ptr, weight_velocity_ptr, weight_momentum_ptr,
            self.shape.0, self.shape.1, self.shape.2,
            weight_ptr, weight_grad_ptr, weight_velocity_ptr, weight_momentum_ptr,
            self.shape.0, self.shape.1, self.shape.2,
            true, true, self.batch_size, optimizer_type, alpha, beta
        );
        
        self.batch_size = 0.0;

        ////println!("{:?}", cuda_ptr_to_array(weight_grad_ptr, &[self.shape.0, self.shape.2, self.out_shape.2]))
        //self.weights -= &(self.lr * (&self.weight_gradients + self.l2 * &self.weights));
        //self.biases -= &(self.lr * &self.bias_gradients);
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
            //zeroes_3d_inplace(io_ptr, self.out_shape.0, self.out_shape.1, self.out_shape.2);
            self.zero_output = true;
        }
        
        if io_ptr_name.contains("weight")
        {
            //zeroes_3d_inplace(io_ptr, self.in_shape.0, self.in_shape.2, self.out_shape.2);
            self.zero_weight_grad = true;
        }
    }

    pub fn details(&self)
    {
        println!("Layer type: ELEMENTWISE");
        println!("Input ptr: {:?} | Input grad ptr: {:?}", self.input_ptr, self.input_grad_ptr);
        println!("Weight ptr: {:?} | Weight grad ptr: {:?}", self.weight_ptr, self.weight_grad_ptr);
        println!("Output ptr: {:?} | Output grad ptr: {:?}", self.output_ptr, self.output_grad_ptr);
        println!("Input shape: {:?}", self.shape);
        
        let activation_str: &str;
        match self.activation_fn_id
        {
            0 => activation_str = "sigmoid",
            1 => activation_str = "tanh",
            2 => activation_str = "silu",
            3 => activation_str = "gelu",
            4 => activation_str = "tanh2",
            5 => activation_str = "softplus",
            6 => activation_str = "linear",
            _ => {println!("Invalid activation id {:?}", self.activation_fn_id); exit(1)},
        };
        println!("Activation: {:?} | Scale: {:?}", activation_str, self.activation_scale);

        if self.op == 0
        {
            println!("Type: add");
        }
        else if self.op == 1
        {
            println!("Type: mul");
        }
        println!("Weights: \n{:?}", self.weights);
    }

    pub fn get_param_count(&self) -> usize
    {
        return self.shape.0 * self.shape.1 * self.shape.2;
    }

    pub fn move_ptrs_to_arrays(&mut self)
    {
        self.weights = cuda_ptr_to_array(string_to_ptr(&self.weight_ptr), &[self.shape.0, self.shape.1, self.shape.2]);
        //self.biases = cuda_ptr_to_array(string_to_ptr(&self.biases_ptr), &[self.shape.0, self.shape.1, self.shape.2]);
        //free_cuda_array(string_to_ptr(&self.weight_ptr));
        //free_cuda_array(string_to_ptr(&self.biases_ptr));
        //free_cuda_array(string_to_ptr(&self.input_ptr));
        //free_cuda_array(string_to_ptr(&self.result_ptr));
        //free_cuda_array(string_to_ptr(&self.input_grads_ptr));
        //free_cuda_array(string_to_ptr(&self.output_grads_ptr));
        //free_cuda_array(string_to_ptr(&self.weight_gradients_ptr));
        //free_cuda_array(string_to_ptr(&self.bias_gradients_ptr));

        self.weight_ptr_allocated = false;
        self.bias_ptr_allocated = false;
        self.in_out_ptrs_allocated = false;
    }

    pub fn set_ptrs_allocated(&mut self)
    {
        self.weight_ptr_allocated = true;
        self.bias_ptr_allocated = true;
        self.in_out_ptrs_allocated = true;
    }
}