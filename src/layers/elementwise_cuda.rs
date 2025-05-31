use std::{os::raw::c_void, process::exit};

use crate::{
    cuda_bridge::{
        elementwise_dropout_backward, elementwise_dropout_forward, 
        gradient_desc_3d, init_random_states, new_cuda_array}, 
        math_functions::random_float_vec, neuralnet::TraversePtrs, 
        pointer_ops::{counter_is_zero, 
            cuda_ptr_to_vec, increment_counter, init_trav_in_ptrs, 
            set_zero_counter, vec_to_cuda_ptr}};

use super::layer_cuda::{AllocationStatus, IOPtrs, LayerCuda, ParameterPtrs, WeightTensors};

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
    pub range: f32,
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
}

impl LayerCuda for ElementwiseCuda
{
    // supports batch matrix multiplication unlike cpu
    fn forward(&mut self, trav_ptr_in: *mut TraversePtrs, trav_ptr_weight: *mut TraversePtrs, use_dropout: bool) -> *mut TraversePtrs
    {

        if !self.allocation_status.ptrs_allocated
        {   

            let shape_flat: u32 = (self.io_ptrs.in_shape.0 * self.io_ptrs.in_shape.1 * self.io_ptrs.in_shape.2) as u32;
            self.rand_state_v_ptr = init_random_states(
                self.io_ptrs.in_shape.0, self.io_ptrs.in_shape.1, self.io_ptrs.in_shape.2
            );
            self.dropout_mask_ptr = new_cuda_array(shape_flat);
            self.parameter_ptrs.weight_vel_ptr = new_cuda_array(shape_flat);
            self.parameter_ptrs.weight_moment_ptr = new_cuda_array(shape_flat);
            
            if trav_ptr_weight.is_null()
            {
                if !self.allocation_status.arrays_allocated
                {
                    // initialise weights and weight pointer
                    self.weight_tensors.weight = random_float_vec(
                        shape_flat as usize, 
                        -self.range, self.range
                    );

                }
                // initialize weight ptrs
                self.parameter_ptrs.weight_ptr = vec_to_cuda_ptr(&mut self.weight_tensors.weight);
                self.parameter_ptrs.weight_grad_ptr = new_cuda_array(
                    shape_flat
                );
            }
            else
            {
                unsafe
                {
                    self.parameter_ptrs.weight_ptr = (*trav_ptr_weight).ptr;
                    self.parameter_ptrs.weight_grad_ptr = (*trav_ptr_weight).grad_ptr;
                    self.io_ptrs.backward_count_weight_prev = (*trav_ptr_weight).backward_pass_count;
                }
            }

            init_trav_in_ptrs(
                &trav_ptr_in, &mut self.io_ptrs.backward_count,
                &mut self.io_ptrs.backward_count_in_prev, 
                &mut self.io_ptrs.input_ptr, &mut self.io_ptrs.input_grad_ptr, 
                &mut self.io_ptrs.output_ptr, &mut self.io_ptrs.output_grad_ptr, 
                &mut self.io_ptrs.output_traverse_ptr, 
                shape_flat as usize
            );

            self.allocation_status.ptrs_allocated = true;
            self.allocation_status.arrays_allocated = true;
        }
        
        elementwise_dropout_forward(
            self.io_ptrs.input_ptr, self.parameter_ptrs.weight_ptr, self.io_ptrs.output_ptr, 
            self.dropout_mask_ptr, self.rand_state_v_ptr,
            self.io_ptrs.in_shape.0, self.io_ptrs.in_shape.1, self.io_ptrs.in_shape.2, 
            self.dropout_rate, self.op,
            self.activation_fn_id, self.activation_scale, 
            use_dropout, self.allocation_status.zero_output
        );

        set_zero_counter(self.io_ptrs.backward_count);
        //println!("input: {:?}", cuda_ptr_to_array(input_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
        //println!("weights: {:?}", cuda_ptr_to_array(weight_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
        //println!("results: {:?}\n", cuda_ptr_to_array(result_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
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

        return self.io_ptrs.output_traverse_ptr;

    }

    fn backward(&mut self, use_dropout: bool)
    {
        if counter_is_zero(self.io_ptrs.backward_count_in_prev)
        {
            self.allocation_status.zero_input_grad = true;
        }
        
        if counter_is_zero(self.io_ptrs.backward_count_weight_prev)
        {
            self.allocation_status.zero_weight_grad = true;
        }

        elementwise_dropout_backward(
            self.io_ptrs.output_grad_ptr, self.dropout_mask_ptr, 
            self.io_ptrs.input_ptr, self.parameter_ptrs.weight_ptr, 
            self.parameter_ptrs.weight_grad_ptr, 
            self.io_ptrs.input_grad_ptr, use_dropout, 
            self.activation_fn_id, self.activation_scale,
            self.io_ptrs.in_shape.0, self.io_ptrs.in_shape.1, self.io_ptrs.in_shape.2, 
            self.op, self.allocation_status.zero_input_grad, self.allocation_status.zero_weight_grad
        );
        
        increment_counter(self.io_ptrs.backward_count);

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
    }

    fn update_params(&mut self, optimizer_type: i32, lr: f32, l2: f32, alpha: f32, beta: f32)
    {   
        gradient_desc_3d(
            lr, l2,
            self.parameter_ptrs.weight_ptr, self.parameter_ptrs.weight_grad_ptr, 
            self.parameter_ptrs.weight_vel_ptr, self.parameter_ptrs.weight_moment_ptr,
            self.io_ptrs.in_shape.0, self.io_ptrs.in_shape.1, self.io_ptrs.in_shape.2,
            
            self.parameter_ptrs.weight_ptr, self.parameter_ptrs.weight_grad_ptr, 
            self.parameter_ptrs.weight_vel_ptr, self.parameter_ptrs.weight_moment_ptr,
            self.io_ptrs.in_shape.0, self.io_ptrs.in_shape.1, self.io_ptrs.in_shape.2,

            true, true, self.batch_size, optimizer_type, alpha, beta
        );
        
        self.batch_size = 0.0;

        ////println!("{:?}", cuda_ptr_to_array(weight_grad_ptr, &[self.shape.0, self.shape.2, self.out_shape.2]))
        //self.weights -= &(self.lr * (&self.weight_gradients + self.l2 * &self.weights));
        //self.biases -= &(self.lr * &self.bias_gradients);
    }

    fn details(&self)
    {
        println!("Layer type: ELEMENTWISE");
        println!("Input ptr: {:?} | Input grad ptr: {:?}", self.io_ptrs.input_ptr, self.io_ptrs.input_grad_ptr);
        println!("Weight ptr: {:?} | Weight grad ptr: {:?}", self.parameter_ptrs.weight_ptr, self.parameter_ptrs.weight_grad_ptr);
        println!("Output ptr: {:?} | Output grad ptr: {:?}", self.io_ptrs.output_ptr, self.io_ptrs.output_grad_ptr);
        println!("Input shape: {:?}", self.io_ptrs.in_shape);
        
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
        println!("Weights: \n{:?}", self.weight_tensors.weight);
    }

    fn get_param_count(&self) -> usize
    {
        return self.io_ptrs.in_shape.0 * self.io_ptrs.in_shape.1 * self.io_ptrs.in_shape.2;
    }

    fn move_ptrs_to_arrays(&mut self)
    {
        self.weight_tensors.weight = cuda_ptr_to_vec(
            self.parameter_ptrs.weight_ptr, 
            self.io_ptrs.in_shape.0 * self.io_ptrs.in_shape.1 * self.io_ptrs.in_shape.2
        );
        //self.biases = cuda_ptr_to_array(string_to_ptr(&self.biases_ptr), &[self.shape.0, self.shape.1, self.shape.2]);
        //free_cuda_array(string_to_ptr(&self.weight_ptr));
        //free_cuda_array(string_to_ptr(&self.biases_ptr));
        //free_cuda_array(string_to_ptr(&self.input_ptr));
        //free_cuda_array(string_to_ptr(&self.result_ptr));
        //free_cuda_array(string_to_ptr(&self.input_grads_ptr));
        //free_cuda_array(string_to_ptr(&self.output_grads_ptr));
        //free_cuda_array(string_to_ptr(&self.weight_gradients_ptr));
        //free_cuda_array(string_to_ptr(&self.bias_gradients_ptr));
    }
}