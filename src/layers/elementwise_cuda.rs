use std::{collections::HashMap, os::raw::c_void, process::exit};

use crate::{
    cuda_bridge::{
        elementwise_dropout_backward, elementwise_dropout_forward, free_cuda_array, gradient_desc_3d, init_random_states, new_cuda_array}, 
        math_functions::random_float_vec, neuralnet::TraversePtrs, 
        pointer_ops::{counter_is_zero, 
            cuda_ptr_to_vec, increment_counter, init_trav_in_ptrs, 
            set_zero_counter, vec_to_cuda_ptr}};

use super::layer_cuda::{AllocationStatus, IOPtrs, LayerCuda, ParameterPtrs, WeightTensors};

/// Elementwise layer performs Hadamard product or 
/// elementwise addition between two tensors
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
    pub activation_fn_id: i32,
    pub activation_scale: f32,

    pub batch_size: f32,
}

// implement constructor
impl ElementwiseCuda
{
    pub fn new(
        batch: usize, rows: usize, cols: usize, 
        range: f32, op: u32, dropout_rate: f32, activation_fn_id: i32, 
        activation_scale: f32
    ) -> Self
    {
        // initialize weights
        let shape_flat: usize = batch * rows * cols;
        let mut weight_tensors: WeightTensors = WeightTensors::new();
        weight_tensors.weight = random_float_vec(
            shape_flat, 
            -range, range
        );

        return Self
        {
            io_ptrs: IOPtrs::new((batch, rows, cols), (batch, rows, cols)),
            parameter_ptrs: ParameterPtrs::new(),
            allocation_status: AllocationStatus::new(),
            weight_tensors,

            dropout_mask_ptr: std::ptr::null_mut(),
            rand_state_v_ptr: std::ptr::null_mut(),

            dropout_rate,
            op,
            activation_fn_id,
            activation_scale,
            batch_size: 0.0,
        }
    }
}

// trait implementation
impl LayerCuda for ElementwiseCuda
{
    fn forward(&mut self, trav_ptr_in: *mut TraversePtrs, trav_ptr_weight: *mut TraversePtrs, use_dropout: bool) -> *mut TraversePtrs
    {
        if !self.allocation_status.ptrs_allocated
        {   
            // initialize pointers for built in dropout functionality
            let shape_flat: u32 = (self.io_ptrs.in_shape.0 * self.io_ptrs.in_shape.1 * self.io_ptrs.in_shape.2) as u32;
            self.rand_state_v_ptr = init_random_states(
                self.io_ptrs.in_shape.0, self.io_ptrs.in_shape.1, self.io_ptrs.in_shape.2
            );
            self.dropout_mask_ptr = new_cuda_array(shape_flat);

            // intialize pointers needed for weight training
            self.parameter_ptrs.weight_vel_ptr = new_cuda_array(shape_flat);
            self.parameter_ptrs.weight_moment_ptr = new_cuda_array(shape_flat);
            
            // decide whether to create weight based on traversal pointer availability
            if trav_ptr_weight.is_null()
            {
                // initialize weight ptrs
                self.parameter_ptrs.weight_ptr = vec_to_cuda_ptr(&mut self.weight_tensors.weight);
                self.parameter_ptrs.weight_grad_ptr = new_cuda_array(
                    shape_flat
                );

                self.parameter_ptrs.weight_ptr_detached = true;
            } 
            else
            {
                // set weight pointers with pointers in second traversal pointer
                // links the output of previous layer with this layer
                unsafe
                {
                    self.parameter_ptrs.weight_ptr = (*trav_ptr_weight).ptr;
                    self.parameter_ptrs.weight_grad_ptr = (*trav_ptr_weight).grad_ptr;
                    self.io_ptrs.backward_count_weight_prev = (*trav_ptr_weight).backward_pass_count;
                }

                self.parameter_ptrs.weight_ptr_detached = false;
            }

            // connect the output pointers of the previous layer with the 
            // current layer's input pointers
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
        
        // CUDA function for elementwise operations, dropout and activation is built in
        // to reduce kernel calls
        elementwise_dropout_forward(
            self.io_ptrs.input_ptr, self.parameter_ptrs.weight_ptr, self.io_ptrs.output_ptr, 
            self.dropout_mask_ptr, self.rand_state_v_ptr,
            self.io_ptrs.in_shape.0, self.io_ptrs.in_shape.1, self.io_ptrs.in_shape.2, 
            self.dropout_rate, self.op,
            self.activation_fn_id, self.activation_scale, 
            use_dropout, self.allocation_status.zero_output
        );

        // important for zeroing gradients during backward pass
        set_zero_counter(self.io_ptrs.backward_count);

        return self.io_ptrs.output_traverse_ptr;

    }

    fn backward(&mut self, use_dropout: bool)
    {
        // if statements control whether this layer will zero the gradients for the 
        // previous layer/s
        // only zeros when the counter in the previous layer is zero
        if !counter_is_zero(self.io_ptrs.backward_count_in_prev)
        {
            self.allocation_status.zero_input_grad = false;
        }
        
        if !counter_is_zero(self.io_ptrs.backward_count_weight_prev)
        {
            self.allocation_status.zero_weight_grad = false;
        }

        // CUDA function to calculate input and weight gradients uusing chain rule
        elementwise_dropout_backward(
            self.io_ptrs.output_grad_ptr, self.dropout_mask_ptr, 
            self.io_ptrs.input_ptr, self.parameter_ptrs.weight_ptr, 
            self.parameter_ptrs.weight_grad_ptr, 
            self.io_ptrs.input_grad_ptr, use_dropout, 
            self.activation_fn_id, self.activation_scale,
            self.io_ptrs.in_shape.0, self.io_ptrs.in_shape.1, self.io_ptrs.in_shape.2, 
            self.op, self.allocation_status.zero_input_grad, self.allocation_status.zero_weight_grad
        );
        
        // increment counter to tell other layers connected to the same previous layer
        // to accumulate the gradient instead of zeroing it first
        increment_counter(self.io_ptrs.backward_count_in_prev);

        // only if weight pointer is connected to another previous layer and isn't
        // standalone
        if !self.parameter_ptrs.weight_ptr_detached
        {
            increment_counter(self.io_ptrs.backward_count_weight_prev);
        }

        self.batch_size += 1.0;
    }

    fn update_params(&mut self, optimizer_type: i32, lr: f32, l2: f32, alpha: f32, beta: f32)
    {   
        // perform gradient descent with SGD or AdamW
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
    }

    fn details(&self)
    {
        // print all details of layer (e.g. IO shape, weights, etc)
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
    }

    fn get_weights_hashmap(&mut self) -> Option<HashMap<&str, Vec<f32>>>
    {
        self.move_ptrs_to_arrays();
        let mut hashmap: HashMap<&str, Vec<f32>> = HashMap::new();
        hashmap.insert("weights", self.weight_tensors.weight.clone());

        return Some(hashmap);
    }

    fn load_weights_from_hashmap(
        &mut self, json_hashmap: &HashMap<&str, Vec<f32>>
    ) 
    {
        self.weight_tensors.weight = json_hashmap.get("weights").unwrap().to_vec();
        self.allocation_status.arrays_allocated = true;
    }

    fn free_detached_ptrs(&self) 
    {
        if self.parameter_ptrs.weight_ptr_detached
        {
            free_cuda_array(self.parameter_ptrs.weight_ptr as *mut c_void);
            free_cuda_array(self.parameter_ptrs.weight_grad_ptr as *mut c_void);
        }
        
        free_cuda_array(self.parameter_ptrs.weight_vel_ptr as *mut c_void);
        free_cuda_array(self.parameter_ptrs.weight_moment_ptr as *mut c_void);

        free_cuda_array(self.rand_state_v_ptr);
        free_cuda_array(self.dropout_mask_ptr as *mut c_void);
    }
}