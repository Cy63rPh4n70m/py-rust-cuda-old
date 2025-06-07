use std::{collections::HashMap, ffi::c_void};

use crate::{
    cuda_bridge::{free_cuda_array, gradient_desc_3d, matmul_add_bias_back, matmul_add_bias_tiled, new_cuda_array}, math_functions::random_float_vec, 
        neuralnet::TraversePtrs, pointer_ops::{counter_is_zero, cuda_ptr_to_vec, 
            increment_counter, init_trav_in_ptrs, 
            set_zero_counter, vec_to_cuda_ptr}};

use super::layer_cuda::{LayerCuda, AllocationStatus, IOPtrs, ParameterPtrs, WeightTensors};

/// Dense layer performs the usual matrix dot product  
/// combined with bias addition  
/// supports batched matrix multiplication
pub struct DenseCuda
{
    pub name: String,
    pub use_bias: bool,
    pub batch_size: f32,
    
    pub io_ptrs: IOPtrs,
    pub parameter_ptrs: ParameterPtrs,
    pub allocation_status: AllocationStatus,
    pub weight_tensors: WeightTensors

}
impl DenseCuda
{
    // weight matrix initialize during first ever run
    pub fn new(
        n_in: usize, n_out: usize, batch: usize, rows: usize, use_bias: bool, name: &str
    ) -> Self
    {        
        let mut weight_tensors: WeightTensors = WeightTensors::new();

        let range: f32 = (6.0 / (n_in + n_out) as f32).sqrt();

        weight_tensors.weight = random_float_vec(
            batch * n_in * n_out, 
            -range, range
        );

        weight_tensors.biases = random_float_vec(
            batch * rows * n_out, 
            0.0, 0.0
        );

        return Self
        {
            name: name.to_string(),
            use_bias,
            batch_size: 0.0,
            io_ptrs: IOPtrs::new((batch, rows, n_in), (batch, rows, n_out)),
            parameter_ptrs: ParameterPtrs::new(),
            allocation_status: AllocationStatus::new(),
            weight_tensors
        }
    }
}

// trait implementation for dense layer
impl LayerCuda for DenseCuda
{
    fn forward(&mut self, trav_ptr_in: *mut TraversePtrs, trav_ptr_weight: *mut TraversePtrs, _use_dropout: bool) -> *mut TraversePtrs
    {
        let batch: usize = self.io_ptrs.in_shape.0;
        let rows: usize = self.io_ptrs.in_shape.1;
        let cols: usize = self.io_ptrs.in_shape.2;

        if !self.allocation_status.ptrs_allocated
        {   
            // initialize bias pointers
            self.parameter_ptrs.biases_ptr = vec_to_cuda_ptr(&mut self.weight_tensors.biases);
            self.parameter_ptrs.bias_grad_ptr = new_cuda_array(
                (batch * rows * self.io_ptrs.out_shape.2) as u32
            );
            self.parameter_ptrs.bias_vel_ptr = new_cuda_array(
                (batch * rows * self.io_ptrs.out_shape.2) as u32
            );
            self.parameter_ptrs.bias_moment_ptr = new_cuda_array(
                (batch * rows * self.io_ptrs.out_shape.2) as u32
            );

            // decide whether to create weight based on pointer availability
            if trav_ptr_weight.is_null()
            {
                // initialize weight ptrs
                self.parameter_ptrs.weight_ptr = vec_to_cuda_ptr(&mut self.weight_tensors.weight);
                self.parameter_ptrs.weight_grad_ptr = new_cuda_array(
                    (batch * cols * self.io_ptrs.out_shape.2) as u32
                );

                self.parameter_ptrs.weight_ptr_detached = true;
            }
            else
            {
                // set weight pointers with pointers in second traversal pointer
                // links the output of another layer with this layer
                unsafe
                {
                    self.parameter_ptrs.weight_ptr = (*trav_ptr_weight).ptr;
                    self.parameter_ptrs.weight_grad_ptr = (*trav_ptr_weight).grad_ptr;
                    self.io_ptrs.backward_count_weight_prev = (*trav_ptr_weight).backward_pass_count;
                }

                self.parameter_ptrs.weight_ptr_detached = false;
            }

            // initialize weight pointers needed for training
            self.parameter_ptrs.weight_vel_ptr = new_cuda_array(
                (batch * cols * self.io_ptrs.out_shape.2) as u32
            );
            self.parameter_ptrs.weight_moment_ptr = new_cuda_array(
                (batch * cols * self.io_ptrs.out_shape.2) as u32
            );

            self.allocation_status.arrays_allocated = true;
            self.allocation_status.ptrs_allocated = true;

            // connect the output pointers of the previous layer with the 
            // current layer's input pointers
            init_trav_in_ptrs(
                &trav_ptr_in, &mut self.io_ptrs.backward_count,
                &mut self.io_ptrs.backward_count_in_prev, 
                &mut self.io_ptrs.input_ptr, &mut self.io_ptrs.input_grad_ptr, 
                &mut self.io_ptrs.output_ptr, &mut self.io_ptrs.output_grad_ptr, 
                &mut self.io_ptrs.output_traverse_ptr, 
                batch * rows * self.io_ptrs.out_shape.2
            );
        }

        // call batched matrix multiplication method from CUDA library
        matmul_add_bias_tiled(
            self.io_ptrs.input_ptr, batch as u32, rows as u32, cols as u32, 
            self.parameter_ptrs.weight_ptr, batch as u32, cols as u32, self.io_ptrs.out_shape.2 as u32,
            self.io_ptrs.output_ptr, self.parameter_ptrs.biases_ptr, 
            self.use_bias, self.allocation_status.zero_output
        );

        set_zero_counter(self.io_ptrs.backward_count);

        return self.io_ptrs.output_traverse_ptr;

    }

    fn backward(&mut self, _use_dropout: bool)
    {
        // controls whether this layer will zero the gradients for the previous layer
        // only zeros when the counter in the previous layer is zero
        if !counter_is_zero(self.io_ptrs.backward_count_in_prev)
        {
            self.allocation_status.zero_input_grad = false;
        }

        if !counter_is_zero(self.io_ptrs.backward_count_weight_prev)
        {
            self.allocation_status.zero_weight_grad = false;
        }
        
        matmul_add_bias_back(
            self.io_ptrs.input_grad_ptr, 
            self.io_ptrs.in_shape.0 as u32, self.io_ptrs.in_shape.1 as u32, 
            self.io_ptrs.in_shape.2 as u32, 
            self.parameter_ptrs.weight_grad_ptr, 
            self.io_ptrs.in_shape.0 as u32, self.io_ptrs.in_shape.2 as u32, 
            self.io_ptrs.out_shape.2 as u32, 
            self.parameter_ptrs.bias_grad_ptr, 
            self.io_ptrs.out_shape.0 as u32, self.io_ptrs.out_shape.1 as u32, 
            self.io_ptrs.out_shape.2 as u32,
            self.io_ptrs.output_grad_ptr, 
            self.io_ptrs.input_ptr,
            self.parameter_ptrs.weight_ptr,
            self.use_bias,
            self.allocation_status.zero_input_grad,
            self.allocation_status.zero_weight_grad
        );

        increment_counter(self.io_ptrs.backward_count_in_prev);

        if !self.parameter_ptrs.weight_ptr_detached
        {
            increment_counter(self.io_ptrs.backward_count_weight_prev);
        }

        self.batch_size += 1.0;

    }

    fn update_params(&mut self, optimizer_type: i32, lr: f32, l2: f32, alpha: f32, beta: f32)
    {   
        gradient_desc_3d(
            lr, l2,
            self.parameter_ptrs.weight_ptr, self.parameter_ptrs.weight_grad_ptr, 
            self.parameter_ptrs.weight_vel_ptr, self.parameter_ptrs.weight_moment_ptr,
            self.io_ptrs.in_shape.0, self.io_ptrs.in_shape.2, self.io_ptrs.out_shape.2,
            
            self.parameter_ptrs.biases_ptr, self.parameter_ptrs.bias_grad_ptr, 
            self.parameter_ptrs.bias_vel_ptr, self.parameter_ptrs.bias_moment_ptr,
            self.io_ptrs.out_shape.0, self.io_ptrs.out_shape.1, self.io_ptrs.out_shape.2,
            
            false, false, self.batch_size, optimizer_type, alpha, beta
        );

        self.batch_size = 0.0;
    }

    fn details(&self)
    {
        println!("Layer type: DENSE | Layer name: {:?}", self.name);
        println!("Input ptr: {:?} | Input grad ptr: {:?}", self.io_ptrs.input_ptr, self.io_ptrs.input_grad_ptr);
        println!("Weight ptr: {:?} | Weight grad ptr: {:?}", self.parameter_ptrs.weight_ptr, self.parameter_ptrs.weight_grad_ptr);
        println!("Output ptr: {:?} | Output grad ptr: {:?}", self.io_ptrs.output_ptr, self.io_ptrs.output_grad_ptr);
        println!("Input shape: {:?}", self.io_ptrs.in_shape);
        println!("Output shape: {:?}", self.io_ptrs.out_shape);
        println!("Weights: \n{:?}", self.weight_tensors.weight);
        if self.use_bias
        {
            println!("Biases: \n{:?}", self.weight_tensors.biases);
        }
    }

    fn get_param_count(&self) -> usize
    {
        let mut count: usize = 0;
        if self.use_bias
        {
            count += self.io_ptrs.out_shape.0 * self.io_ptrs.out_shape.1 * self.io_ptrs.out_shape.2;
        }

        count += self.io_ptrs.in_shape.0 * self.io_ptrs.in_shape.2 * self.io_ptrs.out_shape.2;

        return count
    }

    fn move_ptrs_to_arrays(&mut self)
    {
        self.weight_tensors.weight = cuda_ptr_to_vec(
            self.parameter_ptrs.weight_ptr, 
            self.io_ptrs.in_shape.0 * self.io_ptrs.in_shape.2 * self.io_ptrs.out_shape.2
        );
        
        self.weight_tensors.biases = cuda_ptr_to_vec(
            self.parameter_ptrs.biases_ptr, 
            self.io_ptrs.out_shape.0 * self.io_ptrs.out_shape.1 * self.io_ptrs.out_shape.2
        );
    }

    fn get_weights_hashmap(&mut self) -> Option<HashMap<&str, Vec<f32>>>
    {
        self.move_ptrs_to_arrays();
        let mut hashmap: HashMap<&str, Vec<f32>> = HashMap::new();
        hashmap.insert("weights", self.weight_tensors.weight.clone());

        if self.use_bias
        {
            hashmap.insert("biases", self.weight_tensors.biases.clone());
        }

        return Some(hashmap);
    }

    fn load_weights_from_hashmap(
        &mut self, json_hashmap: &HashMap<&str, Vec<f32>>
    ) 
    {
        self.weight_tensors.weight = json_hashmap.get("weights").unwrap().to_vec();

        if self.use_bias
        {
            self.weight_tensors.biases = json_hashmap.get("biases").unwrap().to_vec();
        }

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
        
        free_cuda_array(self.parameter_ptrs.biases_ptr as *mut c_void);
        free_cuda_array(self.parameter_ptrs.bias_grad_ptr as *mut c_void);
        free_cuda_array(self.parameter_ptrs.bias_vel_ptr as *mut c_void);
        free_cuda_array(self.parameter_ptrs.bias_moment_ptr as *mut c_void);
    }
}