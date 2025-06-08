use std::{collections::HashMap, ffi::c_void, process::exit};

use crate::{cuda_bridge::{conv2d_backward, conv2d_forward, free_cuda_array, gradient_desc_3d, new_cuda_array, zeroes_3d_inplace}, math_functions::random_float_vec, 
     neuralnet::TraversePtrs, pointer_ops::{counter_is_zero, cuda_ptr_to_vec, 
         increment_counter, init_trav_in_ptrs, set_zero_counter, vec_to_cuda_ptr}};

use super::layer_cuda::{AllocationStatus, IOPtrs, LayerCuda, ParameterPtrs, WeightTensors};

/// convolutional layer, mainly used for image processing
/// perform parallel filter computation on 3D tensors
pub struct Conv2dCuda
{
    pub name: String,
    pub n_filters: usize,
    pub in_channels: usize,
    pub strides: usize,
    pub stride_count_y: usize,
    pub stride_count_x: usize,
    pub filter_dim: usize,

    pub io_ptrs: IOPtrs,
    pub parameter_ptrs: ParameterPtrs,
    pub allocation_status: AllocationStatus,
    pub weight_tensors: WeightTensors,
    
    pub filters_grad_count_ptr: *mut f32,
    pub input_grads_count_ptr: *mut f32,

    pub batch_size: f32,
    pub use_bias: bool,
}

// implement constructor
impl Conv2dCuda
{
    pub fn new(
        n_filters: usize, filter_dim: usize, strides: usize,
        batch: usize, rows: usize, cols: usize, name: &str
    ) -> Self
    {
        // calculate the 2D output shape given filter dimensions and stride
        // exits if y or x dimension is zero
        let stride_count_y: usize = ((rows - filter_dim) / strides) + 1;
        let stride_count_x: usize = ((cols - filter_dim) / strides) + 1;

        if stride_count_y == 0 || stride_count_x == 0
        {
            println!("Error: Convolutional output is zero");
            exit(1);
        }

        // initialise the filter weights and biases
        let in_shape: (usize, usize, usize) = (batch, rows, cols);
        let out_shape: (usize, usize, usize) = (n_filters, stride_count_y, stride_count_x);
        
        let mut weight_tensors: WeightTensors = WeightTensors::new();

        let range: f32 = 
            (6.0 / 
                ((batch * filter_dim * filter_dim + 
                 n_filters * filter_dim * filter_dim) as f32)).sqrt();
        let weight_len: usize = n_filters * batch * filter_dim * filter_dim;
        let output_len: usize = n_filters * stride_count_y * stride_count_x;

        weight_tensors.weight = random_float_vec(
            weight_len, 
            -range, range
        );

        weight_tensors.biases = random_float_vec(
            output_len,
            0.0, 0.0
        );
        
        return Self
        {
            name: name.to_string(),

            n_filters,
            filter_dim,
            strides,
            in_channels: 0,
            stride_count_y: 0,
            stride_count_x: 0,
            batch_size: 0.0,
            use_bias: true,

            io_ptrs: IOPtrs::new(in_shape, out_shape),
            parameter_ptrs: ParameterPtrs::new(),
            allocation_status: AllocationStatus::new(),
            weight_tensors,
            
            input_grads_count_ptr: std::ptr::null_mut(),
            filters_grad_count_ptr: std::ptr::null_mut(),
        }
    }

}

// trait implementation
impl LayerCuda for Conv2dCuda
{
    fn forward(&mut self, trav_ptr_in: *mut TraversePtrs, _trav_ptr_weight: *mut TraversePtrs, _use_dropout: bool) -> *mut TraversePtrs
    {
        let batch: usize = self.io_ptrs.in_shape.0;
        let rows: usize = self.io_ptrs.in_shape.1;
        let cols: usize = self.io_ptrs.in_shape.2;

        // initialize pointers if not already
        if !self.allocation_status.ptrs_allocated
        {   
            self.stride_count_y = ((rows - self.filter_dim) / self.strides) + 1;
            self.stride_count_x = ((cols - self.filter_dim) / self.strides) + 1;

            self.in_channels = batch;

            let weight_len: u32 = (self.n_filters * self.in_channels * self.filter_dim * self.filter_dim) as u32;
            let output_len: u32 = (self.n_filters * self.stride_count_y * self.stride_count_x) as u32;
            
            // initialize bias pointers
            self.parameter_ptrs.biases_ptr = vec_to_cuda_ptr(&mut self.weight_tensors.biases);
            self.parameter_ptrs.bias_grad_ptr = new_cuda_array(output_len);
            self.parameter_ptrs.bias_vel_ptr = new_cuda_array(output_len);
            self.parameter_ptrs.bias_moment_ptr = new_cuda_array(output_len);

            self.input_grads_count_ptr = new_cuda_array((batch * rows * cols) as u32);
            
            // initialise weight/filter pointers
            self.parameter_ptrs.weight_ptr = vec_to_cuda_ptr(&mut self.weight_tensors.weight);
            self.parameter_ptrs.weight_grad_ptr = new_cuda_array(weight_len);
            self.parameter_ptrs.weight_vel_ptr = new_cuda_array(weight_len);
            self.parameter_ptrs.weight_moment_ptr = new_cuda_array(weight_len);

            self.filters_grad_count_ptr = new_cuda_array(weight_len);

            // connect the output pointers of the previous layer with the 
            // current layer's input pointers
            init_trav_in_ptrs(
                &trav_ptr_in, &mut self.io_ptrs.backward_count,
                &mut self.io_ptrs.backward_count_in_prev, 
                &mut self.io_ptrs.input_ptr, &mut self.io_ptrs.input_grad_ptr, 
                &mut self.io_ptrs.output_ptr, &mut self.io_ptrs.output_grad_ptr, 
                &mut self.io_ptrs.output_traverse_ptr, 
                self.io_ptrs.out_shape.0 * self.io_ptrs.out_shape.1 * self.io_ptrs.out_shape.2
            );

            self.allocation_status.ptrs_allocated = true;
            self.allocation_status.arrays_allocated = true;
        }

        // CUDA method for convolutional forward pass
        conv2d_forward(
            self.io_ptrs.input_ptr, 
            self.parameter_ptrs.weight_ptr, 
            self.io_ptrs.output_ptr,
            batch as u32, rows as u32, cols as u32, 
            self.n_filters as u32, self.stride_count_y as u32, self.stride_count_x as u32,
            self.filter_dim as u32, self.strides as u32, self.parameter_ptrs.biases_ptr, 
            self.allocation_status.zero_output
        );

        // important for zeroing gradients during backward pass
        set_zero_counter(self.io_ptrs.backward_count);

        return self.io_ptrs.output_traverse_ptr;
    }

    fn backward(&mut self, _use_dropout: bool)
    {
        // zero input gradients
        if counter_is_zero(self.io_ptrs.backward_count_in_prev)
        {
            zeroes_3d_inplace(
                self.io_ptrs.input_grad_ptr, 
                self.io_ptrs.in_shape.0, self.io_ptrs.in_shape.1, self.io_ptrs.in_shape.2
            );
        }
        
        // CUDA function to calculate filter and input gradients
        conv2d_backward(
            self.io_ptrs.input_ptr, self.io_ptrs.input_grad_ptr, self.input_grads_count_ptr,
            self.parameter_ptrs.weight_ptr, self.parameter_ptrs.weight_grad_ptr, 
            self.filters_grad_count_ptr,
            self.io_ptrs.output_grad_ptr, self.parameter_ptrs.bias_grad_ptr,
            self.io_ptrs.in_shape.0 as u32, self.io_ptrs.in_shape.1 as u32, self.io_ptrs.in_shape.2 as u32, 
            self.io_ptrs.out_shape.0 as u32, self.io_ptrs.out_shape.1 as u32, self.io_ptrs.out_shape.2 as u32, 
            self.filter_dim as u32, self.strides as u32
        );

        // increment counter to tell other layers connected to the same previous layer
        // to accumulate the gradient instead of zeroing it first
        increment_counter(self.io_ptrs.backward_count_in_prev);
        self.batch_size += 1.0;
    }

    fn update_params(&mut self, optimizer_type: i32, lr: f32, l2: f32, alpha: f32, beta: f32)
    {   
        // perform gradient descent with SGD or AdamW
        gradient_desc_3d(
            lr, l2,
            self.parameter_ptrs.weight_ptr, self.parameter_ptrs.weight_grad_ptr, 
            self.parameter_ptrs.weight_vel_ptr, self.parameter_ptrs.weight_moment_ptr,
             self.n_filters * self.in_channels, self.filter_dim, self.filter_dim,
            self.parameter_ptrs.biases_ptr, self.parameter_ptrs.bias_grad_ptr, 
            self.parameter_ptrs.bias_vel_ptr, self.parameter_ptrs.bias_moment_ptr,
            self.io_ptrs.out_shape.0, self.io_ptrs.out_shape.1, self.io_ptrs.out_shape.2,
            self.use_bias, false, self.batch_size, optimizer_type, alpha, beta
        );

        self.batch_size = 0.0;
    }

    fn details(&self)
    {
        // print all details of layer (e.g. IO shape, weights, etc)
        println!("Layer type: CONV_2D | Layer name: {:?}", self.name);
        println!("Input shape: {:?}", self.io_ptrs.in_shape);
        println!("Output shape: {:?}", self.io_ptrs.out_shape);
        println!("Filters: \n{:?}", self.weight_tensors.weight);
        println!("Biases: \n{:?}", self.weight_tensors.biases);
    }

    fn get_param_count(&self) -> usize
    {
        return self.n_filters * self.in_channels * self.filter_dim * self.filter_dim +
               self.io_ptrs.out_shape.0 * self.io_ptrs.out_shape.1 * self.io_ptrs.out_shape.2;
    }

    fn move_ptrs_to_arrays(&mut self)
    {
        self.weight_tensors.weight = cuda_ptr_to_vec(
            self.parameter_ptrs.weight_ptr, 
            self.n_filters * self.in_channels * self.filter_dim * self.filter_dim
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
        free_cuda_array(self.parameter_ptrs.weight_ptr as *mut c_void);
        free_cuda_array(self.parameter_ptrs.weight_grad_ptr as *mut c_void);
        free_cuda_array(self.parameter_ptrs.weight_vel_ptr as *mut c_void);
        free_cuda_array(self.parameter_ptrs.weight_moment_ptr as *mut c_void);
        
        free_cuda_array(self.parameter_ptrs.biases_ptr as *mut c_void);
        free_cuda_array(self.parameter_ptrs.bias_grad_ptr as *mut c_void);
        free_cuda_array(self.parameter_ptrs.bias_vel_ptr as *mut c_void);
        free_cuda_array(self.parameter_ptrs.bias_moment_ptr as *mut c_void);

        free_cuda_array(self.input_grads_count_ptr as *mut c_void);
        free_cuda_array(self.filters_grad_count_ptr as *mut c_void);
    }
}