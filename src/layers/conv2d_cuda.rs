use std::process::exit;

use crate::{cuda_bridge::{conv2d_backward, 
    conv2d_forward, gradient_desc_3d, new_cuda_array,
     zeroes_3d_inplace}, math_functions::random_float_vec, 
     neuralnet::TraversePtrs, pointer_ops::{counter_is_zero, cuda_ptr_to_vec, 
         increment_counter, init_trav_in_ptrs, set_zero_counter, vec_to_cuda_ptr}};

use super::layer_cuda::{AllocationStatus, IOPtrs, LayerCuda, ParameterPtrs, WeightTensors};

pub struct Conv2dCuda
{
    pub name: String,
    pub n_filters: usize,
    pub in_channels: usize,
    pub strides: usize, // move interval
    pub stride_count_y: usize, // how many of these moves, output dimension
    pub stride_count_x: usize, // how many of these moves, output dimension
    pub filter_dim: usize,

    pub io_ptrs: IOPtrs,
    pub parameter_ptrs: ParameterPtrs,
    pub allocation_status: AllocationStatus,
    pub weight_tensors: WeightTensors,
    
    pub filters_grad_count_ptr: *mut f32,
    pub input_grads_count_ptr: *mut f32,

    pub batch_size: f32,
    pub count: u128,

    pub use_bias: bool,
    pub flatten: bool
}
impl Conv2dCuda
{
    // weight matrix initialize during first ever run
    pub fn new(
        n_filters: usize, filter_dim: usize, strides: usize,
        batch: usize, rows: usize, cols: usize, flatten: bool, name: &str
    ) -> Self
    {
        let in_shape: (usize, usize, usize) = (batch, rows, cols);
        let out_shape: (usize, usize, usize) = (n_filters, ((rows - filter_dim) / strides) + 1, ((cols - filter_dim) / strides) + 1);
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
            count: 0,
            use_bias: true,
            flatten,

            io_ptrs: IOPtrs::new(in_shape, out_shape),
            parameter_ptrs: ParameterPtrs::new(),
            allocation_status: AllocationStatus::new(),
            weight_tensors: WeightTensors::new(),
            
            input_grads_count_ptr: std::ptr::null_mut(),
            filters_grad_count_ptr: std::ptr::null_mut(),
        }
    }

}
impl LayerCuda for Conv2dCuda
{
    // supports batch matrix multiplication unlike cpu
    fn forward(&mut self, trav_ptr_in: *mut TraversePtrs, _trav_ptr_weight: *mut TraversePtrs, _use_dropout: bool) -> *mut TraversePtrs
    {
        //println!("{:?}", cuda_ptr_to_array(input.get_ptr(), input_shape));
        let batch: usize = self.io_ptrs.in_shape.0;
        let rows: usize = self.io_ptrs.in_shape.1;
        let cols: usize = self.io_ptrs.in_shape.2;

        //println!("{:?}, {:?}, {:?}, {:?}", input.get_ptr(), batch, rows, cols);

        if !self.allocation_status.ptrs_allocated
        {   
            self.stride_count_y = ((rows - self.filter_dim) / self.strides) + 1;
            self.stride_count_x = ((cols - self.filter_dim) / self.strides) + 1;

            if self.stride_count_y == 0 || self.stride_count_x == 0
            {
                println!("Error: Convolutional output is zero");
                exit(1);
            }

            self.in_channels = batch;

            let weight_len: u32 = (self.n_filters * self.in_channels * self.filter_dim * self.filter_dim) as u32;
            let output_len: u32 = (self.n_filters * self.stride_count_y * self.stride_count_x) as u32;

            // initialise biases
            if !self.allocation_status.arrays_allocated
            {
                self.weight_tensors.biases = random_float_vec(
                    self.n_filters * self.stride_count_y * self.stride_count_x, 
                    -0.001, 0.001
                );
            }

            self.parameter_ptrs.biases_ptr = vec_to_cuda_ptr(&mut self.weight_tensors.biases);
            self.parameter_ptrs.bias_grad_ptr = new_cuda_array(output_len);
            self.parameter_ptrs.bias_vel_ptr = new_cuda_array(output_len);
            self.parameter_ptrs.bias_moment_ptr = new_cuda_array(output_len);

            self.input_grads_count_ptr = new_cuda_array((batch * rows * cols) as u32);

            if !self.allocation_status.arrays_allocated
            {
                // initialise weights and weight pointer
                let range: f32 = 
                    (6.0 / 
                        ((batch * self.filter_dim * self.filter_dim + 
                         self.n_filters * self.filter_dim * self.filter_dim) as f32)).sqrt();

                self.weight_tensors.weight = random_float_vec(
                    weight_len as usize,
                    -range, range
                );
            }
    
            self.parameter_ptrs.weight_ptr = vec_to_cuda_ptr(&mut self.weight_tensors.weight);
            self.parameter_ptrs.weight_grad_ptr = new_cuda_array(weight_len);
            self.parameter_ptrs.weight_vel_ptr = new_cuda_array(weight_len);
            self.parameter_ptrs.weight_moment_ptr = new_cuda_array(weight_len);

            self.filters_grad_count_ptr = new_cuda_array(weight_len);

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

        //println!("input: {:?}", cuda_ptr_to_array(input_ptr, &[batch, rows, cols]));
        //println!("filter: {:?}", cuda_ptr_to_array(filter_ptr, &[self.n_filters, batch, self.filter_dim, self.filter_dim]));
        conv2d_forward(
            self.io_ptrs.input_ptr, 
            self.parameter_ptrs.weight_ptr, 
            self.io_ptrs.output_ptr,
            batch as u32, rows as u32, cols as u32, 
            self.n_filters as u32, self.stride_count_y as u32, self.stride_count_x as u32,
            self.filter_dim as u32, self.strides as u32, self.parameter_ptrs.biases_ptr, 
            self.allocation_status.zero_output
        );

        set_zero_counter(self.io_ptrs.backward_count);

        return self.io_ptrs.output_traverse_ptr;
        //println!("output: {:?}", cuda_ptr_to_array(result_ptr, &[self.n_filters, self.stride_count_y, self.stride_count_x]));
        //element_op_3d_inplace(result_ptr, bias_ptr, 0, self.n_filters, self.stride_count_y, self.stride_count_x);

        //println!("-----------------------------");
        //println!("{:?}", cuda_ptr_to_array(input_ptr, &[batch, rows, cols]));
        //println!("{:?}", cuda_ptr_to_array(weight_ptr, &[batch, cols, self.n_out]));
        // overwrite the current pointer with result ptr, to be COPIED to input of next layer
        // current pointer is already recorded by previous layer, don't free

        //println!("{:?}", cuda_ptr_to_array(input.get_ptr(), input.get_shape()));
        //println!("-----------------------------");
        //exit(1);
    }

    fn backward(&mut self, _use_dropout: bool)
    {
        if counter_is_zero(self.io_ptrs.backward_count_in_prev)
        {
            zeroes_3d_inplace(
                self.io_ptrs.input_grad_ptr, 
                self.io_ptrs.in_shape.0, self.io_ptrs.in_shape.1, self.io_ptrs.in_shape.2
            );
        }
        
        conv2d_backward(
            self.io_ptrs.input_ptr, self.io_ptrs.input_grad_ptr, self.input_grads_count_ptr,
            self.parameter_ptrs.weight_ptr, self.parameter_ptrs.weight_grad_ptr, 
            self.filters_grad_count_ptr,
            self.io_ptrs.output_grad_ptr, self.parameter_ptrs.bias_grad_ptr,
            self.io_ptrs.in_shape.0 as u32, self.io_ptrs.in_shape.1 as u32, self.io_ptrs.in_shape.2 as u32, 
            self.io_ptrs.out_shape.0 as u32, self.io_ptrs.out_shape.1 as u32, self.io_ptrs.out_shape.2 as u32, 
            self.filter_dim as u32, self.strides as u32
        );

        //println!("{:?}", cuda_ptr_to_array(input_grad_ptr, &[self.in_shape.0, self.in_shape.1, self.in_shape.2]));
        //println!("{:?}", cuda_ptr_to_array(filter_grad_ptr, &[self.n_filters, self.in_channels, self.filter_dim, self.filter_dim]));
        increment_counter(self.io_ptrs.backward_count);
        //exit(1);
        self.batch_size += 1.0;
        //let end = start.elapsed();
        //println!("backward: {:.6}", end.as_secs_f64());

        //println!("\noriginal_grads: {:?}", cuda_ptr_to_array(ptr.get_ptr(), ptr.get_shape()));
        //ptr.set_ptr(input_grad_ptr, vec![self.in_shape.0, self.in_shape.1, self.in_shape.2]);
        //println!("\nchained_gradients: {:?}", cuda_ptr_to_array(ptr.get_ptr(), ptr.get_shape()));
        //println!("\nweight_gradients: {:?}", cuda_ptr_to_array(weight_grad_ptr, &[self.in_shape.0, self.in_shape.2, self.out_shape.2]));
        //println!("\nbias_gradients: {:?}", cuda_ptr_to_array(bias_grad_ptr, &[self.out_shape.0, self.out_shape.1, self.out_shape.2]));
        //println!("=========================================================");
        //exit(1);      
        // calculate summed respect to bias
        // calculate bias gradients
    }

    fn update_params(&mut self, optimizer_type: i32, lr: f32, l2: f32, alpha: f32, beta: f32)
    {   
        //println!("filter: {:?}", cuda_ptr_to_array(filter_ptr, &[self.n_filters, self.in_channels, self.filter_dim, self.filter_dim]));
        //println!("filter_grad: {:?}", cuda_ptr_to_array(filter_grad_ptr, &[self.n_filters, self.in_channels, self.filter_dim, self.filter_dim]));
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

        //println!("filter after: {:?}", cuda_ptr_to_array(filter_ptr, &[self.n_filters, self.in_channels, self.filter_dim, self.filter_dim]));
        //println!("filter_grad after: {:?}", cuda_ptr_to_array(filter_grad_ptr, &[self.n_filters, self.in_channels, self.filter_dim, self.filter_dim]));

        self.batch_size = 0.0;

        //println!("{:?}", cuda_ptr_to_array(weight_grad_ptr, &[self.in_shape.0, self.in_shape.2, self.out_shape.2]))
        //self.weights -= &(self.lr * (&self.weight_gradients + self.l2 * &self.weights));
        //self.biases -= &(self.lr * &self.bias_gradients);
    }

    fn details(&self)
    {
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