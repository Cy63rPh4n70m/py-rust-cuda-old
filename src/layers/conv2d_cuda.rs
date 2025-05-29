use std::process::exit;

use ndarray::{ArrayD, IxDyn};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{cuda_bridge::{conv2d_backward, conv2d_forward, gradient_desc_3d, zeroes_3d_inplace}, pointer_ops::{array_to_cuda_ptr_str, counter_is_zero, cuda_ptr_to_array, increment_counter, init_layer_connections, new_cuda_ptr_str, set_zero_counter, string_to_ptr}};

#[derive(Serialize, Deserialize)]
pub struct Conv2dCuda
{
    pub name: String,
    pub in_shape: (usize, usize, usize),
    pub out_shape: (usize, usize, usize),
    pub n_filters: usize,
    pub in_channels: usize,
    pub strides: usize, // move interval
    pub stride_count_y: usize, // how many of these moves, output dimension
    pub stride_count_x: usize, // how many of these moves, output dimension
    pub filter_dim: usize,

    pub input: ArrayD<f32>,
    pub filters: ArrayD<f32>,
    pub biases: ArrayD<f32>,

    //pub io_ptrs: HashMap<String, String>,

    pub filters_ptr: String,
    pub filters_grad_count_ptr: String,
    pub filter_gradients_ptr: String,
    pub filter_velocity_ptr: String,
    pub filter_momentum_ptr: String,
    pub input_grads_count_ptr: String,
    pub biases_ptr: String,
    pub bias_gradients_ptr: String,
    pub bias_velocity_ptr: String,
    pub bias_momentum_ptr: String,

    pub backward_count: String,
    pub backward_count_prev: String,

    pub input_ptr: String,
    pub input_grad_ptr: String,
    pub output_ptr: String,
    pub output_grad_ptr: String,

    pub output_traverse_ptr: String,
    
    //pub result_ptr: String,
    //pub result_ptr_t: String,

    pub batch_size: f32,
    pub count: u128,

    pub filter_ptr_allocated: bool, 
    pub bias_ptr_allocated: bool, 
    pub filter_array_allocated: bool,
    pub bias_array_allocated: bool,
    //pub grads_ptr_allocated: bool,
    pub use_bias: bool,
    pub zero_output: bool,
    pub zero_input_grad: bool,
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
        let filters: ArrayD<f32> = ArrayD::zeros(IxDyn(&[0]));
        let biases: ArrayD<f32> = ArrayD::zeros(IxDyn(&[0]));

        return Self
        {
            name: name.to_string(),
            input: ArrayD::zeros(IxDyn(&[0])),
            filters,
            filters_ptr: "".to_string(),
            filter_gradients_ptr: "".to_string(),
            filters_grad_count_ptr: "".to_string(),
            filter_velocity_ptr: "".to_string(),
            filter_momentum_ptr: "".to_string(),
            
            //io_ptrs,

            input_grads_count_ptr: "".to_string(),

            biases,
            biases_ptr: "".to_string(),
            bias_gradients_ptr: "".to_string(),
            bias_velocity_ptr: "".to_string(),
            bias_momentum_ptr: "".to_string(),

            input_ptr: "".to_string(),
            input_grad_ptr: "".to_string(),
            output_ptr: "".to_string(),
            output_grad_ptr: "".to_string(),

            output_traverse_ptr: "".to_string(),

            backward_count: String::from("none"),
            backward_count_prev: String::from("none"),
            
            //broadcast_array: ArrayD::zeros(IxDyn(&[0])),
            //broadcast_array_ptr: "".to_string(),
            in_shape: (batch, rows, cols),
            out_shape: (n_filters, ((rows - filter_dim) / strides) + 1, ((cols - filter_dim) / strides) + 1),
            n_filters,
            filter_dim,
            strides,
            in_channels: 0,
            stride_count_y: 0,
            stride_count_x: 0,
            batch_size: 0.0,
            count: 0,
            filter_ptr_allocated: false, 
            bias_ptr_allocated: false,
            filter_array_allocated: false,
            bias_array_allocated: false,
            //grads_ptr_allocated: false,
            use_bias: true,
            zero_output: true,
            zero_input_grad: false,
            flatten
        }
    }

    // supports batch matrix multiplication unlike cpu
    pub fn forward(&mut self, str_ptr_in: String) -> String
    {
        //println!("{:?}", cuda_ptr_to_array(input.get_ptr(), input_shape));
        let batch: usize = self.in_shape.0;
        let rows: usize = self.in_shape.1;
        let cols: usize = self.in_shape.2;

        //println!("{:?}, {:?}, {:?}, {:?}", input.get_ptr(), batch, rows, cols);

        if !self.bias_ptr_allocated
        {   
            self.stride_count_y = ((rows - self.filter_dim) / self.strides) + 1;
            self.stride_count_x = ((cols - self.filter_dim) / self.strides) + 1;

            if self.stride_count_y == 0 || self.stride_count_x == 0
            {
                println!("Error: Convolutional output is zero");
                exit(1);
            }

            // initialise biases and pointers
            if !self.bias_array_allocated
            {

                self.biases = ArrayD::zeros(IxDyn(&[self.n_filters, self.stride_count_y, self.stride_count_x]));
                self.bias_array_allocated = true
            }

            self.biases_ptr = array_to_cuda_ptr_str(&mut self.biases);
            self.bias_gradients_ptr = new_cuda_ptr_str(&[self.n_filters, self.stride_count_y, self.stride_count_x]);
            self.bias_velocity_ptr = new_cuda_ptr_str(&[self.n_filters, self.stride_count_y, self.stride_count_x]);
            self.bias_momentum_ptr = new_cuda_ptr_str(&[self.n_filters, self.stride_count_y, self.stride_count_x]);

            // initialise input pointer, set the input as the result pointer from previous layer
            // tensor struct at this stage will contain the result ptr of the previous layer
            ////////////////////////////////////////////////////////////////
            //self.input_ptr = input.get_ptr_as_str();
            //self.input_t_ptr = new_cuda_ptr_str(&[batch, cols, rows]);
            //////////////////////////////////////////////////////////////////

            // initialise the result tensor/pointer
            //self.result_ptr = new_cuda_ptr_str(&[self.n_filters, self.stride_count_y, self.stride_count_x]);
            //self.result_ptr_t = new_cuda_ptr_str(&[batch, self.n_out, rows]);
            
            //self.input_grads_ptr = new_cuda_ptr_str(&[batch, rows, cols]);
            self.input_grads_count_ptr = new_cuda_ptr_str(&[batch, rows, cols]);
            //self.output_grads_ptr = new_cuda_ptr_str(&[self.n_filters, self.stride_count_y, self.stride_count_x]);

            //self.in_shape = (batch, rows, cols);
            //self.out_shape = (self.n_filters, self.stride_count_y, self.stride_count_x);
            self.in_channels = batch;

            self.bias_ptr_allocated = true;

            if !self.filter_ptr_allocated
            {
                if !self.filter_array_allocated
                {
                    // initialise weights and weight pointer
                    let range: f32 = 
                        (6.0 / 
                            ((batch * self.filter_dim * self.filter_dim + 
                             self.n_filters * self.filter_dim * self.filter_dim) as f32)).sqrt();

                    self.filters = ArrayD::from_shape_fn(
                        IxDyn(&[self.n_filters, self.in_channels, self.filter_dim, self.filter_dim]), 
                        |_| rand::thread_rng().gen_range(-range..range)
                    );

                    self.filter_array_allocated = true;
                }

                //self.weights = Array3::from_shape_fn(
                //    (batch, cols, self.n_out), 
                //    |(i, j, k)|
                //    {
                //        (i * cols * self.n_out + j * self.n_out + k) as f32
                //    }
                //).into_dyn() / (batch * cols * self.n_out) as f32;
                self.filters_ptr = array_to_cuda_ptr_str(&mut self.filters);
                self.filter_gradients_ptr = new_cuda_ptr_str(&[self.n_filters, self.in_channels, self.filter_dim, self.filter_dim]);
                self.filters_grad_count_ptr = new_cuda_ptr_str(&[self.n_filters, self.in_channels, self.filter_dim, self.filter_dim]);
                self.filter_velocity_ptr = new_cuda_ptr_str(&[self.n_filters, self.in_channels, self.filter_dim, self.filter_dim]);
                self.filter_momentum_ptr = new_cuda_ptr_str(&[self.n_filters, self.in_channels, self.filter_dim, self.filter_dim]);

                init_layer_connections(
                    &mut self.backward_count, &str_ptr_in, 
                    &mut self.backward_count_prev, &mut self.input_ptr, 
                    &mut self.input_grad_ptr, &mut self.output_ptr, 
                    &mut self.output_grad_ptr, &mut self.output_traverse_ptr, 
                    &[self.out_shape.0, self.out_shape.1, self.out_shape.2]
                );

                self.filter_ptr_allocated = true;
            }
        }

        //let broadcast_buf: String = new_cuda_ptr_str(batch * rows * cols * self.n_out);

        // convert strings to pointers
        let input_ptr: *mut f32 = string_to_ptr(&self.input_ptr);
        let filter_ptr: *mut f32 = string_to_ptr(&self.filters_ptr);
        let result_ptr: *mut f32 = string_to_ptr(&self.output_ptr);
        let bias_ptr: *mut f32 = string_to_ptr(&self.biases_ptr);

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

        //println!("input: {:?}", cuda_ptr_to_array(input_ptr, &[batch, rows, cols]));
        //println!("filter: {:?}", cuda_ptr_to_array(filter_ptr, &[self.n_filters, batch, self.filter_dim, self.filter_dim]));
        conv2d_forward(
            input_ptr, filter_ptr, result_ptr,
            batch as u32, rows as u32, cols as u32, 
            self.n_filters as u32, self.stride_count_y as u32, self.stride_count_x as u32,
            self.filter_dim as u32, self.strides as u32, bias_ptr, self.zero_output
        );

        set_zero_counter(&self.backward_count);

        return self.output_traverse_ptr.clone();
        //println!("output: {:?}", cuda_ptr_to_array(result_ptr, &[self.n_filters, self.stride_count_y, self.stride_count_x]));
        //element_op_3d_inplace(result_ptr, bias_ptr, 0, self.n_filters, self.stride_count_y, self.stride_count_x);

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
        //let input_t_ptr: *mut f32 = string_to_ptr(&self.input_t_ptr);
        let filter_ptr: *mut f32 = string_to_ptr(&self.filters_ptr);
        //let weight_t_ptr: *mut f32 = string_to_ptr(&self.weight_t_ptr);
        let input_grad_ptr: *mut f32 = string_to_ptr(&self.input_grad_ptr);
        let input_grad_count_ptr: *mut f32 = string_to_ptr(&self.input_grads_count_ptr);
        let filter_grad_ptr: *mut f32 = string_to_ptr(&self.filter_gradients_ptr);
        let filter_grad_count_ptr: *mut f32 = string_to_ptr(&self.filters_grad_count_ptr);
        let bias_grad_ptr: *mut f32 = string_to_ptr(&self.bias_gradients_ptr);
        //let result_ptr: *mut f32 = string_to_ptr(self.io_ptrs.get_mut("output").unwrap());
        let original_grads: *mut f32 = string_to_ptr(&self.output_grad_ptr);

        //println!("=========================================================");
        //println!("input_array: {:?}", cuda_ptr_to_array(input_ptr, &[self.in_shape.0, self.in_shape.1, self.in_shape.2]));
        //println!("\nweights: {:?}", cuda_ptr_to_array(weight_ptr, &[self.in_shape.0, self.in_shape.2, self.out_shape.2]));
        //println!("\nmatmul result: {:?}", cuda_ptr_to_array(result_ptr, &[self.out_shape.0, self.out_shape.1, self.out_shape.2]));

        //let start: Instant = Instant::now();

        // array is always stored as 1d
        // bias gradients
        //println!("{:?}", cuda_ptr_to_array(original_grads, &[self.n_filters, self.stride_count_y, self.stride_count_x]));
        //element_op_3d_inplace(bias_grad_ptr, original_grads, 0, self.n_filters, self.stride_count_y, self.stride_count_x);
        //if self.zero_input_grad
        //{
        //    zeroes_3d_inplace(input_grad_ptr, self.in_shape.0, self.in_shape.1, self.in_shape.2);
        //}

        if counter_is_zero(&self.backward_count_prev)
        {
            zeroes_3d_inplace(input_grad_ptr, self.in_shape.0, self.in_shape.1, self.in_shape.2);
        }
        
        conv2d_backward(
            input_ptr, input_grad_ptr, input_grad_count_ptr,
            filter_ptr, filter_grad_ptr, filter_grad_count_ptr,
            original_grads, bias_grad_ptr,
            self.in_shape.0 as u32, self.in_shape.1 as u32, self.in_shape.2 as u32, 
            self.out_shape.0 as u32, self.out_shape.1 as u32, self.out_shape.2 as u32, 
            self.filter_dim as u32, self.strides as u32
        );

        //println!("{:?}", cuda_ptr_to_array(input_grad_ptr, &[self.in_shape.0, self.in_shape.1, self.in_shape.2]));
        //println!("{:?}", cuda_ptr_to_array(filter_grad_ptr, &[self.n_filters, self.in_channels, self.filter_dim, self.filter_dim]));
        increment_counter(&self.backward_count);
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

    pub fn get_weight_grads_ptr(&mut self) -> String
    {
        return self.filter_gradients_ptr.clone();
    }

    pub fn update_params(&mut self, optimizer_type: i32, lr: f32, l2: f32, alpha: f32, beta: f32, only_bias: bool)
    {   
        let filter_ptr: *mut f32 = string_to_ptr(&self.filters_ptr);
        let bias_ptr: *mut f32 = string_to_ptr(&self.biases_ptr);
        let filter_grad_ptr: *mut f32 = string_to_ptr(&self.filter_gradients_ptr);
        let bias_grad_ptr: *mut f32 = string_to_ptr(&self.bias_gradients_ptr);
        let filter_velocity_ptr: *mut f32 = string_to_ptr(&self.filter_velocity_ptr);
        let bias_velocity_ptr: *mut f32 = string_to_ptr(&self.bias_velocity_ptr);
        let filter_momentum_ptr: *mut f32 = string_to_ptr(&self.filter_momentum_ptr);
        let bias_momentum_ptr: *mut f32 = string_to_ptr(&self.bias_momentum_ptr);

        //scalar_op_3d_inplace(weight_grad_ptr, self.lr, 2, self.in_shape.0, self.in_shape.2, self.out_shape.2);
        //scalar_op_3d_inplace(bias_grad_ptr, self.lr, 2, self.out_shape.0, self.out_shape.1, self.out_shape.2);
        //element_op_3d_inplace(weight_ptr, weight_grad_ptr, 1, self.in_shape.0, self.in_shape.2, self.out_shape.2);
        //element_op_3d_inplace(bias_ptr, bias_grad_ptr, 1, self.out_shape.0, self.out_shape.1, self.out_shape.2);
        //println!("filter: {:?}", cuda_ptr_to_array(filter_ptr, &[self.n_filters, self.in_channels, self.filter_dim, self.filter_dim]));
        //println!("filter_grad: {:?}", cuda_ptr_to_array(filter_grad_ptr, &[self.n_filters, self.in_channels, self.filter_dim, self.filter_dim]));
        gradient_desc_3d(
            lr, l2,
            filter_ptr, filter_grad_ptr, filter_velocity_ptr, filter_momentum_ptr,
             self.n_filters * self.in_channels, self.filter_dim, self.filter_dim,
            bias_ptr, bias_grad_ptr, bias_velocity_ptr, bias_momentum_ptr,
            self.out_shape.0, self.out_shape.1, self.out_shape.2,
            only_bias, false, self.batch_size, optimizer_type, alpha, beta
        );

        //println!("filter after: {:?}", cuda_ptr_to_array(filter_ptr, &[self.n_filters, self.in_channels, self.filter_dim, self.filter_dim]));
        //println!("filter_grad after: {:?}", cuda_ptr_to_array(filter_grad_ptr, &[self.n_filters, self.in_channels, self.filter_dim, self.filter_dim]));

        self.batch_size = 0.0;

        //println!("{:?}", cuda_ptr_to_array(weight_grad_ptr, &[self.in_shape.0, self.in_shape.2, self.out_shape.2]))
        //self.weights -= &(self.lr * (&self.weight_gradients + self.l2 * &self.weights));
        //self.biases -= &(self.lr * &self.bias_gradients);
    }

    pub fn zero_io(&mut self, io_ptr_name: &String)
    {
        if io_ptr_name.contains("input")
        {
            self.zero_input_grad = true;
        }
        else if io_ptr_name.contains("output")
        {
            self.zero_output = true;
        }
    }

    pub fn details(&self)
    {
        println!("Layer type: CONV_2D | Layer name: {:?}", self.name);
        println!("Input shape: {:?}", self.in_shape);
        println!("Output shape: {:?}", self.out_shape);
        println!("Filters: \n{:?}", self.filters);
        println!("Biases: \n{:?}", self.biases);
    }

    pub fn get_param_count(&self) -> usize
    {
        return self.n_filters * self.in_channels * self.filter_dim * self.filter_dim +
               self.out_shape.0 * self.out_shape.1 * self.out_shape.2;
    }

    pub fn move_ptrs_to_arrays(&mut self, bias_only: bool)
    {
        if !bias_only
        {
            self.filters = cuda_ptr_to_array(string_to_ptr(&self.filters_ptr), &[self.n_filters, self.in_channels, self.filter_dim, self.filter_dim]);
        }
        else
        {
            self.filters = ArrayD::zeros(IxDyn(&[0]));
            self.filter_array_allocated = false;
        }

        if self.use_bias
        {
            self.biases = cuda_ptr_to_array(string_to_ptr(&self.biases_ptr), 
                &[self.out_shape.0, self.out_shape.1, self.out_shape.2]);
        }
        else
        {
            self.bias_array_allocated = false;
        }
        //free_cuda_array(string_to_ptr(&self.weight_ptr));
        //free_cuda_array(string_to_ptr(&self.biases_ptr));
        //free_cuda_array(string_to_ptr(&self.input_ptr));
        //free_cuda_array(string_to_ptr(&self.result_ptr));
        //free_cuda_array(string_to_ptr(&self.input_grads_ptr));
        //free_cuda_array(string_to_ptr(&self.output_grads_ptr));
        //free_cuda_array(string_to_ptr(&self.weight_gradients_ptr));
        //free_cuda_array(string_to_ptr(&self.bias_gradients_ptr));

        self.filter_ptr_allocated = false;
        self.bias_ptr_allocated = false;
    }

    pub fn set_ptrs_allocated(&mut self)
    {
        self.filter_ptr_allocated = true;
        self.bias_ptr_allocated = true;
    }
}