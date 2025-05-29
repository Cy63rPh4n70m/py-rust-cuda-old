
use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{cuda_bridge::{gradient_desc_3d_dense, layer_norm_backward, layer_norm_forward, zeroes_3d_inplace}, pointer_ops::{array_to_cuda_ptr_str, cuda_ptr_to_array, new_cuda_ptr_str, string_to_ptr}};

#[derive(Serialize, Deserialize)]
pub struct LayerNormCuda
{
    pub io_ptrs: HashMap<String, String>,

    pub original_tensor: ArrayD<f32>,
    pub normalized_tensor: ArrayD<f32>,
    pub output_tensor: ArrayD<f32>,
    pub beta_tensor: ArrayD<f32>,
    pub beta_grads: ArrayD<f32>,
    pub alpha_tensor: ArrayD<f32>,
    pub alpha_grads: ArrayD<f32>,

    pub normalized_ptr: String,
    pub beta_ptr: String,
    pub beta_vel_ptr: String,
    pub beta_grads_ptr: String,
    pub alpha_ptr: String,
    pub alpha_grads_ptr: String,
    pub alpha_vel_ptr: String,
    pub mean_ptr: String,
    //pub mean_temp_ptr: String,
    pub var_ptr: String,

    pub mean_grad_ptr: String,
    pub var_grad_ptr: String,

    pub lr: f32,
    pub l2: f32,
    pub axis: usize,
    pub batch_size: f32,

    pub shape: (usize, usize, usize),
    pub ptrs_allocated: bool,
    pub array_allocated: bool
    //pub temp_shape: Vec<usize>,

    //pub first_pass: usize, second_pass: usize
}
impl LayerNormCuda
{
    pub fn new(batch: usize, rows: usize, cols: usize, lr: f32, l2: f32, axis: usize) -> Self
    {
        let io_ptrs: HashMap<String, String> = HashMap::from([
            ("input".to_string(), "none".to_string()),
            ("input_grad".to_string(), "none".to_string()),
            ("output".to_string(), "none".to_string()),
            ("output_grad".to_string(), "none".to_string())
        ]);

        return Self
        {
            io_ptrs,
            original_tensor: ArrayD::zeros(IxDyn(&[0])),
            //original_ptr: String::from("NONE"),
            normalized_tensor: ArrayD::zeros(IxDyn(&[0])),
            normalized_ptr: String::from("NONE"),
            output_tensor: ArrayD::zeros(IxDyn(&[0])),
            //output_ptr: String::from("NONE"),
            beta_tensor: ArrayD::zeros(IxDyn(&[0])),
            beta_ptr: String::from("NONE"),
            beta_grads: ArrayD::zeros(IxDyn(&[0])),
            beta_grads_ptr: String::from("NONE"),
            beta_vel_ptr: String::from("NONE"),
            alpha_tensor: ArrayD::zeros(IxDyn(&[0])),
            alpha_ptr: String::from("NONE"),
            alpha_grads: ArrayD::zeros(IxDyn(&[0])),
            alpha_grads_ptr: String::from("NONE"),
            alpha_vel_ptr: String::from("NONE"),

            mean_ptr: String::from("NONE"),
            var_ptr: String::from("NONE"),

            //output_grad_ptr: String::from("NONE"),
            //input_grad_ptr: String::from("NONE"),
            mean_grad_ptr: String::from("NONE"),
            var_grad_ptr: String::from("NONE"),

            batch_size: 0.0,
            axis,
            lr,
            l2,
            shape: (batch, rows, cols),

            ptrs_allocated: false,
            array_allocated: false
        }
    }

    pub fn forward(&mut self)
    {
        if !self.ptrs_allocated
        {
            let input_shape: &[usize; 3] = &[self.shape.0, self.shape.1, self.shape.2];
            self.original_tensor = ArrayD::zeros(IxDyn(input_shape));
            self.normalized_tensor = ArrayD::zeros(IxDyn(input_shape));
            self.output_tensor = ArrayD::zeros(IxDyn(input_shape));

            if !self.array_allocated
            {
                self.alpha_tensor = ArrayD::ones(IxDyn(input_shape));
                self.beta_tensor = ArrayD::zeros(IxDyn(input_shape));
                
                // initialize alpha parameters with random values
                let shape_prod: f32 = (self.shape.0 * self.shape.1 * self.shape.2) as f32;

                let range: f32 = (6.0 / shape_prod + shape_prod).sqrt();
                self.alpha_tensor.map_mut(
                    |v: &mut f32|
                    *v = rand::thread_rng().gen_range(-range..range)
                );

                self.array_allocated = true;
            }

            self.alpha_grads = ArrayD::zeros(IxDyn(input_shape));
            self.beta_grads = ArrayD::zeros(IxDyn(input_shape));

            //self.original_ptr = original_ptr_struct.get_ptr_as_str();
            self.normalized_ptr = new_cuda_ptr_str(input_shape);
            //self.output_ptr = new_cuda_ptr_str(input_shape);
            self.alpha_ptr = array_to_cuda_ptr_str(&mut self.alpha_tensor);
            self.beta_ptr = array_to_cuda_ptr_str(&mut self.beta_tensor);
            self.alpha_grads_ptr = new_cuda_ptr_str(input_shape);
            self.beta_grads_ptr = new_cuda_ptr_str(input_shape);
            self.alpha_vel_ptr = new_cuda_ptr_str(input_shape);
            self.beta_vel_ptr = new_cuda_ptr_str(input_shape);

            // shape at selected axis will be zero, normalization is
            // performed along that axis
            let mut mean_var_shape: Vec<usize> = input_shape.to_vec();
            //let mut mean_var_temp_shape: Vec<usize> = input_shape.to_vec();

            //self.first_pass = 64;
            //let n_blocks_on_axis: f32 = (mean_var_shape[self.axis] as f32 / self.first_pass as f32).ceil();
            // round up to nearest power of two 
            //self.second_pass = 2.0_f32.powf((n_blocks_on_axis.log2()).ceil()) as usize;

            //println!("{:?}, {:?}", self.first_pass, self.second_pass);

            mean_var_shape[self.axis] = 1;
            //mean_var_temp_shape[self.axis] = self.second_pass;

            self.mean_ptr = new_cuda_ptr_str(mean_var_shape.as_slice());
            self.var_ptr = new_cuda_ptr_str(mean_var_shape.as_slice());

            //self.mean_temp_ptr = new_cuda_ptr_str(mean_var_temp_shape.as_slice());
            //self.temp_shape = mean_var_temp_shape;

            //self.output_grad_ptr = new_cuda_ptr_str(input_shape);
            self.mean_grad_ptr = new_cuda_ptr_str(input_shape);
            self.var_grad_ptr = new_cuda_ptr_str(input_shape);

            self.ptrs_allocated = true;
        }
        
        let original_ptr: *mut f32 = string_to_ptr(self.io_ptrs.get_mut("input").unwrap());
        let normalized_ptr: *mut f32 = string_to_ptr(&self.normalized_ptr);
        let output_ptr: *mut f32 = string_to_ptr(self.io_ptrs.get_mut("output").unwrap());
        let mean_ptr: *mut f32 = string_to_ptr(&self.mean_ptr);
        //let mean_temp_ptr: *mut f32 = string_to_ptr(&self.mean_temp_ptr);
        let var_ptr: *mut f32 = string_to_ptr(&self.var_ptr);
        let alpha_ptr: *mut f32 = string_to_ptr(&self.alpha_ptr);
        let beta_ptr: *mut f32 = string_to_ptr(&self.beta_ptr);

        //println!("2: {:?}", cuda_ptr_to_array(original_ptr_struct.get_ptr(), original_ptr_struct.get_shape()));
        //println!("{:?}", cuda_ptr_to_array(original_ptr, input_shape));

        // copy input array
        //copy_cuda_to_cuda(
        //    original_ptr, 
        //    original_ptr_struct.get_ptr(), 
        //    original_ptr_struct.get_shape()
        //);

        layer_norm_forward(
            output_ptr, normalized_ptr, original_ptr, 
            mean_ptr, 
            //self.first_pass, self.second_pass,
            //mean_temp_ptr, self.temp_shape[0], self.temp_shape[1], self.temp_shape[2],
            var_ptr, alpha_ptr, beta_ptr, 
            self.shape.0, self.shape.1, self.shape.2, 
            self.axis
        );

        //println!("{:?}", cuda_ptr_to_array(original_ptr, input_shape));
        //let mut mean_var_shape: Vec<usize> = input_shape.to_vec();
        //mean_var_shape[self.axis] = 1;
        //println!("{:?}", self.temp_shape);
        //println!("{:?}", cuda_ptr_to_array(mean_ptr, mean_var_shape.as_slice()));
        //exit(1);
        //println!("{:?}", cuda_ptr_to_array(var_ptr, mean_var_shape.as_slice()));
        //println!("{:?}", cuda_ptr_to_array(normalized_ptr, input_shape));

        //original_ptr_struct.set_ptr(output_ptr, input_shape.to_vec());
        //exit(1);
    }

    pub fn backward(&mut self)
    {
        let input_grads_ptr: *mut f32 = string_to_ptr(self.io_ptrs.get_mut("input_grad").unwrap());
        let output_grads_ptr: *mut f32 = string_to_ptr(self.io_ptrs.get_mut("output_grad").unwrap());
        let mean_grads_ptr: *mut f32 = string_to_ptr(&self.mean_grad_ptr);
        let var_grads_ptr: *mut f32 = string_to_ptr(&self.var_grad_ptr);
        let original_ptr: *mut f32 = string_to_ptr(self.io_ptrs.get_mut("input").unwrap());
        let normalized_ptr: *mut f32 = string_to_ptr(&self.normalized_ptr);
        let mean_ptr: *mut f32 = string_to_ptr(&self.mean_ptr);
        let var_ptr: *mut f32 = string_to_ptr(&self.var_ptr);
        let alpha_ptr: *mut f32 = string_to_ptr(&self.alpha_ptr);
        let beta_ptr: *mut f32 = string_to_ptr(&self.beta_ptr);
        let alpha_grads_ptr: *mut f32 = string_to_ptr(&self.alpha_grads_ptr);
        let beta_grads_ptr: *mut f32 = string_to_ptr(&self.beta_grads_ptr);

        //println!("{:?}", cuda_ptr_to_array(beta_ptr, grad_shape));
        //println!("{:?}", cuda_ptr_to_array(alpha_ptr, grad_shape));
        //exit(1);
        
        //let start: Instant = Instant::now();
        layer_norm_backward(
            input_grads_ptr, 
            output_grads_ptr, 
            original_ptr, normalized_ptr, 
            mean_ptr, var_ptr, mean_grads_ptr, var_grads_ptr, 
            alpha_ptr, beta_ptr, alpha_grads_ptr, beta_grads_ptr, 
            self.shape.0, self.shape.1, self.shape.2, self.axis
        );

        self.batch_size += 1.0;
        //let end = start.elapsed();
        //println!("backward: {:.6}", end.as_secs_f64());

        //println!("return: {:?}", cuda_ptr_to_array(output_grads_ptr, grad_shape));
        //println!("beta: {:?}", cuda_ptr_to_array(beta_grads_ptr, grad_shape));
        //println!("alpha: {:?}", cuda_ptr_to_array(alpha_grads_ptr, grad_shape));
        //grads.set_ptr(output_grads_ptr, grad_shape.to_vec());
        //exit(1);

        /*
        // backpropagate through transform
        // dtransform/dscaled = alpha
        // dtransform/dalpha = scaled
        // dtransform/dbeta = 1.0
        let transformed_r_scaled: ArrayD<f64> = self.alpha_tensor.clone();
        let transformed_r_alpha: ArrayD<f64> = self.scaled_tensor.clone();
        let transformed_r_beta: f64 = 1.0;

        self.alpha_grads += &(&loss_r_transformed * &transformed_r_alpha);
        self.beta_grads += &(&loss_r_transformed * transformed_r_beta);
        let loss_r_scaled: ArrayD<f64> = loss_r_transformed * transformed_r_scaled;

        // backpropagate through scaling
        // dscaled/dinput = 1.0 / var
        let scaled_r_inputs: f64 = 1.0 / (self.var + self.epsilon).sqrt();
        let scaled_r_mean: f64 = -1.0 / (self.var + self.epsilon).sqrt();
        let scaled_r_var: ArrayD<f64> = 
            -((&self.input_tensor - self.mean) / 
            (2.0 * (self.var + self.epsilon).powf(3.0/2.0)));
        
        let mean_r_inputs: f64 = 1.0 / (self.input_tensor.len() as f64);
        let var_r_inputs: ArrayD<f64> = 
            (2.0 * (&self.input_tensor - self.mean)) / 
            (self.input_tensor.len() as f64);

        let var_r_mean_tensor: ArrayD<f64> = 
            -(2.0 * (&self.input_tensor - self.mean)) / 
            (self.input_tensor.len() as f64);

        let var_r_mean: f64 = var_r_mean_tensor.sum();

        let loss_r_input: ArrayD<f64> = 
            (&loss_r_scaled * scaled_r_inputs) +
            ((&loss_r_scaled * scaled_r_mean + &loss_r_scaled * &scaled_r_var * var_r_mean) * mean_r_inputs) +
            (&loss_r_scaled * scaled_r_var * var_r_inputs);
        //println!("{:?}", loss_r_input);
        //std::process::exit(1);

        return loss_r_input;
        */
    }

    pub fn update_params(&mut self, optimizer_type: i32, lr: f32, alpha: f32, lower_lr: f32, upper_lr: f32)
    {
        //self.alpha_tensor -= &(self.lr * &self.alpha_grads);
        //self.beta_tensor -= &(self.lr * &self.beta_grads);

        let alpha_ptr: *mut f32 = string_to_ptr(&self.alpha_ptr);
        let beta_ptr: *mut f32 = string_to_ptr(&self.beta_ptr);
        let alpha_grads_ptr: *mut f32 = string_to_ptr(&self.alpha_grads_ptr);
        let beta_grads_ptr: *mut f32 = string_to_ptr(&self.beta_grads_ptr);
        let alpha_vel_ptr: *mut f32 = string_to_ptr(&self.alpha_vel_ptr);
        let beta_vel_ptr: *mut f32 = string_to_ptr(&self.beta_vel_ptr);

        // normalization layer has two parameters, gradient descent for dense can be reused
        gradient_desc_3d_dense(
            lr, self.l2,
            alpha_ptr, alpha_grads_ptr, alpha_vel_ptr, self.shape.0, self.shape.1, self.shape.2, 
            beta_ptr, beta_grads_ptr, beta_vel_ptr, self.shape.0, self.shape.1, self.shape.2,
            false, false, self.batch_size, optimizer_type, alpha, lower_lr, upper_lr
        );

        self.batch_size = 0.0;
    }

    pub fn zero_io(&mut self, ptr_name: &String)
    {
        let io_ptr: *mut f32 = string_to_ptr(self.io_ptrs.get(ptr_name).unwrap());
        zeroes_3d_inplace(io_ptr, self.shape.0, self.shape.1, self.shape.2);
    }

    pub fn details(&self)
    {
        println!("Layer type: LAYER_NORM");
        println!("Input shape: {:?}", self.shape);
        println!("Alphas: {:?}", self.alpha_tensor);
        println!("--------------------------------------");
        println!("Betas: {:?}", self.beta_tensor);
    }

    pub fn get_param_count(&self) -> usize
    {   
        return (self.shape.0 * self.shape.1 * self.shape.2) * 2;
    }

    pub fn move_ptrs_to_arrays(&mut self)
    {
        self.alpha_tensor = cuda_ptr_to_array(string_to_ptr(&self.alpha_ptr), &[self.shape.0, self.shape.1, self.shape.2]);
        self.beta_tensor = cuda_ptr_to_array(string_to_ptr(&self.beta_ptr), &[self.shape.0, self.shape.1, self.shape.2]);

        /*
        free_cuda_array(string_to_ptr(&self.mean_ptr));
        free_cuda_array(string_to_ptr(&self.mean_grad_ptr));
        free_cuda_array(string_to_ptr(&self.var_ptr));
        free_cuda_array(string_to_ptr(&self.var_grad_ptr));
        free_cuda_array(string_to_ptr(&self.normalized_ptr));
        free_cuda_array(string_to_ptr(&self.alpha_ptr));
        free_cuda_array(string_to_ptr(&self.alpha_grads_ptr));
        free_cuda_array(string_to_ptr(&self.beta_ptr));
        free_cuda_array(string_to_ptr(&self.beta_grads_ptr));
        */

        self.ptrs_allocated = false;
    }

}