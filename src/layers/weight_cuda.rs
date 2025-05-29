use std::{process::exit, time::Instant};

use ndarray::{Array1, Array3, ArrayD, ArrayViewD, Axis, IxDyn};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{cuda_bridge::{copy_cuda_to_cuda, copy_host_to_cuda, element_op_3d_inplace, free_cuda_array, gradient_desc_3d_dense, matmul_add_bias, matmul_add_bias_back, matmul_add_bias_tiled, new_cuda_array, round_3d_inplace, scalar_op_3d_inplace, to_cpu, to_cuda, transpose_2d, zeroes_3d_inplace}, layers::activation, neuralnet::CudaTensorPtr, pointer_ops::{array_to_cuda_ptr, array_to_cuda_ptr_str, cuda_ptr_to_array, cuda_ptr_to_array_f16, new_cuda_ptr_str, ptr_to_string, string_to_ptr}};

#[derive(Serialize, Deserialize)]
pub struct WeightCuda
{
    pub in_shape: (usize, usize, usize),
    pub out_shape: (usize, usize, usize),
    pub n_out: usize,

    pub input: ArrayD<f32>,
    pub weights: ArrayD<f32>,
    pub biases: ArrayD<f32>,

    //pub input_ptr: String,
    //pub input_t_ptr: String,
    pub weight_ptr: String,
    pub weight_gradients_ptr: String,
    pub weight_vel_ptr: String,
    //pub input_grads_ptr: String,
    //pub output_grads_ptr: String,
    //pub biases_ptr: String,
    //pub bias_gradients_ptr: String,
    //pub bias_vel_ptr: String,
    
    pub result_ptr: String,
    //pub result_ptr_t: String,

    pub l2: f32,
    pub lr: f32,

    pub backward_passes_count: u128,
    pub count: u128,

    pub weight_ptr_allocated: bool, 
    pub bias_ptr_allocated: bool, 
    pub weight_array_allocated: bool,
    pub bias_array_allocated: bool,
    //pub grads_ptr_allocated: bool,
    pub use_tiled: bool
}
impl DenseCuda
{
    // weight matrix initialize during first ever run
    pub fn new(
        n_out: usize,
        l2: f32, lr: f32, use_tiled: bool
    ) -> Self
    {
        let weights: ArrayD<f32> = ArrayD::zeros(IxDyn(&[0]));
        let biases: ArrayD<f32> = ArrayD::zeros(IxDyn(&[0]));

        return Self
        {
            input: ArrayD::zeros(IxDyn(&[0])),
            input_ptr: "".to_string(),
            input_t_ptr: "".to_string(),
            weights,
            weight_ptr: "".to_string(),
            weight_gradients_ptr: "".to_string(),
            weight_vel_ptr: "".to_string(),

            input_grads_ptr: String::from("NONE"),
            output_grads_ptr: String::from("NONE"),

            biases,
            biases_ptr: "".to_string(),
            bias_gradients_ptr: "".to_string(),
            bias_vel_ptr: "".to_string(),
            //broadcast_array: ArrayD::zeros(IxDyn(&[0])),
            //broadcast_array_ptr: "".to_string(),
            result_ptr: "".to_string(),
            //result_ptr_t: "".to_string(),
            in_shape: (0, 0, 0),
            out_shape: (0, 0, 0),
            n_out,
            l2,
            backward_passes_count: 0,
            count: 0,
            lr,
            weight_ptr_allocated: false, 
            bias_ptr_allocated: false,
            weight_array_allocated: false,
            bias_array_allocated: false,
            //grads_ptr_allocated: false,
            use_tiled
        }
    }

    // assumes 3 dimensional weights
    pub fn set_weights_from_ptr(
        &mut self, 
        tensor_ptr: *mut f32, 
        shape: (usize, usize, usize), 
        transpose_before_copy: bool)
    {   
        if !self.weight_ptr_allocated
        {
            if transpose_before_copy
            {
                self.weight_ptr = new_cuda_ptr_str(&[shape.0, shape.2, shape.1]);
                self.weight_gradients_ptr = new_cuda_ptr_str(&[shape.0, shape.2, shape.1]);
                self.weight_vel_ptr = new_cuda_ptr_str(&[shape.0, shape.2, shape.1]);
            }
            else
            {
                self.weight_ptr = new_cuda_ptr_str(&[shape.0, shape.1, shape.2]);
                self.weight_gradients_ptr = new_cuda_ptr_str(&[shape.0, shape.1, shape.2]);
                self.weight_vel_ptr = new_cuda_ptr_str(&[shape.0, shape.1, shape.2]);
            }
            self.weight_ptr_allocated = true;
            //self.weight_array_allocated = true;
        }

        let weight_ptr: *mut f32 = string_to_ptr(&self.weight_ptr);

        if transpose_before_copy
        {
            transpose_2d(weight_ptr, tensor_ptr, shape.0, shape.1, shape.2);
        }
        else
        {
            copy_cuda_to_cuda(weight_ptr, tensor_ptr, &[shape.0, shape.1, shape.2]);
        }
    }

    pub fn get_weight_grads_as_ptr(&mut self) -> *mut f32
    {
        return string_to_ptr(&self.weight_gradients_ptr);
    }

    // supports batch matrix multiplication unlike cpu
    pub fn forward(&mut self, input: &mut CudaTensorPtr)
    {
        let input_shape: &[usize] = input.get_shape();
        //println!("{:?}", cuda_ptr_to_array(input.get_ptr(), input_shape));
        let batch: usize = input_shape[0];
        let rows: usize = input_shape[1];
        let cols: usize = input_shape[2];

        //println!("{:?}, {:?}, {:?}, {:?}", input.get_ptr(), batch, rows, cols);

        if !self.bias_ptr_allocated
        {   
            // initialise biases and pointers
            if !self.bias_array_allocated
            {
                self.biases = ArrayD::zeros(IxDyn(&[batch, rows, self.n_out]));
                self.bias_array_allocated = true
            }

            self.biases_ptr = array_to_cuda_ptr_str(&mut self.biases);
            self.bias_gradients_ptr = new_cuda_ptr_str(&[batch, rows, self.n_out]);
            self.bias_vel_ptr = new_cuda_ptr_str(&[batch, rows, self.n_out]);

            // initialise input pointer, set the input as the result pointer from previous layer
            // tensor struct at this stage will contain the result ptr of the previous layer
            ////////////////////////////////////////////////////////////////
            self.input_ptr = input.get_ptr_as_str();
            //self.input_t_ptr = new_cuda_ptr_str(&[batch, cols, rows]);
            //////////////////////////////////////////////////////////////////

            // initialise the result tensor/pointer
            self.result_ptr = new_cuda_ptr_str(&[batch, rows, self.n_out]);
            //self.result_ptr_t = new_cuda_ptr_str(&[batch, self.n_out, rows]);
            
            self.input_grads_ptr = new_cuda_ptr_str(&[batch, rows, cols]);
            self.output_grads_ptr = new_cuda_ptr_str(&[batch, rows, self.n_out]);

            self.in_shape = (batch, rows, cols);
            self.out_shape = (batch, rows, self.n_out);

            self.bias_ptr_allocated = true;

            if !self.weight_ptr_allocated
            {
                if !self.weight_array_allocated
                {
                    // initialise weights and weight pointer
                    let range: f32 = (6.0 / (cols + self.n_out) as f32).sqrt();
                    self.weights = ArrayD::from_shape_fn(
                        IxDyn(&[batch, cols, self.n_out]), 
                        |_| rand::thread_rng().gen_range(-range..range)
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
                self.weight_gradients_ptr = new_cuda_ptr_str(&[batch, cols, self.n_out]);
                self.weight_vel_ptr = new_cuda_ptr_str(&[batch, cols, self.n_out]);
                //self.weight_t_ptr = new_cuda_ptr_str(&[batch, self.n_out, cols]);

                self.weight_ptr_allocated = true;
            }
        }

        //let broadcast_buf: String = new_cuda_ptr_str(batch * rows * cols * self.n_out);

        // convert strings to pointers
        let input_ptr: *mut f32 = string_to_ptr(&self.input_ptr);
        let weight_ptr: *mut f32 = string_to_ptr(&self.weight_ptr);
        let result_ptr: *mut f32 = string_to_ptr(&self.result_ptr);
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
        matmul_add_bias_tiled(
            input_ptr, batch as u32, rows as u32, cols as u32, 
            weight_ptr, batch as u32, cols as u32, self.n_out as u32,
            result_ptr, bias_ptr
        );
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
        input.set_ptr(result_ptr, vec![batch, rows, self.n_out]);

        //println!("{:?}", cuda_ptr_to_array(input.get_ptr(), input.get_shape()));
        //println!("-----------------------------");
        //exit(1);
        // previous pointer will be recorded in previous layer

    }

    pub fn backward(&mut self, ptr: &mut CudaTensorPtr)
    {
        let input_ptr: *mut f32 = string_to_ptr(&self.input_ptr);
        //let input_t_ptr: *mut f32 = string_to_ptr(&self.input_t_ptr);
        let weight_ptr: *mut f32 = string_to_ptr(&self.weight_ptr);
        //let weight_t_ptr: *mut f32 = string_to_ptr(&self.weight_t_ptr);
        let input_grad_ptr: *mut f32 = string_to_ptr(&self.input_grads_ptr);
        let weight_grad_ptr: *mut f32 = string_to_ptr(&self.weight_gradients_ptr);
        let bias_grad_ptr: *mut f32 = string_to_ptr(&self.bias_gradients_ptr);
        let result_ptr: *mut f32 = string_to_ptr(&self.result_ptr);
        let original_grads: *mut f32 = ptr.get_ptr();

        //println!("=========================================================");
        //println!("input_array: {:?}", cuda_ptr_to_array(input_ptr, &[self.in_shape.0, self.in_shape.1, self.in_shape.2]));
        //println!("\nweights: {:?}", cuda_ptr_to_array(weight_ptr, &[self.in_shape.0, self.in_shape.2, self.out_shape.2]));
        //println!("\nmatmul result: {:?}", cuda_ptr_to_array(result_ptr, &[self.out_shape.0, self.out_shape.1, self.out_shape.2]));

        //let start: Instant = Instant::now();
        matmul_add_bias_back(
            input_grad_ptr, self.in_shape.0 as u32, self.in_shape.1 as u32, self.in_shape.2 as u32, 
            weight_grad_ptr, self.in_shape.0 as u32, self.in_shape.2 as u32, self.out_shape.2 as u32, 
            bias_grad_ptr, self.out_shape.0 as u32, self.out_shape.1 as u32, self.out_shape.2 as u32,
            original_grads, 
            input_ptr,
            weight_ptr,
        );
        //let end = start.elapsed();
        //println!("backward: {:.6}", end.as_secs_f64());

        //println!("\noriginal_grads: {:?}", cuda_ptr_to_array(ptr.get_ptr(), ptr.get_shape()));
        ptr.set_ptr(input_grad_ptr, vec![self.in_shape.0, self.in_shape.1, self.in_shape.2]);
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
        return self.weight_gradients_ptr.clone();
    }

    pub fn update_params(&mut self, only_bias: bool)
    {   
        let weight_ptr: *mut f32 = string_to_ptr(&self.weight_ptr);
        let bias_ptr: *mut f32 = string_to_ptr(&self.biases_ptr);
        let weight_grad_ptr: *mut f32 = string_to_ptr(&self.weight_gradients_ptr);
        let bias_grad_ptr: *mut f32 = string_to_ptr(&self.bias_gradients_ptr);
        let weight_vel_ptr: *mut f32 = string_to_ptr(&self.weight_vel_ptr);
        let bias_vel_ptr: *mut f32 = string_to_ptr(&self.bias_vel_ptr);

        //scalar_op_3d_inplace(weight_grad_ptr, self.lr, 2, self.in_shape.0, self.in_shape.2, self.out_shape.2);
        //scalar_op_3d_inplace(bias_grad_ptr, self.lr, 2, self.out_shape.0, self.out_shape.1, self.out_shape.2);
        //element_op_3d_inplace(weight_ptr, weight_grad_ptr, 1, self.in_shape.0, self.in_shape.2, self.out_shape.2);
        //element_op_3d_inplace(bias_ptr, bias_grad_ptr, 1, self.out_shape.0, self.out_shape.1, self.out_shape.2);
        gradient_desc_3d_dense(
            self.lr, 
            weight_ptr, weight_grad_ptr, weight_vel_ptr, self.in_shape.0, self.in_shape.2, self.out_shape.2,
            bias_ptr, bias_grad_ptr, bias_vel_ptr, self.out_shape.0, self.out_shape.1, self.out_shape.2,
            only_bias
        );

        //println!("{:?}", cuda_ptr_to_array(weight_grad_ptr, &[self.in_shape.0, self.in_shape.2, self.out_shape.2]))
        //self.weights -= &(self.lr * (&self.weight_gradients + self.l2 * &self.weights));
        //self.biases -= &(self.lr * &self.bias_gradients);
    }

    pub fn zero_grads(&mut self)
    {
        //let weight_grad_ptr: *mut f32 = string_to_ptr(&self.weight_gradients_ptr);
        //let bias_grad_ptr: *mut f32 = string_to_ptr(&self.bias_gradients_ptr);
        //let input_grad_ptr: *mut f32 = string_to_ptr(&self.input_grads_ptr);

        //unsafe
        //{
            // scalar multiply 0.0
            //println!("{:?}", cuda_ptr_to_array(weight_grad_ptr, &[self.in_shape.0, self.in_shape.2, self.out_shape.1]));
            //zeroes_3d_inplace(weight_grad_ptr, self.in_shape.0, self.in_shape.2, self.out_shape.2);
            //zeroes_3d_inplace(bias_grad_ptr, self.out_shape.0, self.out_shape.1, self.out_shape.2);
            //zeroes_3d_inplace(input_grad_ptr, self.in_shape.0, self.in_shape.1, self.in_shape.2);
            //println!("{:?}", cuda_ptr_to_array(weight_grad_ptr, &[self.in_shape.0, self.in_shape.2, self.out_shape.1]));
            //exit(1);
        //}        
    }

    pub fn details(&self)
    {
        println!("Layer type: DENSE");
        println!("Input shape: {:?}", self.in_shape);
        println!("Output shape: {:?}", self.out_shape);
        println!("L2: {}", self.l2);
        println!("Weights: \n{:?}", self.weights);
        println!("--------------------------------------");
        println!("Biases: \n{:?}", self.biases);
    }

    pub fn move_ptrs_to_arrays(&mut self, bias_only: bool)
    {
        if !bias_only
        {
            self.weights = cuda_ptr_to_array(string_to_ptr(&self.weight_ptr), &[self.in_shape.0, self.in_shape.2, self.out_shape.2]);
        }
        else
        {
            self.weights = ArrayD::zeros(IxDyn(&[0]));
            self.weight_array_allocated = false;
        }

        self.biases = cuda_ptr_to_array(string_to_ptr(&self.biases_ptr), &[self.out_shape.0, self.out_shape.1, self.out_shape.2]);
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
    }
}

 /*   
        self.weight_gradients.map_mut(
            |v: &mut f32| 
            if *v > 0.0
            {
                *v = (*v).min(1.0);
            }
            else if *v < 0.0
            {
                *v = (*v).max(-1.0);
            }
            );
        */

        /*
        // add squared weight_grads
        let weights_powered: Array2<f32> = self.weight_gradients.mapv(|v: f32| v.powf(2.0));
        let bias_powered: Array1<f32> = self.bias_gradients.mapv(|v: f32| v.powf(2.0));
        //self.weight_grads_mov_ave += &weights_powered;
        //self.bias_grads_mov_ave += &bias_powered;

        //self.weight_grads_sqr_sum.push_back(weights_powered);
        //self.bias_grads_sqr_sum.push_back(bias_powered);

        //self.count += 1;

        //if self.weight_grads_sqr_sum.len() > 3000
        //{
        //    let oldest_weights: Array2<f32> = self.weight_grads_sqr_sum.pop_front().unwrap();
        //    let oldest_biases: Array1<f32> = self.bias_grads_sqr_sum.pop_front().unwrap();
        //    self.weight_grads_mov_ave -= &oldest_weights;
        //    self.bias_grads_mov_ave -= &oldest_biases;
        //}
        if self.backward_passes_count == 0
        {
            self.weight_grads_mov_ave += &weights_powered;
            self.bias_grads_mov_ave += &bias_powered;
            self.backward_passes_count += 1;
        }
        else
        {
            //self.weight_m = self.b1 * &self.weight_m + (1.0 - self.b1) * &self.weight_gradients;
            self.weight_grads_mov_ave = self.b * &self.weight_grads_mov_ave + (1.0 - self.b) * &weights_powered;
            //self.bias_m = self.b1 * &self.bias_m + (1.0 - self.b1) * &self.bias_gradients;
            self.bias_grads_mov_ave = self.b * &self.bias_grads_mov_ave + (1.0 - self.b) * &bias_powered;
        }
        
        //let weight_m1: Array2<f32> = &self.weight_m / (1.0 - self.b1.powf(self.backward_passes_count));
        //let weight_v1: Array2<f32> = &self.weight_v / (1.0 - self.b2.powf(self.backward_passes_count));
        //let bias_m1: Array1<f32> = &self.bias_m / (1.0 - self.b1.powf(self.backward_passes_count));
        //let bias_v1: Array1<f32> = &self.bias_v / (1.0 - self.b2.powf(self.backward_passes_count));
        
        //let denominator: Array2<f32> = self.weight_grads_mov_ave.mapv(|v: f32| (v + 1e-8).sqrt());
        let weight_adaptive_lrs: Array2<f32> = self.weight_grads_mov_ave.mapv(
            |v: f32| 
            if lr / (v + 1e-8).sqrt() > max_lr { max_lr } 
            else if lr / (v + 1e-8).sqrt() < min_lr { min_lr } 
            else { lr / (v + 1e-8).sqrt() }
        );

        //let denominator: Array1<f32> = self.bias_grads_mov_ave.mapv(|v: f32| (v + 1e-8).sqrt());
        let bias_adaptive_lrs: Array1<f32> = self.bias_grads_mov_ave.mapv(
            |v: f32| 
            if lr / (v + 1e-8).sqrt() > max_lr { max_lr } 
            else if lr / (v + 1e-8).sqrt() < min_lr { min_lr } 
            else { lr / (v + 1e-8).sqrt() }
        );
        */

        //self.weights2d -= &(lr * &self.weight_gradients);
        //self.biases1d -= &(lr * &self.bias_gradients);

        /**/
        /*
        // calculate summed respect to bias
        // calculate bias gradients
        self.bias_gradients += &(1.0 * &loss_r_summed); // bias derivative is 1.0

        // from each output calculate summed respect to weights and
        // summed respect to inputs (activated from next layer in reverse)
        
        //// backpropagating for weights
        // transpose input vector

        let input2d = self.input.clone().into_dimensionality::<Ix2>().unwrap();
        let grads2d = loss_r_summed.into_dimensionality::<Ix2>().unwrap();
        let weights2d = self.weights.clone().into_dimensionality::<Ix2>().unwrap();
        let transposed_input: ArrayView2<f32> = input2d.t();

        // derivative of y = w * inputs respect to w is inputs 
        self.weight_gradients += &transposed_input.dot(&grads2d);

        //// backpropagating for inputs
        // derivative of y = w * inputs respect to inputs is w
        let loss_r_next_layer: Array2<f32> = grads2d.dot(&weights2d.t());
        
        //println!("{:?}", loss_r_summed);
        //println!("{:?}", self.weights2d);
        //println!("{:?}", loss_r_next_layer);
        //std::process::exit(1);
        */