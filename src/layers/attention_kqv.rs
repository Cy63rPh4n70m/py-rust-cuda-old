use std::{process::exit, time::Instant};

use ndarray::{Array1, Array3, ArrayD, ArrayViewD, Axis, IxDyn};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{cuda_bridge::{copy_cuda_to_cuda, copy_host_to_cuda, element_op_3d_inplace, free_cuda_array, kqv_backward, kqv_forward, kqv_update_params, matmul_add_bias, matmul_add_bias_back, new_cuda_array, round_3d_inplace, scalar_op_3d_inplace, to_cpu, to_cuda, transpose_2d, zeroes_3d_inplace}, layers::activation, neuralnet::CudaTensorPtr, pointer_ops::{array_to_cuda_ptr, array_to_cuda_ptr_str, cuda_ptr_to_array, new_cuda_ptr_str, ptr_to_string, string_to_ptr}};

#[derive(Serialize, Deserialize)]
pub struct KQVCuda
{
    pub input_ptr: String,
    pub input_grads_ptr: String,

    // key matrix
    pub k_in_shape: (usize, usize, usize),
    pub k_out_shape: (usize, usize, usize),
    pub k_weights: ArrayD<f32>,
    pub k_weight_ptr: String,
    pub k_weight_gradients_ptr: String,
    pub k_biases: ArrayD<f32>,
    pub k_biases_ptr: String,
    pub k_bias_gradients_ptr: String,
    pub k_result_ptr: String,

    // query matrix
    pub q_in_shape: (usize, usize, usize),
    pub q_out_shape: (usize, usize, usize),
    pub q_weights: ArrayD<f32>,
    pub q_weight_ptr: String,
    pub q_weight_gradients_ptr: String,
    pub q_biases: ArrayD<f32>,
    pub q_biases_ptr: String,
    pub q_bias_gradients_ptr: String,
    pub q_result_ptr: String,
    
    // value matrix
    pub v_in_shape: (usize, usize, usize),
    pub v_out_shape: (usize, usize, usize),
    pub v_weights: ArrayD<f32>,
    pub v_weight_ptr: String,
    pub v_weight_gradients_ptr: String,
    pub v_biases: ArrayD<f32>,
    pub v_biases_ptr: String,
    pub v_bias_gradients_ptr: String,
    pub v_result_ptr: String,

    pub n_out: usize,

    pub l2: f32,
    pub lr: f32,

    pub new: bool,
}
impl KQVCuda
{
    // weight matrix initialize during first ever run
    pub fn new(
        n_out: usize,
        l2: f32, lr: f32
    ) -> Self
    {
        return Self
        {
            input_ptr: "".to_string(),
            input_grads_ptr: String::from("NONE"),
            
            k_weights: ArrayD::zeros(IxDyn(&[0])),
            k_weight_ptr: "".to_string(),
            k_weight_gradients_ptr: "".to_string(),
            k_biases: ArrayD::zeros(IxDyn(&[0])),
            k_biases_ptr: "".to_string(),
            k_bias_gradients_ptr: "".to_string(),
            k_result_ptr: "".to_string(),
            k_in_shape: (0, 0, 0),
            k_out_shape: (0, 0, 0),

            q_weights: ArrayD::zeros(IxDyn(&[0])),
            q_weight_ptr: "".to_string(),
            q_weight_gradients_ptr: "".to_string(),
            q_biases: ArrayD::zeros(IxDyn(&[0])),
            q_biases_ptr: "".to_string(),
            q_bias_gradients_ptr: "".to_string(),
            q_result_ptr: "".to_string(),
            q_in_shape: (0, 0, 0),
            q_out_shape: (0, 0, 0),

            v_weights: ArrayD::zeros(IxDyn(&[0])),
            v_weight_ptr: "".to_string(),
            v_weight_gradients_ptr: "".to_string(),
            v_biases: ArrayD::zeros(IxDyn(&[0])),
            v_biases_ptr: "".to_string(),
            v_bias_gradients_ptr: "".to_string(),
            v_result_ptr: "".to_string(),
            v_in_shape: (0, 0, 0),
            v_out_shape: (0, 0, 0),
            l2, lr,
            new: true,

            n_out
        }
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

        if self.new
        {   
            // tensor struct has pointer of previous layer
            self.input_ptr = input.get_ptr_as_str();
            self.input_grads_ptr = new_cuda_ptr_str(&[batch, rows, cols]);

            let range: f32 = (6.0 / (cols + self.n_out) as f32).sqrt();
            self.k_weights = ArrayD::from_shape_fn(
                IxDyn(&[batch, cols, self.n_out]), 
                |_| rand::thread_rng().gen_range(-range..range)
            );
            self.k_weight_ptr = array_to_cuda_ptr_str(&mut self.k_weights);
            self.k_weight_gradients_ptr = new_cuda_ptr_str(&[batch, cols, self.n_out]);
            self.k_biases = ArrayD::zeros(IxDyn(&[batch, rows, self.n_out]));
            self.k_biases_ptr = new_cuda_ptr_str(&[batch, rows, self.n_out]);
            self.k_bias_gradients_ptr = new_cuda_ptr_str(&[batch, rows, self.n_out]);
            self.k_result_ptr = new_cuda_ptr_str(&[batch, rows, self.n_out]);
            self.k_in_shape = (batch, rows, cols);
            self.k_out_shape = (batch, rows, self.n_out);

            self.q_weights = ArrayD::from_shape_fn(
                IxDyn(&[batch, cols, self.n_out]), 
                |_| rand::thread_rng().gen_range(-range..range)
            );
            self.q_weight_ptr = array_to_cuda_ptr_str(&mut self.q_weights);
            self.q_weight_gradients_ptr = new_cuda_ptr_str(&[batch, cols, self.n_out]);
            self.q_biases = ArrayD::zeros(IxDyn(&[batch, rows, self.n_out]));
            self.q_biases_ptr = new_cuda_ptr_str(&[batch, rows, self.n_out]);
            self.q_bias_gradients_ptr = new_cuda_ptr_str(&[batch, rows, self.n_out]);
            self.q_result_ptr = new_cuda_ptr_str(&[batch, rows, self.n_out]);
            self.q_in_shape = (batch, rows, cols);
            self.q_out_shape = (batch, rows, self.n_out);

            self.v_weights = ArrayD::from_shape_fn(
                IxDyn(&[batch, cols, cols]), 
                |_| rand::thread_rng().gen_range(-range..range)
            );
            self.v_weight_ptr = array_to_cuda_ptr_str(&mut self.v_weights);
            self.v_weight_gradients_ptr = new_cuda_ptr_str(&[batch, cols, cols]);
            self.v_biases = ArrayD::zeros(IxDyn(&[batch, rows, cols]));
            self.v_biases_ptr = new_cuda_ptr_str(&[batch, rows, cols]);
            self.v_bias_gradients_ptr = new_cuda_ptr_str(&[batch, rows, cols]);
            self.v_result_ptr = new_cuda_ptr_str(&[batch, rows, cols]);
            self.v_in_shape = (batch, rows, cols);
            self.v_out_shape = (batch, rows, cols);

            self.new = false;

            /*
            // initialise biases and pointers
            self.biases = ArrayD::zeros(IxDyn(&[batch, rows, self.n_out]));
            self.biases_ptr = array_to_cuda_ptr_str(&mut self.biases);
            self.bias_gradients_ptr = new_cuda_ptr_str(&[batch, rows, self.n_out]);
            self.input_ptr = new_cuda_ptr_str(&[batch, rows, cols]);
            self.input_t_ptr = new_cuda_ptr_str(&[batch, cols, rows]);

            // initialise the result tensor/pointer
            self.result_ptr = new_cuda_ptr_str(&[batch, rows, self.n_out]);
            
            self.input_grads_ptr = new_cuda_ptr_str(&[batch, rows, cols]);
            self.output_grads_ptr = new_cuda_ptr_str(&[batch, rows, self.n_out]);

            self.in_shape = (batch, rows, cols);
            self.out_shape = (batch, rows, self.n_out);

            // initialise weights and weight pointer
            let range: f32 = (6.0 / (cols + self.n_out) as f32).sqrt();
            self.weights = ArrayD::from_shape_fn(
                IxDyn(&[batch, cols, self.n_out]), 
                |_| rand::thread_rng().gen_range(-range..range)
            );
            self.weight_ptr = array_to_cuda_ptr_str(&mut self.weights);
            self.weight_gradients_ptr = new_cuda_ptr_str(&[batch, cols, self.n_out]);
            self.weight_t_ptr = new_cuda_ptr_str(&[batch, self.n_out, cols]);
            */
        }

        // convert required strings to pointers
        let input_ptr: *mut f32 = string_to_ptr(&self.input_ptr);
        let k_weight_ptr: *mut f32 = string_to_ptr(&self.k_weight_ptr);
        let k_result_ptr: *mut f32 = string_to_ptr(&self.k_result_ptr);
        let k_bias_ptr: *mut f32 = string_to_ptr(&self.k_biases_ptr);

        let q_weight_ptr: *mut f32 = string_to_ptr(&self.q_weight_ptr);
        let q_result_ptr: *mut f32 = string_to_ptr(&self.q_result_ptr);
        let q_bias_ptr: *mut f32 = string_to_ptr(&self.q_biases_ptr);

        let v_weight_ptr: *mut f32 = string_to_ptr(&self.v_weight_ptr);
        let v_result_ptr: *mut f32 = string_to_ptr(&self.v_result_ptr);
        let v_bias_ptr: *mut f32 = string_to_ptr(&self.v_biases_ptr);

        // result pointer updated
        //let start: Instant = Instant::now();
        kqv_forward(
            input_ptr, self.k_in_shape.0, self.k_in_shape.1, self.k_in_shape.2,
            k_weight_ptr, k_bias_ptr, self.k_in_shape.0, self.k_in_shape.2, self.n_out, 
            q_weight_ptr, q_bias_ptr, 
            v_weight_ptr, v_bias_ptr, 
            k_result_ptr, q_result_ptr, v_result_ptr // three output ptrs
        );

        //let end = start.elapsed();
        //println!("forward: {:.6}", end.as_secs_f64());
        
        //matmul_add_bias(
        //    input_ptr, batch as u32, rows as u32, cols as u32, 
        //    weight_ptr, batch as u32, cols as u32, self.n_out as u32,
        //    result_ptr, bias_ptr//, broadcast_ptr
        //);

        // overwrite the current pointer with result ptr, to be COPIED to input of next layer
        // current pointer is already recorded by previous layer, don't free
        // input.set_ptr(result_ptr, vec![batch, rows, self.n_out]);

        //exit(1);
        // previous pointer will be recorded in previous layer

    }

    pub fn backward(&mut self, k_original_grads: *mut f32, q_original_grads: *mut f32, v_original_grads: *mut f32)
    {
        let input_ptr: *mut f32 = string_to_ptr(&self.input_ptr);
        let input_grad_ptr: *mut f32 = string_to_ptr(&self.input_grads_ptr);

        let k_weight_tensor: *mut f32 = string_to_ptr(&self.k_weight_ptr);
        let k_weight_grads: *mut f32 = string_to_ptr(&self.k_weight_gradients_ptr);
        let k_bias_grads: *mut f32 = string_to_ptr(&self.k_bias_gradients_ptr);

        let q_weight_tensor: *mut f32 = string_to_ptr(&self.q_weight_ptr);
        let q_weight_grads: *mut f32 = string_to_ptr(&self.q_weight_gradients_ptr);
        let q_bias_grads: *mut f32 = string_to_ptr(&self.q_bias_gradients_ptr);

        let v_weight_tensor: *mut f32 = string_to_ptr(&self.v_weight_ptr);
        let v_weight_grads: *mut f32 = string_to_ptr(&self.v_weight_gradients_ptr);
        let v_bias_grads: *mut f32 = string_to_ptr(&self.v_bias_gradients_ptr);
        //let result_ptr: *mut f32 = string_to_ptr(&self.result_ptr);

        //println!("=========================================================");
        //println!("input_array: {:?}", cuda_ptr_to_array(input_ptr, &[self.in_shape.0, self.in_shape.1, self.in_shape.2]));
        //println!("\nweights: {:?}", cuda_ptr_to_array(weight_ptr, &[self.in_shape.0, self.in_shape.2, self.out_shape.2]));
        //println!("\nmatmul result: {:?}", cuda_ptr_to_array(result_ptr, &[self.out_shape.0, self.out_shape.1, self.out_shape.2]));

        //let start: Instant = Instant::now();
        kqv_backward(
            input_ptr, 
            input_grad_ptr, self.k_in_shape.0, self.k_in_shape.1, self.k_in_shape.2, 
            k_weight_grads, self.k_in_shape.0, self.k_in_shape.2, self.n_out, 
            k_bias_grads, self.k_out_shape.0, self.k_out_shape.1, self.k_out_shape.2, 
            k_original_grads, 
            k_weight_tensor, 
            
            q_weight_grads, 
            q_bias_grads, 
            q_original_grads, 
            q_weight_tensor, 
            
            v_weight_grads, 
            v_bias_grads, 
            v_original_grads, 
            v_weight_tensor
        );
        //let end = start.elapsed();
        //println!("backward: {:.6}", end.as_secs_f64());

        //matmul_add_bias_back(
        //    input_grad_ptr, self.in_shape.0 as u32, self.in_shape.1 as u32, self.in_shape.2 as u32, 
        //    weight_grad_ptr, self.in_shape.0 as u32, self.in_shape.2 as u32, self.out_shape.2 as u32, 
        //    bias_grad_ptr, self.out_shape.0 as u32, self.out_shape.1 as u32, self.out_shape.2 as u32,
        //    original_grads, 
        //    input_ptr, input_t_ptr,
        //    weight_ptr, weight_t_ptr,
        //);

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

    pub fn update_params(&mut self)
    {   
        let k_weight: *mut f32 = string_to_ptr(&self.k_weight_ptr);
        let k_biases: *mut f32 = string_to_ptr(&self.k_biases_ptr);
        let k_weight_grads: *mut f32 = string_to_ptr(&self.k_weight_gradients_ptr);
        let k_biases_grads: *mut f32 = string_to_ptr(&self.k_bias_gradients_ptr);

        let q_weight: *mut f32 = string_to_ptr(&self.q_weight_ptr);
        let q_biases: *mut f32 = string_to_ptr(&self.q_biases_ptr);
        let q_weight_grads: *mut f32 = string_to_ptr(&self.q_weight_gradients_ptr);
        let q_biases_grads: *mut f32 = string_to_ptr(&self.q_bias_gradients_ptr);

        let v_weight: *mut f32 = string_to_ptr(&self.v_weight_ptr);
        let v_biases: *mut f32 = string_to_ptr(&self.v_biases_ptr);
        let v_weight_grads: *mut f32 = string_to_ptr(&self.v_weight_gradients_ptr);
        let v_biases_grads: *mut f32 = string_to_ptr(&self.v_bias_gradients_ptr);

        //let start: Instant = Instant::now();
        kqv_update_params(
            self.lr, 
            k_weight, k_weight_grads, self.k_in_shape.0, self.k_in_shape.2, self.n_out, 
            k_biases, k_biases_grads, self.k_out_shape.0, self.k_out_shape.1, self.k_out_shape.2, 
            q_weight, q_weight_grads, 
            q_biases, q_biases_grads, 
            v_weight, v_weight_grads, self.k_in_shape.0, self.k_in_shape.2, self.k_in_shape.2, 
            v_biases, v_biases_grads, self.k_in_shape.0, self.k_in_shape.1, self.k_in_shape.2
        );
        //let end = start.elapsed();
        //println!("update: {:.6}", end.as_secs_f64());
        //gradient_desc_3d(weight_ptr, weight_grad_ptr, self.lr, self.in_shape.0, self.in_shape.2, self.out_shape.2);
        //gradient_desc_3d(bias_ptr, bias_grad_ptr, self.lr, self.out_shape.0, self.out_shape.1, self.out_shape.2);

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
        //println!("Layer type: DENSE");
        //println!("Input shape: {:?}", self.in_shape);
        //println!("Output shape: {:?}", self.out_shape);
        //println!("L2: {}", self.l2);
        //println!("Weights: \n{:?}", self.weights);
        //println!("--------------------------------------");
        //println!("Biases: \n{:?}", self.biases);
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