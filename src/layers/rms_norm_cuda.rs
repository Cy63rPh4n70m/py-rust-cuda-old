use std::{process::exit, time::Instant};

use ndarray::{Array1, Array3, ArrayD, ArrayViewD, Axis, IxDyn};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{cuda_bridge::{copy_cuda_to_cuda, copy_host_to_cuda, element_op_3d_inplace, free_cuda_array, gradient_desc_3d_dense, l2norm_backward, l2norm_forward, matmul_add_bias, matmul_add_bias_back, matmul_add_bias_tiled, new_cuda_array, rms_norm_backward, rms_norm_forward, round_3d_inplace, scalar_op_3d_inplace, to_cpu, to_cuda, transpose_2d, zeroes_3d_inplace}, layers::activation, neuralnet::CudaTensorPtr, pointer_ops::{array_to_cuda_ptr, array_to_cuda_ptr_str, cuda_ptr_to_array, cuda_ptr_to_array_f16, new_cuda_ptr_str, ptr_to_string, string_to_ptr}};

#[derive(Serialize, Deserialize)]
pub struct RMSNormCuda
{
    pub in_shape: (usize, usize, usize),
    pub out_shape: (usize, usize, usize),

    pub input_ptr: String,
    pub input_pow2_ptr: String,
    pub norm_ptr: String,
    pub scale_ptr: String,
    pub bias_ptr: String,

    pub scale_grad_ptr: String,
    pub bias_grad_ptr: String,
    pub input_grads_ptr: String,
    pub output_grads_ptr: String,
    
    pub result_ptr: String,
    //pub result_ptr_t: String,

    pub scale_array: ArrayD<f32>,
    pub bias_array: ArrayD<f32>,

    pub backward_passes_count: u128,
    pub count: u128,

    pub ptrs_allocated: bool, 
    pub arrays_allocated: bool,
    pub lr: f32
}
impl RMSNormCuda
{
    // weight matrix initialize during first ever run
    pub fn new(lr: f32) -> Self
    {
        return Self
        {
            input_ptr: "".to_string(),
            input_pow2_ptr: "".to_string(),
            norm_ptr: "".to_string(),
            scale_ptr: "".to_string(),
            bias_ptr: "".to_string(),

            scale_grad_ptr: "".to_string(),
            bias_grad_ptr: "".to_string(),
            input_grads_ptr: String::from("NONE"),
            output_grads_ptr: String::from("NONE"),
            result_ptr: "".to_string(),
            //result_ptr_t: "".to_string(),

            scale_array: ArrayD::zeros(IxDyn(&[0])),
            bias_array: ArrayD::zeros(IxDyn(&[0])),
            in_shape: (0, 0, 0),
            out_shape: (0, 0, 0),
            backward_passes_count: 0,
            count: 0,
            ptrs_allocated: false, 
            arrays_allocated: false,
            lr
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

        if !self.ptrs_allocated
        {   
            // initialise input pointer, set the input as the result pointer from previous layer
            // tensor struct at this stage will contain the result ptr of the previous layer
            ////////////////////////////////////////////////////////////////
            self.input_ptr = input.get_ptr_as_str();
            self.input_pow2_ptr = new_cuda_ptr_str(&[batch, rows, cols]);
            //self.input_t_ptr = new_cuda_ptr_str(&[batch, cols, rows]);
            //////////////////////////////////////////////////////////////////

            // initialise the result tensor/pointer
            self.result_ptr = new_cuda_ptr_str(&[batch, rows, cols]);
            self.norm_ptr = new_cuda_ptr_str(&[batch, rows, 1]);

            if !self.arrays_allocated
            {
                self.scale_array = ArrayD::ones(IxDyn(&[batch, rows, cols]));
                self.scale_array.map_mut(|v: &mut f32| *v = rand::thread_rng().gen_range(-1.0..1.0));

                self.bias_array = ArrayD::zeros(IxDyn(&[batch, rows, cols]));
                self.arrays_allocated = true;
            }
            self.scale_ptr = array_to_cuda_ptr_str(&mut self.scale_array);
            self.bias_ptr = array_to_cuda_ptr_str(&mut self.bias_array);
            self.scale_grad_ptr = new_cuda_ptr_str(&[batch, rows, cols]);
            self.bias_grad_ptr = new_cuda_ptr_str(&[batch, rows, cols]);
            //self.result_ptr_t = new_cuda_ptr_str(&[batch, self.n_out, rows]);
            
            self.input_grads_ptr = new_cuda_ptr_str(&[batch, rows, cols]);
            self.output_grads_ptr = new_cuda_ptr_str(&[batch, rows, cols]);

            self.in_shape = (batch, rows, cols);
            self.out_shape = (batch, rows, cols);

            self.ptrs_allocated = true;
        }

        //let broadcast_buf: String = new_cuda_ptr_str(batch * rows * cols * self.n_out);

        // convert strings to pointers
        let input_ptr: *mut f32 = string_to_ptr(&self.input_ptr);
        let input_pow2_ptr: *mut f32 = string_to_ptr(&self.input_pow2_ptr);
        let norm_ptr: *mut f32 = string_to_ptr(&self.norm_ptr);
        let scale_ptr: *mut f32 = string_to_ptr(&self.scale_ptr);
        let bias_ptr: *mut f32 = string_to_ptr(&self.bias_ptr);
        let result_ptr: *mut f32 = string_to_ptr(&self.result_ptr);

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
        //println!("{:?}", cuda_ptr_to_array(input_ptr, &[batch, rows, cols]));
        rms_norm_forward(
            input_ptr, input_pow2_ptr, 
            batch, rows, cols, 
            norm_ptr, scale_ptr, bias_ptr, result_ptr, true
        );
        //println!("{:?}", cuda_ptr_to_array(norm_ptr, &[batch, rows, 1]));
        //println!("{:?}", cuda_ptr_to_array(result_ptr, &[batch, rows, cols]));
        //exit(1);
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
        input.set_ptr(result_ptr, vec![batch, rows, cols]);

        //println!("{:?}", cuda_ptr_to_array(input.get_ptr(), input.get_shape()));
        //println!("-----------------------------");
        //exit(1);
        // previous pointer will be recorded in previous layer

    }

    pub fn backward(&mut self, ptr: &mut CudaTensorPtr)
    {
        let input_ptr: *mut f32 = string_to_ptr(&self.input_ptr);
        let norm_ptr: *mut f32 = string_to_ptr(&self.norm_ptr);
        let scale_ptr: *mut f32 = string_to_ptr(&self.scale_ptr);
        let scale_grad_ptr: *mut f32 = string_to_ptr(&self.scale_grad_ptr);
        let input_grad_ptr: *mut f32 = string_to_ptr(&self.input_grads_ptr);
        let bias_grad_ptr: *mut f32 = string_to_ptr(&self.bias_grad_ptr);
        let original_grads: *mut f32 = ptr.get_ptr();

        //println!("=========================================================");
        //println!("input_array: {:?}", cuda_ptr_to_array(input_ptr, &[self.in_shape.0, self.in_shape.1, self.in_shape.2]));
        //println!("\nweights: {:?}", cuda_ptr_to_array(weight_ptr, &[self.in_shape.0, self.in_shape.2, self.out_shape.2]));
        //println!("\nmatmul result: {:?}", cuda_ptr_to_array(result_ptr, &[self.out_shape.0, self.out_shape.1, self.out_shape.2]));

        //let start: Instant = Instant::now();
        //println!("\noriginal_grads: {:?}", cuda_ptr_to_array(ptr.get_ptr(), ptr.get_shape()));
        rms_norm_backward(
            original_grads, 
            self.in_shape.0, self.in_shape.1, self.in_shape.2, 
            norm_ptr, input_ptr, input_grad_ptr, scale_ptr,
            scale_grad_ptr, bias_grad_ptr
        );
        //let end = start.elapsed();
        //println!("backward: {:.6}", end.as_secs_f64());

        //println!("\noriginal_grads: {:?}", cuda_ptr_to_array(scale_grad_ptr, ptr.get_shape()));
        //exit(1);
        ptr.set_ptr(input_grad_ptr, vec![self.in_shape.0, self.in_shape.1, self.in_shape.2]);
        //println!("\nchained_gradients: {:?}", cuda_ptr_to_array(ptr.get_ptr(), ptr.get_shape()));
        //exit(1);
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

    pub fn update_params(&self)
    {
        let scale_ptr: *mut f32 = string_to_ptr(&self.scale_ptr);
        let scale_grad_ptr: *mut f32 = string_to_ptr(&self.scale_grad_ptr);
        let bias_ptr: *mut f32 = string_to_ptr(&self.bias_ptr);
        let bias_grad_ptr: *mut f32 = string_to_ptr(&self.bias_grad_ptr);

        gradient_desc_3d_dense(
            self.lr, scale_ptr, scale_grad_ptr, 
            self.in_shape.0, self.in_shape.1, self.in_shape.2, 
            bias_ptr, bias_grad_ptr,
            self.in_shape.0, self.in_shape.1, self.in_shape.2, 
            false
        );
    }

    pub fn details(&self)
    {
        println!("Layer type: RMS norm");
        println!("Input shape: {:?}", self.in_shape);
        println!("Output shape: {:?}", self.out_shape);
    }

    pub fn move_ptrs_to_arrays(&mut self)
    {
        self.scale_array = cuda_ptr_to_array(
            string_to_ptr(&self.scale_ptr), 
            &[self.in_shape.0, self.in_shape.1, self.in_shape.2]
        );
        self.bias_array = cuda_ptr_to_array(
            string_to_ptr(&self.bias_ptr), 
            &[self.in_shape.0, self.in_shape.1, self.in_shape.2]
        );
        free_cuda_array(string_to_ptr(&self.norm_ptr));
        self.ptrs_allocated = false;
    }
}