
use std::collections::HashMap;

use ndarray::{Array3, ArrayD, IxDyn};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{cuda_bridge::{copy_cuda_to_cuda, element_op_3d_inplace, element_op_3d_ret, gradient_desc_3d_dense, zeroes_3d_inplace}, pointer_ops::{array_to_cuda_ptr_str, cuda_ptr_to_array, new_cuda_ptr_str, string_to_ptr}};

#[derive(Serialize, Deserialize)]
pub struct PosEncoding2DCuda
{
    pub shape: (usize, usize, usize),

    pub io_ptrs: HashMap<String, String>,
    pub pos_encoding_mat: ArrayD<f32>,
    pub pos_encoding_ptr: String,
    pub pos_encoding_grad_ptr: String,
    pub pos_encoding_vel_ptr: String,
    
    pub pos_mat_ptr_allocated: bool, 
    pub pos_mat_initialised: bool,

    pub batch_size: f32,

    pub lr: f32,
    pub l2: f32
}
impl PosEncoding2DCuda
{
    // weight matrix initialize during first ever run
    pub fn new(lr: f32, l2: f32, batch: usize, rows: usize, cols: usize) -> Self
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
            shape: (batch, rows, cols),
            pos_encoding_mat: ArrayD::zeros(IxDyn(&[0])),
            pos_encoding_ptr: "".to_string(),
            pos_encoding_grad_ptr: "".to_string(),
            pos_encoding_vel_ptr: "".to_string(),
            pos_mat_ptr_allocated: false,
            pos_mat_initialised: false,
            batch_size: 0.0,
            lr,
            l2
        }
    }

    // supports batch matrix multiplication unlike cpu
    pub fn forward(&mut self)
    {
        let batch: usize = self.shape.0;
        let rows: usize = self.shape.1;
        let cols: usize = self.shape.2;

        //println!("{:?}, {:?}, {:?}, {:?}", input.get_ptr(), batch, rows, cols);

        if !self.pos_mat_ptr_allocated
        {   
            if !self.pos_mat_initialised
            {   
                let mut pos_encoding_mat: Array3<f32> = Array3::zeros((batch, rows, cols));
                /*
                for i in 0..rows
                {
                    for j in (0..cols).step_by(2)
                    {
                        pos_encoding_mat[(0, i, j)] = ((i as f32) / (10.0_f32.powf((2.0 * j as f32) / (cols as f32)))).sin();
                        if j < cols - 1
                        {
                            pos_encoding_mat[(0, i, j + 1)] = ((i as f32) / (10.0_f32.powf((2.0 * j as f32) / (cols as f32)))).cos();
                        }
                    }
                }
                */
                //let range: f32 = (6.0 / (rows + cols) as f32).sqrt();
                pos_encoding_mat.map_mut(|v: &mut f32| *v = rand::thread_rng().gen_range(-1e-8..1e-8));
                
                //println!("{:?}", pos_encoding_mat);
                self.pos_encoding_mat = pos_encoding_mat.into_dyn();

                self.pos_mat_initialised = true;
            }

            self.pos_encoding_ptr = array_to_cuda_ptr_str(&mut self.pos_encoding_mat);
            self.pos_encoding_grad_ptr = new_cuda_ptr_str(&[batch, rows, cols]);
            self.pos_encoding_vel_ptr = new_cuda_ptr_str(&[batch, rows, cols]);

            // initialise input pointer, set the input as the result pointer from previous layer
            // tensor struct at this stage will contain the result ptr of the previous layer
            ////////////////////////////////////////////////////////////////
            //self.input_ptr = input.get_ptr_as_str();
            //self.input_ptr = new_cuda_ptr_str(&[batch, rows, cols]);
            //////////////////////////////////////////////////////////////////

            // initialise the result tensor/pointer
            //self.result_ptr = new_cuda_ptr_str(&[batch, rows, cols]);
            //self.result_ptr_t = new_cuda_ptr_str(&[batch, self.n_out, rows]);
            
            //self.input_grads_ptr = new_cuda_ptr_str(&[batch, rows, cols]);
            //self.output_grads_ptr = new_cuda_ptr_str(&[batch, rows, cols]);

            //self.shape = (batch, rows, cols);
            //self.out_shape = (batch, rows, cols);

            self.pos_mat_ptr_allocated = true;
        }

        //let broadcast_buf: String = new_cuda_ptr_str(batch * rows * cols * self.n_out);

        // convert strings to pointers
        let input_ptr: *mut f32 = string_to_ptr(self.io_ptrs.get_mut("input").unwrap());
        let pos_encoding_ptr: *mut f32 = string_to_ptr(&self.pos_encoding_ptr);
        let result_ptr: *mut f32 = string_to_ptr(self.io_ptrs.get_mut("output").unwrap());

        // data does not need to be copied as result ptr from previous layer
        // is set as the input

        // copy data to the input pointer, prevent reallocation

        // parallel perform matrix multiplication
        // and sum with bias tensor
        // result pointer updated
        //let start: Instant = Instant::now();
        //if self.use_tiled || !self.use_tiled
        //{
        //println!("{:?}", cuda_ptr_to_array(result_ptr, &[self.in_shape.0, self.in_shape.1, self.in_shape.2]));
        //println!("{:?}", cuda_ptr_to_array(pos_encoding_ptr, &[self.in_shape.0, self.in_shape.1, self.in_shape.2]));
        element_op_3d_ret(
            result_ptr, input_ptr, pos_encoding_ptr, 0, 
            batch, rows, cols
        );

        //println!("{:?}", cuda_ptr_to_array(result_ptr, &[self.in_shape.0, self.in_shape.1, self.in_shape.2]));
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
        //input.set_ptr(result_ptr, vec![batch, rows, cols]);

        //println!("{:?}", cuda_ptr_to_array(input.get_ptr(), input.get_shape()));
        //println!("-----------------------------");
        //exit(1);
        // previous pointer will be recorded in previous layer

    }

    pub fn backward(&mut self)
    {
        let input_grad_ptr: *mut f32 = string_to_ptr(self.io_ptrs.get_mut("input_grad").unwrap());
        let pos_encoding_grad_ptr: *mut f32 = string_to_ptr(&self.pos_encoding_grad_ptr);
        let output_grad_ptr = string_to_ptr(self.io_ptrs.get_mut("output_grad").unwrap());
        //let weight_grad_ptr: *mut f32 = string_to_ptr(&self.weight_gradients_ptr);
        //let bias_grad_ptr: *mut f32 = string_to_ptr(&self.bias_gradients_ptr);

        // positional encoding is C = A + B, derivative is 1, just copy memory
        element_op_3d_inplace(pos_encoding_grad_ptr, output_grad_ptr, 0, self.shape.0, self.shape.1, self.shape.2);
        copy_cuda_to_cuda(input_grad_ptr, output_grad_ptr, &[self.shape.0, self.shape.1, self.shape.2]);

        self.batch_size += 1.0;
        //println!("{:?}", cuda_ptr_to_array(ptr.get_ptr(), ptr.get_shape()));
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

    pub fn update_params(&mut self, optimizer_type: i32, alpha: f32, lower_lr: f32, upper_lr: f32)
    {   
        let pos_encoding_ptr: *mut f32 = string_to_ptr(&self.pos_encoding_ptr);
        let pos_encoding_grad_ptr: *mut f32 = string_to_ptr(&self.pos_encoding_grad_ptr);
        let pos_encoding_vel_ptr: *mut f32 = string_to_ptr(&self.pos_encoding_vel_ptr);

        //scalar_op_3d_inplace(pos_encoding_grad_ptr, self.lr, 2, self.out_shape.0, self.out_shape.1, self.out_shape.2);
        //element_op_3d_inplace(pos_encoding_ptr, pos_encoding_grad_ptr, 1, self.out_shape.0, self.out_shape.1, self.out_shape.2);
        gradient_desc_3d_dense(
            self.lr, self.l2,
            pos_encoding_ptr, pos_encoding_grad_ptr, pos_encoding_vel_ptr,
            self.shape.0, self.shape.1, self.shape.2, 
            pos_encoding_ptr, pos_encoding_grad_ptr, pos_encoding_vel_ptr,
            self.shape.0, self.shape.1, self.shape.2, true, false,
            self.batch_size, optimizer_type, alpha, lower_lr, upper_lr
        );
        self.batch_size = 0.0;
        //println!("{:?}", cuda_ptr_to_array(weight_grad_ptr, &[self.in_shape.0, self.in_shape.2, self.out_shape.2]))
        //self.weights -= &(self.lr * (&self.weight_gradients + self.l2 * &self.weights));
        //self.biases -= &(self.lr * &self.bias_gradients);
    }

    pub fn zero_io(&mut self, ptr_name: &String)
    {
        let io_ptr: *mut f32 = string_to_ptr(self.io_ptrs.get(ptr_name).unwrap());
        zeroes_3d_inplace(io_ptr, self.shape.0, self.shape.1, self.shape.2);
    }

    pub fn details(&self)
    {
        println!("Layer type: POS_ENCODING2D");
        println!("Shape: {:?}", self.shape);
        println!("Pos encoding matrix:");
        println!("{:?}", self.pos_encoding_mat);
    }

    pub fn get_param_count(&self) -> usize
    {
        return self.shape.0 * self.shape.1 * self.shape.2;
    }

    pub fn move_ptrs_to_arrays(&mut self)
    {
        self.pos_encoding_mat = cuda_ptr_to_array(string_to_ptr(&self.pos_encoding_ptr), &[self.shape.0, self.shape.1, self.shape.2]);
        //free_cuda_array(string_to_ptr(&self.pos_encoding_ptr));
        //free_cuda_array(string_to_ptr(&self.pos_encoding_grad_ptr));

        self.pos_mat_ptr_allocated = false;
    }
}