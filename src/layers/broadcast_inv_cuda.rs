
use ndarray::{ArrayD, IxDyn};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{cuda_bridge::{broadcast_2d_to_3d, copy_cuda_to_cuda, element_op_3d_ret, gradient_desc_3d_dense, sum_axis}, neuralnet::CudaTensorPtr, pointer_ops::{array_to_cuda_ptr_str, cuda_ptr_to_array, new_cuda_ptr_str, string_to_ptr}};

#[derive(Serialize, Deserialize)]
pub struct BroadcastInvMulCuda
{
    pub in_shape: (usize, usize, usize),
    pub weight_shape: (usize, usize, usize),

    pub input: ArrayD<f32>,
    pub weights: ArrayD<f32>,
    pub biases: ArrayD<f32>,

    pub range: f32,

    pub input_ptr: String,
    pub input_t_ptr: String,
    pub weight_ptr: String,
    pub weight_gradients_ptr: String,
    pub weight_gradients_temp_ptr: String,
    pub weight_vel_ptr: String,
    pub input_grads_ptr: String,
    pub weight_intermediate_grads_ptr: String,
    pub output_grads_ptr: String,
    pub biases_ptr: String,
    pub bias_gradients_ptr: String,
    pub bias_vel_ptr: String,
    
    pub result_ptr: String,
    //pub result_ptr_t: String,

    pub lr: f32,
    pub l2: f32,
    pub axis: usize,

    pub batch_size: f32,
    pub count: u128,

    pub weight_ptr_allocated: bool, 
    pub bias_ptr_allocated: bool, 
    pub weight_array_allocated: bool,
    pub bias_array_allocated: bool,
    //pub grads_ptr_allocated: bool,
}
impl BroadcastInvMulCuda
{
    // weight matrix initialize during first ever run
    pub fn new(
        weight_shape: (usize, usize, usize), range: f32, axis: usize, lr: f32, l2: f32
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
            weight_gradients_temp_ptr: "".to_string(),
            weight_vel_ptr: "".to_string(),

            input_grads_ptr: String::from("NONE"),
            weight_intermediate_grads_ptr: String::from("NONE"),
            output_grads_ptr: String::from("NONE"),

            biases,
            biases_ptr: "".to_string(),
            bias_gradients_ptr: "".to_string(),
            bias_vel_ptr: "".to_string(),
            //broadcast_array: ArrayD::zeros(IxDyn(&[0])),
            //broadcast_array_ptr: "".to_string(),
            result_ptr: "".to_string(),
            //result_ptr_t: "".to_string(),

            range,
            in_shape: (0, 0, 0),
            weight_shape,
            axis,
            batch_size: 0.0,
            count: 0,
            lr,
            l2,
            weight_ptr_allocated: false, 
            bias_ptr_allocated: false,
            weight_array_allocated: false,
            bias_array_allocated: false,
            //grads_ptr_allocated: false,
        }
    }

    pub fn set_weights_from_ptr(
        &mut self, 
        tensor_ptr: *mut f32,
        shape: (usize, usize, usize)
    )
    {
        if !self.weight_ptr_allocated
        {
            self.weight_ptr = new_cuda_ptr_str(&[shape.0, shape.1, shape.2]);
            self.weight_gradients_ptr = new_cuda_ptr_str(&[shape.0, shape.1, shape.2]);
            self.weight_vel_ptr = new_cuda_ptr_str(&[shape.0, shape.1, shape.2]);
            self.weight_ptr_allocated = true;
        }

        let weight_ptr: *mut f32 = string_to_ptr(&self.weight_ptr);

        copy_cuda_to_cuda(weight_ptr, tensor_ptr, &[shape.0, shape.1, shape.2]);
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
                self.biases = ArrayD::zeros(IxDyn(&[batch, rows, cols]));
                self.bias_array_allocated = true
            }

            self.biases_ptr = array_to_cuda_ptr_str(&mut self.biases);
            self.bias_gradients_ptr = new_cuda_ptr_str(&[batch, rows, cols]);
            self.bias_vel_ptr = new_cuda_ptr_str(&[batch, rows, cols]);

            // initialise input pointer, set the input as the result pointer from previous layer
            // tensor struct at this stage will contain the result ptr of the previous layer
            ////////////////////////////////////////////////////////////////
            self.input_ptr = input.get_ptr_as_str();
            //self.input_t_ptr = new_cuda_ptr_str(&[batch, cols, rows]);
            //////////////////////////////////////////////////////////////////

            // initialise the result tensor/pointer
            self.result_ptr = new_cuda_ptr_str(&[batch, rows, cols]);
            //self.result_ptr_t = new_cuda_ptr_str(&[batch, self.n_out, rows]);
            
            self.input_grads_ptr = new_cuda_ptr_str(&[batch, rows, cols]);
            self.weight_intermediate_grads_ptr = new_cuda_ptr_str(&[batch, rows, cols]);
            self.output_grads_ptr = new_cuda_ptr_str(&[batch, rows, cols]);

            self.in_shape = (batch, rows, cols);
            //self.out_shape = (batch, rows, self.n_out);

            self.bias_ptr_allocated = true;

            if !self.weight_ptr_allocated
            {
                if !self.weight_array_allocated
                {
                    // initialise weights and weight pointer
                    self.weights = ArrayD::from_shape_fn(
                        IxDyn(&[self.weight_shape.0, self.weight_shape.1, self.weight_shape.2]), 
                        |_| rand::thread_rng().gen_range(-self.range..self.range)
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
                self.weight_gradients_ptr = new_cuda_ptr_str(&[self.weight_shape.0, self.weight_shape.1, self.weight_shape.2]);
                self.weight_gradients_temp_ptr = new_cuda_ptr_str(&[self.weight_shape.0, self.weight_shape.1, self.weight_shape.2]);
                self.weight_vel_ptr = new_cuda_ptr_str(&[self.weight_shape.0, self.weight_shape.1, self.weight_shape.2]);

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
        copy_cuda_to_cuda(result_ptr, input_ptr, &[self.in_shape.0, self.in_shape.1, self.in_shape.2]);
        broadcast_2d_to_3d(
            result_ptr, 
            self.in_shape.0, self.in_shape.1, self.in_shape.2, 
            weight_ptr, self.axis as i32, 2, false
        );

        //println!("{:?}", cuda_ptr_to_array(weight_ptr, &[self.weight_shape.0, self.weight_shape.1, self.weight_shape.2]));
        //exit(1);
        //element_op_3d_inplace(
        //    result_ptr, bias_ptr, 0, 
        //    self.out_shape.0, self.out_shape.1, self.out_shape.2
        //);

        //println!("-----------------------------");
        //println!("{:?}", cuda_ptr_to_array(input_ptr, &[batch, rows, cols]));
        //println!("{:?}", cuda_ptr_to_array(weight_ptr, &[batch, rows, 1]));
        //println!("{:?}", cuda_ptr_to_array(result_ptr, &[batch, rows, cols]));
        //exit(1);
        // overwrite the current pointer with result ptr, to be COPIED to input of next layer
        // current pointer is already recorded by previous layer, don't free
        input.set_ptr(result_ptr, vec![self.in_shape.0, self.in_shape.1, self.in_shape.2]);

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
        let weight_intermediate_grads_ptr: *mut f32 = string_to_ptr(&self.weight_intermediate_grads_ptr);
        let weight_grad_ptr: *mut f32 = string_to_ptr(&self.weight_gradients_ptr);
        let weight_grad_temp_ptr: *mut f32 = string_to_ptr(&self.weight_gradients_temp_ptr);
        let bias_grad_ptr: *mut f32 = string_to_ptr(&self.bias_gradients_ptr);
        let result_ptr: *mut f32 = string_to_ptr(&self.result_ptr);
        let original_grads: *mut f32 = ptr.get_ptr();

        //println!("=========================================================");
        //println!("input_array: {:?}", cuda_ptr_to_array(input_ptr, &[self.in_shape.0, self.in_shape.1, self.in_shape.2]));
        //println!("\nweights: {:?}", cuda_ptr_to_array(weight_ptr, &[self.in_shape.0, self.in_shape.2, self.out_shape.2]));
        //println!("\nmatmul result: {:?}", cuda_ptr_to_array(result_ptr, &[self.out_shape.0, self.out_shape.1, self.out_shape.2]));

        // backpropagate for bias
        //copy_cuda_to_cuda(bias_grad_ptr, original_grads, &[self.out_shape.0, self.out_shape.1, self.out_shape.2]);

        // calculate gradient for input (broadcast multiplying original_grads along axis)
        copy_cuda_to_cuda(input_grad_ptr, original_grads, &[self.in_shape.0, self.in_shape.1, self.in_shape.2]);
        broadcast_2d_to_3d(
            input_grad_ptr, self.in_shape.0, self.in_shape.1, self.in_shape.2, 
            weight_ptr, self.axis as i32, 2, false
        );
        
        // calculate gradient for weight (elementwise multiplication followed by summation along axis)
        element_op_3d_ret(
            weight_intermediate_grads_ptr, 
            original_grads, input_ptr, 2, 
            self.in_shape.0, self.in_shape.1, self.in_shape.2
        );
        
        sum_axis(
            weight_grad_temp_ptr, weight_intermediate_grads_ptr, 
            self.in_shape.0, self.in_shape.1, self.in_shape.2, 
            2, true
        );

        // average the gradients
        //scalar_op_3d_inplace(
        //    weight_grad_ptr, self.in_shape.2 as f32, 3, 
        //    self.in_shape.0, self.in_shape.1, 1
        //);

        self.batch_size += 1.0;
        
        //let end = start.elapsed();
        //println!("backward: {:.6}", end.as_secs_f64());

        //println!("\noriginal_grads: {:?}", cuda_ptr_to_array(ptr.get_ptr(), ptr.get_shape()));
        ptr.set_ptr(input_grad_ptr, vec![self.in_shape.0, self.in_shape.1, self.in_shape.2]);
        //println!("\nchained_gradients: {:?}", cuda_ptr_to_array(ptr.get_ptr(), ptr.get_shape()));
        //println!("\nweight_gradients: {:?}", cuda_ptr_to_array(weight_grad_ptr, &[self.in_shape.0, self.in_shape.1, 1]));
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
            self.lr, self.l2, 
            weight_ptr, weight_grad_ptr, weight_vel_ptr, self.weight_shape.0, self.weight_shape.1, self.weight_shape.2,
            bias_ptr, bias_grad_ptr, bias_vel_ptr, self.in_shape.0, self.in_shape.1, self.in_shape.2,
            false, true, self.batch_size
        );

        self.batch_size = 0.0;

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
        println!("Layer type: BROADCAST_INV");
        println!("Input shape: {:?}", self.in_shape);
        //println!("Output shape: {:?}", self.out_shape);
        //println!("L2: {}", self.l2);
        println!("Weights: \n{:?}", self.weights);
        println!("--------------------------------------");
        println!("Biases: \n{:?}", self.biases);
    }

    pub fn get_param_count(&self) -> usize
    {
        return self.weight_shape.0 * self.weight_shape.1 * self.weight_shape.2;
    }

    pub fn move_ptrs_to_arrays(&mut self, bias_only: bool)
    {
        if !bias_only
        {
            self.weights = cuda_ptr_to_array(string_to_ptr(&self.weight_ptr), &[self.weight_shape.0, self.weight_shape.1, self.weight_shape.2]);
        }
        else
        {
            self.weights = ArrayD::zeros(IxDyn(&[0]));
            self.weight_array_allocated = false;
        }

        self.biases = cuda_ptr_to_array(string_to_ptr(&self.biases_ptr), &[self.in_shape.0, self.in_shape.1, self.in_shape.2]);
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