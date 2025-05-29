use serde::{Deserialize, Serialize};

use crate::{cuda_bridge::transpose_2d, pointer_ops::{increment_counter, init_layer_connections, set_zero_counter, string_to_ptr}};

#[derive(Serialize, Deserialize)]
pub struct BatchTransposeCuda
{
    pub shape: (usize, usize, usize),
    //pub io_ptrs: HashMap<String, String>,

    //pub weights: ArrayD<f32>,

    //pub weight_shift: f32,
    //pub range: f32,

    //pub weight_ptr: String,
    //pub weight_shifted_ptr: String,
    //pub weight_act_ptr: String,
    //pub weight_gradients_ptr: String,
    //pub weight_gradients_temp_ptr: String,
    //pub weight_vel_ptr: String,
    //pub weight_act_grad_ptr: String,
    //pub biases_ptr: String,
    //pub bias_gradients_ptr: String,
    //pub bias_vel_ptr: String,

    //pub dropout_mask_ptr: String,
    //pub rand_state_v_ptr: String,

    //pub dropout_rate: f32,

    //pub lr: f32,
    //pub l2: f32,
    //pub op: i32,

    pub backward_count: String,
    pub backward_count_prev: String,

    pub input_ptr: String,
    pub input_grad_ptr: String,
    pub output_ptr: String,
    pub output_grad_ptr: String,

    pub output_traverse_ptr: String,

    pub batch_size: f32,
    pub count: u128,
    pub zero_output: bool,
    pub zero_weights: bool,
    pub zero_input_grad: bool,
    pub zero_weight_grad: bool,

    //pub weight_ptr_allocated: bool, 
    pub in_out_ptrs_allocated: bool,
    //pub bias_ptr_allocated: bool, 
    //pub weight_array_allocated: bool,
    //pub bias_array_allocated: bool,
    //pub grads_ptr_allocated: bool,
}
impl BatchTransposeCuda
{
    // weight matrix initialize during first ever run
    pub fn new(
        batch: usize, rows: usize, cols: usize
    ) -> Self
    {
        //let weights: ArrayD<f32> = ArrayD::zeros(IxDyn(&[0]));

        return Self
        {
            //io_ptrs,
            //weights,
            //weight_ptr: "".to_string(),
            //weight_shifted_ptr: "".to_string(),
            //weight_act_ptr: "".to_string(),
            //weight_gradients_ptr: "".to_string(),
            //weight_gradients_temp_ptr: "".to_string(),
            //weight_vel_ptr: "".to_string(),
            //weight_act_grad_ptr: "".to_string(),

            //dropout_mask_ptr: "".to_string(),
            //rand_state_v_ptr: "".to_string(),
            //dropout_result_ptr: "".to_string(),

            //dropout_rate,
            //op,

            //biases,
            //biases_ptr: "".to_string(),
            //bias_gradients_ptr: "".to_string(),
            //broadcast_array: ArrayD::zeros(IxDyn(&[0])),
            //broadcast_array_ptr: "".to_string(),

            //activation,
            //result_ptr_t: "".to_string(),
            //range,
            shape: (batch, rows, cols),
            input_ptr: "".to_string(),
            input_grad_ptr: "".to_string(),
            output_ptr: "".to_string(),
            output_grad_ptr: "".to_string(),

            output_traverse_ptr: "".to_string(),

            backward_count: String::from("none"),
            backward_count_prev: String::from("none"),
            
            zero_output: true,
            zero_weights: false,
            zero_input_grad: false,
            zero_weight_grad: false,
            //out_shape,
            batch_size: 0.0,
            count: 0,
            //lr,
            //l2,
            //weight_ptr_allocated: false, 
            in_out_ptrs_allocated: false,
            //bias_ptr_allocated: false,
            //weight_array_allocated: false,
            //bias_array_allocated: false,
            //grads_ptr_allocated: false,
        }
    }

    // supports batch matrix multiplication unlike cpu
    pub fn forward(&mut self, str_ptr_in: String) -> String
    {
        //println!("{:?}", cuda_ptr_to_array(input.get_ptr(), input_shape));

        //println!("{:?}, {:?}, {:?}, {:?}", input.get_ptr(), batch, rows, cols);

        if !self.in_out_ptrs_allocated
        {   
            init_layer_connections(
                &mut self.backward_count, &str_ptr_in, 
                &mut self.backward_count_prev, &mut self.input_ptr, 
                &mut self.input_grad_ptr, &mut self.output_ptr, 
                &mut self.output_grad_ptr, &mut self.output_traverse_ptr, 
                &[self.shape.0, self.shape.2, self.shape.1]
            );
            self.in_out_ptrs_allocated = true;
        }

        //let broadcast_buf: String = new_cuda_ptr_str(batch * rows * cols * self.n_out);

        // convert strings to pointers
        let input_ptr: *mut f32 = string_to_ptr(&self.input_ptr);
        //let weight_ptr: *mut f32 = string_to_ptr(self.io_ptrs.get("weight").unwrap());
        //let mask_ptr: *mut f32 = string_to_ptr(&self.dropout_mask_ptr);
        //let rand_states: *mut c_void = string_to_ptr_void(&self.rand_state_v_ptr);
        let result_ptr: *mut f32 = string_to_ptr(&self.output_ptr);

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
        //copy_cuda_to_cuda(weight_shifted_ptr, weight_ptr, &[self.shape.0, self.shape.1, self.shape.2]);
        //scalar_op_3d_inplace(weight_shifted_ptr, self.weight_shift, 1, self.shape.0, self.shape.1, self.shape.2);
        //activation3d_cuda(
        //    weight_act_ptr, weight_shifted_ptr, 
        //    input_shape[0] as u32, input_shape[1] as u32, input_shape[2] as u32, 
        //    "softplus"
        //);

        transpose_2d(result_ptr, input_ptr, self.shape.0, self.shape.1, self.shape.2);

        set_zero_counter(&self.backward_count);

        //println!("input: {:?}\n", cuda_ptr_to_array(input_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
        //println!("results: {:?}\n", cuda_ptr_to_array(result_ptr, &[self.shape.0, self.shape.2, self.shape.1]));

        return self.output_traverse_ptr.clone();
        //element_op_3d_ret(result_ptr, input_ptr, weight_ptr, 2, self.shape.0, self.shape.1, self.shape.2);

        //element_op_3d_inplace(
        //    result_ptr, bias_ptr, 
        //    0, 
        //    self.shape.0, self.shape.1, self.shape.2
        //);

        //println!("-----------------------------");
        //println!("input: {:?}\n", cuda_ptr_to_array(input_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
        //println!("mask: {:?}\n", cuda_ptr_to_array(mask_ptr, &[batch, rows, cols]));
        //println!("weights: {:?}\n", cuda_ptr_to_array(weight_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
        // overwrite the current pointer with result ptr, to be COPIED to input of next layer
        // current pointer is already recorded by previous layer, don't free
        //input.set_ptr(result_ptr, vec![self.shape.0, self.shape.1, self.shape.2]);

        //println!("results: {:?}\n", cuda_ptr_to_array(result_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
        //exit(1);
        //println!("-----------------------------");
        //exit(1);
        // previous pointer will be recorded in previous layer

    }

    pub fn backward(&mut self)
    {
        //let input_ptr: *mut f32 = string_to_ptr(self.io_ptrs.get("input").unwrap());
        //let input_t_ptr: *mut f32 = string_to_ptr(&self.input_t_ptr);
        //let weight_ptr: *mut f32 = string_to_ptr(self.io_ptrs.get("weight").unwrap());
        //let weight_t_ptr: *mut f32 = string_to_ptr(&self.weight_t_ptr);
        let input_grad_ptr: *mut f32 = string_to_ptr(&self.input_grad_ptr);
        //let weight_grad_ptr: *mut f32 = string_to_ptr(self.io_ptrs.get("weight_grad").unwrap());
        //let weight_grad_temp_ptr: *mut f32 = string_to_ptr(&self.weight_gradients_temp_ptr);
        let original_grads: *mut f32 = string_to_ptr(&self.output_grad_ptr);

        //let mask_ptr: *mut f32 = string_to_ptr(&self.dropout_mask_ptr);

        //println!("=========================================================");
        //println!("input_array: {:?}", cuda_ptr_to_array(input_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
        //println!("\nweights: {:?}", cuda_ptr_to_array(weight_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
        //println!("\nmatmul result: {:?}", cuda_ptr_to_array(result_ptr, &[self.shape.0, self.shape.1, self.shape.2]));

        // calculate gradient for bias
        //copy_cuda_to_cuda(
        //    bias_grad_ptr, original_grads, 
        //    &[self.shape.0, self.shape.1, self.shape.2]
        //);

        /*
        // calculate gradient for weights (broadcast multiplying original_grads along axis)
        element_op_3d_ret(
            weight_grad_temp_ptr, 
            original_grads, input_ptr, 2, 
            self.shape.0, self.shape.1, self.shape.2
        );
        element_op_3d_inplace(weight_grad_ptr, weight_grad_temp_ptr, 0, self.shape.0, self.shape.1, self.shape.2);

        //activation3d_cuda_backward(
        //    weight_grad_ptr, weight_shifted_ptr, weight_act_grad_ptr, 
        //    self.shape.0 as u32, self.shape.1 as u32, self.shape.2 as u32, 
        //    "softplus"
        //);
        
        // calculate gradient for input (elementwise multiplication followed by summation along axis)
        element_op_3d_ret(
            input_grad_ptr, 
            original_grads, weight_ptr, 2, 
            self.shape.0, self.shape.1, self.shape.2
        );
        */

        transpose_2d(input_grad_ptr, original_grads, self.shape.0, self.shape.2, self.shape.1);

        increment_counter(&self.backward_count);
        self.batch_size += 1.0;
        
        //let end = start.elapsed();
        //println!("backward: {:.6}", end.as_secs_f64());

        
        //println!("original_grads: {:?}\n", cuda_ptr_to_array(&[self.shape.0, self.shape.1, self.shape.2]));
        //ptr.set_ptr(input_grad_ptr, vec![self.shape.0, self.shape.1, self.shape.2]);
        //println!("input_gradients: {:?}\n", cuda_ptr_to_array(ptr.get_ptr(), ptr.get_shape()));
        //println!("weight_gradients: {:?}\n", cuda_ptr_to_array(input_grad_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
        //println!("mask: {:?}\n", cuda_ptr_to_array(mask_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
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
        shape.insert(shape.len() - 1, self.shape.1);
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
        /*
        let weight_ptr: *mut f32 = string_to_ptr(self.io_ptrs.get("weight").unwrap());
        let weight_grad_ptr: *mut f32 = string_to_ptr(self.io_ptrs.get("weight_grad").unwrap());
        let weight_vel_ptr: *mut f32 = string_to_ptr(&self.weight_vel_ptr);

        //scalar_op_3d_inplace(weight_grad_ptr, self.lr, 2, self.shape.0, self.shape.2, self.shape.2);
        //scalar_op_3d_inplace(bias_grad_ptr, self.lr, 2, self.shape.0, self.shape.1, self.shape.2);
        //element_op_3d_inplace(weight_ptr, weight_grad_ptr, 1, self.shape.0, self.shape.2, self.shape.2);
        //element_op_3d_inplace(bias_ptr, bias_grad_ptr, 1, self.shape.0, self.shape.1, self.shape.2);
        gradient_desc_3d_dense(
            self.lr, self.l2,
            weight_ptr, weight_grad_ptr, weight_vel_ptr,self.shape.0, self.shape.1, self.shape.2,
            weight_ptr, weight_grad_ptr, weight_vel_ptr, self.shape.0, self.shape.1, self.shape.2,
            true, true, self.batch_size
        );

        //println!("{:?}", cuda_ptr_to_array(weight_grad_ptr, &[self.shape.0, self.shape.2, self.shape.2]))
        //self.weights -= &(self.lr * (&self.weight_gradients + self.l2 * &self.weights));
        //self.biases -= &(self.lr * &self.bias_gradients);
        */
        self.batch_size = 0.0;
    }

    pub fn zero_io(&mut self, io_ptr_name: &String)
    {
        //let io_ptr: *mut f32 = string_to_ptr(self.io_ptrs.get(io_ptr_name).unwrap());

        if io_ptr_name.contains("input")
        {
            //zeroes_3d_inplace(io_ptr, self.in_shape.0, self.in_shape.1, self.in_shape.2);
            self.zero_input_grad = true;
        }
        
        if io_ptr_name.contains("output")
        {
            //zeroes_3d_inplace(io_ptr, self.shape.0, self.shape.1, self.shape.2);
            self.zero_output = true;
        }
    }

    pub fn details(&self)
    {
        println!("Layer type: BATCH_TRANSPOSE");
        println!("Input ptr: {:?} | Input grad ptr: {:?}", self.input_ptr, self.input_grad_ptr);
        println!("Output ptr: {:?} | Output grad ptr: {:?}", self.output_ptr, self.output_grad_ptr);
        println!("Input shape: {:?}", self.shape);
    }

    pub fn get_param_count(&self) -> usize
    {
        return 0_usize;
    }

    pub fn move_ptrs_to_arrays(&mut self)
    {
        //self.weights = cuda_ptr_to_array(
        //    string_to_ptr(self.io_ptrs.get("weight").unwrap()), 
        //    &[self.shape.0, self.shape.1, self.shape.2]
        //);
        //self.biases = cuda_ptr_to_array(string_to_ptr(&self.biases_ptr), &[self.shape.0, self.shape.1, self.shape.2]);
        //free_cuda_array(string_to_ptr(&self.weight_ptr));
        //free_cuda_array(string_to_ptr(&self.biases_ptr));
        //free_cuda_array(string_to_ptr(&self.input_ptr));
        //free_cuda_array(string_to_ptr(&self.result_ptr));
        //free_cuda_array(string_to_ptr(&self.input_grads_ptr));
        //free_cuda_array(string_to_ptr(&self.output_grads_ptr));
        //free_cuda_array(string_to_ptr(&self.weight_gradients_ptr));
        //free_cuda_array(string_to_ptr(&self.bias_gradients_ptr));
        self.in_out_ptrs_allocated = false;
    }

    pub fn set_ptrs_allocated(&mut self)
    {
        self.in_out_ptrs_allocated = true;
    }
}