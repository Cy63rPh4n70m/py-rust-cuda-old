
use crate::{cuda_bridge::{copy_cuda_to_cuda, element_op_3d_inplace, scalar_op_3d_inplace, softmax_forward}, pointer_ops::{counter_is_zero, increment_counter, init_layer_connections, new_cuda_ptr_str, set_zero_counter, string_to_ptr}};

use super::layer_cuda::{AllocationStatus, IOPtrs};

pub struct SoftmaxCuda
{
    pub io_ptrs: IOPtrs,
    pub allocation_status: AllocationStatus,

    pub input_exp_ptr: *mut f32,
    pub exp_sum_ptr: *mut f32,
    pub broadcast_temp_ptr: *mut f32,
    
    pub temperature: f32,
    pub input_grad_temp: *mut f32,

    pub backward_passes_count: u128,
    pub count: u128,
    pub batch_size: f32,
}
impl SoftmaxCuda
{
    // weight matrix initialize during first ever run
    pub fn new(batch: usize, rows: usize, cols: usize, temperature: f32) -> Self
    {
        return Self
        {   
            io_ptrs: IOPtrs::new((batch, rows, cols), (batch, rows, cols)),
            allocation_status: AllocationStatus::new(),
            
            input_exp_ptr: std::ptr::null_mut(),
            exp_sum_ptr: std::ptr::null_mut(),
            input_grad_temp: std::ptr::null_mut(),
            broadcast_temp_ptr: std::ptr::null_mut(),

            temperature,
            backward_passes_count: 0,
            count: 0,
            batch_size: 0.0
        }
    }

    // supports batch matrix multiplication unlike cpu
    pub fn forward(&mut self, str_ptr_in: String) -> String
    {
        ////println!("{:?}", cuda_ptr_to_array(input.get_ptr(), input_shape));
        let batch: usize = self.shape.0;
        let rows: usize = self.shape.1;
        let cols: usize = self.shape.2;

        ////println!("{:?}, {:?}, {:?}, {:?}", input.get_ptr(), batch, rows, cols);

        if !self.ptrs_allocated
        {   
            // initialise input pointer, set the input as the result pointer from previous layer
            // tensor struct at this stage will contain the result ptr of the previous layer
            ////////////////////////////////////////////////////////////////
            //self.input_ptr = input.get_ptr_as_str();
            self.input_exp_ptr = new_cuda_ptr_str(&[batch, rows, cols]);
            //self.input_t_ptr = new_cuda_ptr_str(&[batch, cols, rows]);
            //////////////////////////////////////////////////////////////////

            // initialise the result tensor/pointer
            //self.result_ptr = new_cuda_ptr_str(&[batch, rows, cols]);
            self.exp_sum_ptr = new_cuda_ptr_str(&[batch, rows, 1]);
            self.broadcast_temp_ptr = new_cuda_ptr_str(&[batch, rows, cols]);
            self.input_grad_temp = new_cuda_ptr_str(&[self.shape.0, self.shape.1, self.shape.2]);

            //self.result_ptr_t = new_cuda_ptr_str(&[batch, self.n_out, rows]);
            
            //self.input_grads_ptr = new_cuda_ptr_str(&[batch, rows, cols]);
            //self.output_grads_ptr = new_cuda_ptr_str(&[batch, rows, cols]);

            //self.shape = (batch, rows, cols);
            //self.out_shape = (batch, rows, cols);

            init_layer_connections(
                &mut self.backward_count, &str_ptr_in, 
                &mut self.backward_count_prev, &mut self.input_ptr, 
                &mut self.input_grad_ptr, &mut self.output_ptr, 
                &mut self.output_grad_ptr, &mut self.output_traverse_ptr, 
                &[self.shape.0, self.shape.1, self.shape.2]
            );

            self.ptrs_allocated = true;
        }

        //let broadcast_buf: String = new_cuda_ptr_str(batch * rows * cols * self.n_out);

        // convert strings to pointers
        let input_ptr: *mut f32 = string_to_ptr(&self.input_ptr);
        let input_exp_ptr: *mut f32 = string_to_ptr(&self.input_exp_ptr);
        let exp_sum_ptr: *mut f32 = string_to_ptr(&self.exp_sum_ptr);
        let broadcast_temp_ptr: *mut f32 = string_to_ptr(&self.broadcast_temp_ptr);
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
        //{
        ////println!("{:?}", cuda_ptr_to_array(input_ptr, &[batch, rows, cols]));
        softmax_forward(
            input_ptr, input_exp_ptr, exp_sum_ptr,
            result_ptr, broadcast_temp_ptr, self.temperature,
            batch, rows, cols, self.zero_output
        );

        set_zero_counter(&self.backward_count);

        //println!("{:?}", cuda_ptr_to_array(input_ptr, &[batch, rows, cols]));
        //println!("{:?}\n", cuda_ptr_to_array(result_ptr, &[batch, rows, cols]));

        return self.output_traverse_ptr.clone();
        ////println!("{:?}", cuda_ptr_to_array(input_exp_ptr, &[batch, rows, cols]));
        ////println!("{:?}", cuda_ptr_to_array(exp_sum_ptr, &[batch, rows, 1]));
        ////println!("{:?}", cuda_ptr_to_array(result_ptr, &[batch, rows, cols]));
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
        ////println!("dense: {:.6}", end.as_secs_f64());

        ////println!("-----------------------------");
        ////println!("{:?}", cuda_ptr_to_array(input_ptr, &[batch, rows, cols]));
        ////println!("{:?}", cuda_ptr_to_array(weight_ptr, &[batch, cols, self.n_out]));
        // overwrite the current pointer with result ptr, to be COPIED to input of next layer
        // current pointer is already recorded by previous layer, don't free
        //input.set_ptr(result_ptr, vec![batch, rows, cols]);

        ////println!("{:?}", cuda_ptr_to_array(input.get_ptr(), input.get_shape()));
        ////println!("-----------------------------");
        //exit(1);
        // previous pointer will be recorded in previous layer
    }

    pub fn get_param_count(&self) -> usize
    {
        return 0_usize;
    }

    pub fn backward(&mut self)
    {
        //if self.input_grad_temp.contains("none")
        //{
        //    self.input_grad_temp = new_cuda_ptr_str(&[self.shape.0, self.shape.1, self.shape.2]);
        //}
        //let input_exp_ptr: *mut f32 = string_to_ptr(&self.input_exp_ptr);
        //let exp_sum_ptr: *mut f32 = string_to_ptr(&self.exp_sum_ptr);
        let input_grad_ptr: *mut f32 = string_to_ptr(&self.input_grad_ptr);
        let input_grad_temp: *mut f32 = string_to_ptr(&self.input_grad_temp);
        let original_grads: *mut f32 = string_to_ptr(&self.output_grad_ptr);

        copy_cuda_to_cuda(input_grad_temp, original_grads, &[self.shape.0, self.shape.1, self.shape.2]);
        // for temperature
        scalar_op_3d_inplace(
            input_grad_temp, self.temperature, 3, 
            self.shape.0, self.shape.1, self.shape.2
        );

        if counter_is_zero(&self.backward_count_prev)
        {
            copy_cuda_to_cuda(input_grad_ptr, input_grad_temp, &[self.shape.0, self.shape.1, self.shape.2]);
        }
        else
        {
            element_op_3d_inplace(input_grad_ptr, input_grad_temp, 0, self.shape.0, self.shape.1, self.shape.2);
        }
        increment_counter(&self.backward_count);
        self.batch_size += 1.0;

        ////println!("=========================================================");
        ////println!("original_grad: {:?}", cuda_ptr_to_array(original_grads, &[self.shape.0, self.shape.1, self.shape.2]));
        ////println!("input_grad: {:?}", cuda_ptr_to_array(input_grad_ptr, &[self.shape.0, self.shape.1, self.shape.2]));

        //let start: Instant = Instant::now();
        //softmax_backward(
        //    original_grads, input_exp_ptr, input_grad_ptr,
        //    exp_sum_ptr, 
        //    self.in_shape.0, self.in_shape.1, self.in_shape.2
        //);
        //let end = start.elapsed();
        ////println!("backward: {:.6}", end.as_secs_f64());

        ////println!("\noriginal_grads: {:?}", cuda_ptr_to_array(ptr.get_ptr(), ptr.get_shape()));
        //exit(1);
        //ptr.set_ptr(input_grad_ptr, vec![self.in_shape.0, self.in_shape.1, self.in_shape.2]);
        ////println!("\nchained_gradients: {:?}", cuda_ptr_to_array(ptr.get_ptr(), ptr.get_shape()));
        //exit(1);
        ////println!("\nweight_gradients: {:?}", cuda_ptr_to_array(weight_grad_ptr, &[self.in_shape.0, self.in_shape.2, self.out_shape.2]));
        ////println!("\nbias_gradients: {:?}", cuda_ptr_to_array(bias_grad_ptr, &[self.out_shape.0, self.out_shape.1, self.out_shape.2]));
        ////println!("=========================================================");
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
        self.batch_size = 0.0;
    }

    pub fn zero_io(&mut self, ptr_name: &String)
    {
        //let io_ptr: *mut f32 = string_to_ptr(self.io_ptrs.get(ptr_name).unwrap());
        //zeroes_3d_inplace(io_ptr, self.shape.0, self.shape.1, self.shape.2);
        if ptr_name.contains("output")
        {
            self.zero_output = true;
        }
        else if ptr_name.contains("input")
        {
            self.zero_input_grad = true;
        }
    }

    pub fn details(&self)
    {
        println!("Layer type: Softmax");
        println!("Input ptr: {:?} | Input grad ptr: {:?}", self.input_ptr, self.input_grad_ptr);
        println!("Output ptr: {:?} | Output grad ptr: {:?}", self.output_ptr, self.output_grad_ptr);
        println!("Shape: {:?}", self.shape);
    }

    pub fn move_ptrs_to_arrays(&mut self)
    {
        //free_cuda_array(string_to_ptr(&self.norm_ptr));
        self.ptrs_allocated = false;
    }

    pub fn set_ptrs_allocated(&mut self)
    {
        self.ptrs_allocated = true;
    }
}