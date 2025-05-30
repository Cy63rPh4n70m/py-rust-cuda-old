

use ndarray::{ArrayD, IxDyn};
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::de::IoRead;

use crate::{cuda_bridge::{free_cuda_array, gradient_desc_3d, matmul_add_bias_back, matmul_add_bias_tiled, new_cuda_array}, math_functions::random_float_vec, neuralnet::TraversePtrs, pointer_ops::{array_to_cuda_ptr_str, counter_is_zero, cuda_ptr_to_array, get_traverse_str_ptr, increment_counter, init_trav_in_ptrs, new_cuda_ptr_str, ptr_to_string, set_zero_counter, string_to_ptr, vec_to_cuda_ptr}};

use super::layer_cuda::{LayerCuda, AllocationStatus, IOPtrs, ParameterPtrs, WeightTensors};

pub struct DenseCuda
{
    pub name: String,
    pub use_bias: bool,
    pub batch_size: f32,
    
    pub io_ptrs: IOPtrs,
    pub parameter_ptrs: ParameterPtrs,
    pub allocation_status: AllocationStatus,
    pub weight_tensors: WeightTensors

}
impl DenseCuda
{
    // weight matrix initialize during first ever run
    pub fn new(
        n_in: usize, n_out: usize, batch: usize, rows: usize, use_bias: bool, name: &str
    ) -> Self
    {        
        return Self
        {
            //io_ptrs,
            name: name.to_string(),
            use_bias,
            batch_size: 0.0,
            io_ptrs: IOPtrs::new((batch, rows, n_in), (batch, rows, n_out)),
            parameter_ptrs: ParameterPtrs::new(),
            allocation_status: AllocationStatus::new(),
            weight_tensors: WeightTensors::new()
        }
    }
}

impl LayerCuda for DenseCuda
{
    // supports batch matrix multiplication unlike cpu
    fn forward(&mut self, trav_ptr_in: *mut TraversePtrs, trav_ptr_weight: *mut TraversePtrs, _use_dropout: bool) -> *mut TraversePtrs
    {
        let batch: usize = self.io_ptrs.in_shape.0;
        let rows: usize = self.io_ptrs.in_shape.1;
        let cols: usize = self.io_ptrs.in_shape.2;

        let range: f32 = (6.0 / (cols + self.io_ptrs.in_shape.2) as f32).sqrt();

        if !self.allocation_status.ptrs_allocated
        {   
            // initialise biases and pointers
            if !self.allocation_status.arrays_allocated
            {
                // initialize bias arrays
                self.weight_tensors.biases = random_float_vec(
                    batch * rows * self.io_ptrs.out_shape.2, 
                    -0.001, 0.001
                );
            }

            // initialize bias pointers
            self.parameter_ptrs.biases_ptr = vec_to_cuda_ptr(&mut self.weight_tensors.biases);
            self.parameter_ptrs.bias_grad_ptr = new_cuda_array(
                (batch * rows * self.io_ptrs.out_shape.2) as u32
            );
            self.parameter_ptrs.bias_vel_ptr = new_cuda_array(
                (batch * rows * self.io_ptrs.out_shape.2) as u32
            );
            self.parameter_ptrs.bias_moment_ptr = new_cuda_array(
                (batch * rows * self.io_ptrs.out_shape.2) as u32
            );


            // decide whether to create weight based on pointer availability
            if trav_ptr_weight.is_null()
            {
                // create pointer and vector for weights
                if !self.allocation_status.arrays_allocated
                {
                    self.weight_tensors.weight = random_float_vec(
                        batch * cols * self.io_ptrs.out_shape.2, 
                        -range, range
                    );
                }

                // initialize weight ptrs
                self.parameter_ptrs.weight_ptr = vec_to_cuda_ptr(&mut self.weight_tensors.weight);
                self.parameter_ptrs.weight_grad_ptr = new_cuda_array(
                    (batch * cols * self.io_ptrs.out_shape.2) as u32
                );
            }
            else
            {
                unsafe
                {
                    self.parameter_ptrs.weight_ptr = (*trav_ptr_weight).ptr;
                    self.parameter_ptrs.weight_grad_ptr = (*trav_ptr_weight).grad_ptr;
                    self.io_ptrs.backward_count_weight_prev = (*trav_ptr_weight).backward_pass_count;
                }
            }

            self.parameter_ptrs.weight_vel_ptr = new_cuda_array(
                (batch * cols * self.io_ptrs.out_shape.2) as u32
            );
            self.parameter_ptrs.weight_moment_ptr = new_cuda_array(
                (batch * cols * self.io_ptrs.out_shape.2) as u32
            );

            self.allocation_status.arrays_allocated = true;

            // initialize the input traversal pointers from input/previous layer
            init_trav_in_ptrs(
                &trav_ptr_in, &mut self.io_ptrs.backward_count,
                &mut self.io_ptrs.backward_count_in_prev, 
                &mut self.io_ptrs.input_ptr, &mut self.io_ptrs.input_grad_ptr, 
                &mut self.io_ptrs.output_ptr, &mut self.io_ptrs.output_grad_ptr, 
                &mut self.io_ptrs.output_traverse_ptr, 
                batch * rows * self.io_ptrs.out_shape.2
            );
        }

        matmul_add_bias_tiled(
            self.io_ptrs.input_ptr, batch as u32, rows as u32, cols as u32, 
            self.parameter_ptrs.weight_ptr, batch as u32, cols as u32, self.io_ptrs.out_shape.2 as u32,
            self.io_ptrs.output_ptr, self.parameter_ptrs.biases_ptr, 
            self.use_bias, self.allocation_status.zero_output
        );

        set_zero_counter(self.io_ptrs.backward_count);

        //println!("input: {:?}", cuda_ptr_to_array(input_ptr, &[batch, rows, cols]));
        //println!("weight: {:?}", cuda_ptr_to_array(weight_ptr, &[batch, cols, self.n_out]));
        //println!("output: {:?}\n", cuda_ptr_to_array(output_ptr, &[batch, rows, self.n_out]));

        return self.io_ptrs.output_traverse_ptr;

    }

    fn backward(&mut self)
    {

        ////println!("=========================================================");
        ////println!("input_array: {:?}", cuda_ptr_to_array(input_ptr, &[self.in_shape.0, self.in_shape.1, self.in_shape.2]));
        ////println!("\nweights: {:?}", cuda_ptr_to_array(weight_ptr, &[self.in_shape.0, self.in_shape.2, self.out_shape.2]));
        ////println!("\nmatmul result: {:?}", cuda_ptr_to_array(output_ptr, &[self.out_shape.0, self.out_shape.1, self.out_shape.2]));

        //let start: Instant = Instant::now();
        ////println!("\nchained_gradients: {:?}", cuda_ptr_to_array(input_grad_ptr, &[self.in_shape.0, self.in_shape.1, self.in_shape.2]));
        if counter_is_zero(self.io_ptrs.backward_count_in_prev)
        {
            self.allocation_status.zero_input_grad = true;
        }
        
        ////println!("{:?}", self.backward_count_weight_prev);
        if counter_is_zero(self.io_ptrs.backward_count_weight_prev)
        {
            self.allocation_status.zero_weight_grad = true;
        }
        
        matmul_add_bias_back(
            self.io_ptrs.input_grad_ptr, 
            self.io_ptrs.in_shape.0 as u32, self.io_ptrs.in_shape.1 as u32, 
            self.io_ptrs.in_shape.2 as u32, 
            self.parameter_ptrs.weight_grad_ptr, 
            self.io_ptrs.in_shape.0 as u32, self.io_ptrs.in_shape.2 as u32, 
            self.io_ptrs.out_shape.2 as u32, 
            self.parameter_ptrs.bias_grad_ptr, 
            self.io_ptrs.out_shape.0 as u32, self.io_ptrs.out_shape.1 as u32, 
            self.io_ptrs.out_shape.2 as u32,
            self.io_ptrs.output_grad_ptr, 
            self.io_ptrs.input_ptr,
            self.parameter_ptrs.weight_ptr,
            self.use_bias,
            self.allocation_status.zero_input_grad,
            self.allocation_status.zero_weight_grad
        );

        increment_counter(self.io_ptrs.backward_count);

        self.batch_size += 1.0;

        //let end = start.elapsed();
        ////println!("backward: {:.6}", end.as_secs_f64());

        ////println!("\noriginal_grads: {:?}", cuda_ptr_to_array(output_grad_ptr, &[self.out_shape.0, self.out_shape.1, self.out_shape.2]));
        //ptr.set_ptr(input_grad_ptr, vec![self.in_shape.0, self.in_shape.1, self.in_shape.2]);
        ////println!("\nchained_gradients: {:?}", cuda_ptr_to_array(input_grad_ptr, &[self.in_shape.0, self.in_shape.1, self.in_shape.2]));
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

    //pub fn get_weight_grads_ptr(&mut self) -> String
    //{
    //    return self.weight_gradients_ptr.clone();
    //}

    fn update_params(&mut self, optimizer_type: i32, lr: f32, l2: f32, alpha: f32, beta: f32)
    {   
        gradient_desc_3d(
            lr, l2,
            self.parameter_ptrs.weight_ptr, self.parameter_ptrs.weight_grad_ptr, 
            self.parameter_ptrs.weight_vel_ptr, self.parameter_ptrs.weight_moment_ptr,
            self.io_ptrs.in_shape.0, self.io_ptrs.in_shape.2, self.io_ptrs.out_shape.2,
            self.parameter_ptrs.biases_ptr, self.parameter_ptrs.bias_grad_ptr, 
            self.parameter_ptrs.bias_vel_ptr, self.parameter_ptrs.bias_moment_ptr,
            self.io_ptrs.out_shape.0, self.io_ptrs.out_shape.1, self.io_ptrs.out_shape.2,
            self.use_bias, false, self.batch_size, optimizer_type, alpha, beta
        );

        self.batch_size = 0.0;
    }

    fn details(&self)
    {
        println!("Layer type: DENSE | Layer name: {:?}", self.name);
        println!("Input ptr: {:?} | Input grad ptr: {:?}", self.io_ptrs.input_ptr, self.io_ptrs.input_grad_ptr);
        println!("Weight ptr: {:?} | Weight grad ptr: {:?}", self.parameter_ptrs.weight_ptr, self.parameter_ptrs.weight_grad_ptr);
        println!("Output ptr: {:?} | Output grad ptr: {:?}", self.io_ptrs.output_ptr, self.io_ptrs.output_grad_ptr);
        println!("Input shape: {:?}", self.io_ptrs.in_shape);
        println!("Output shape: {:?}", self.io_ptrs.out_shape);
        println!("Weights: \n{:?}", self.weight_tensors.weight);
        if self.use_bias
        {
            println!("Biases: \n{:?}", self.weight_tensors.biases);
        }
    }

    fn get_param_count(&self) -> usize
    {
        let mut count: usize = 0;
        if self.use_bias
        {
            count += self.io_ptrs.out_shape.0 * self.io_ptrs.out_shape.1 * self.io_ptrs.out_shape.2;
        }

        count += self.io_ptrs.in_shape.0 * self.io_ptrs.in_shape.2 * self.io_ptrs.out_shape.2;

        return count
    }

    fn move_ptrs_to_arrays(&mut self)
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

        if self.use_bias
        {
            self.biases = cuda_ptr_to_array(string_to_ptr(&self.biases_ptr), 
                &[self.out_shape.0, self.out_shape.1, self.out_shape.2]);
        }
        else
        {
            self.bias_array_allocated = false;
        }

        self.weight_ptr_allocated = false;
        self.bias_ptr_allocated = false;
    }

    pub fn set_ptrs_allocated(&mut self)
    {
        self.weight_ptr_allocated = true;
        self.bias_ptr_allocated = true;
    }
}