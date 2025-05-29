use ndarray::{Array3, ArrayD, IxDyn};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{cuda_bridge::{embedding_backward, embedding_forward, gradient_desc_3d}, pointer_ops::{array_to_cuda_ptr_str, cuda_ptr_to_array, init_layer_connections, new_cuda_ptr_str, set_zero_counter, string_to_ptr}};

#[derive(Serialize, Deserialize)]
pub struct Embedding2DCuda
{
    pub in_shape: (usize, usize, usize),
    pub out_shape: (usize, usize, usize),

    //pub io_ptrs: HashMap<String, String>,

    pub vocab_size: usize,
    pub embedding_len: usize,
    pub seq_len: usize,

    pub embedding_lookup_mat: ArrayD<f32>,
    pub embedding_lookup_ptr: String,
    pub embedding_lookup_grad_ptr: String,
    pub embedding_lookup_velocity_ptr: String,
    pub embedding_lookup_momentum_ptr: String,
    pub embedding_lookup_grad_count_ptr: String,
    pub embedding_lookup_grad_temp_ptr: String,

    pub backward_count: String,
    pub backward_count_prev: String,

    pub input_ptr: String,
    pub input_grad_ptr: String,
    pub output_ptr: String,
    pub output_grad_ptr: String,

    pub output_traverse_ptr: String,
    
    pub embedding_ptr_allocated: bool, 
    pub embedding_mat_initialised: bool,

    pub zero_output: bool,
    pub zero_input_grad: bool,

    pub batch_size: f32,
}
impl Embedding2DCuda
{
    // weight matrix initialize during first ever run
    pub fn new(vocab_size: usize, embedding_len: usize, seq_len: usize) -> Self
    {
        return Self
        {
            //io_ptrs,

            embedding_len,
            vocab_size,
            seq_len,

            in_shape: (1, 1, seq_len),
            out_shape: (1, seq_len, embedding_len),
            embedding_lookup_mat: ArrayD::zeros(IxDyn(&[1, vocab_size, embedding_len])),
            embedding_lookup_ptr: "".to_string(),
            embedding_lookup_grad_ptr: "".to_string(),
            embedding_lookup_velocity_ptr: "".to_string(),
            embedding_lookup_momentum_ptr: "".to_string(),
            embedding_lookup_grad_count_ptr: "".to_string(),
            embedding_lookup_grad_temp_ptr: "".to_string(),

            input_ptr: "".to_string(),
            input_grad_ptr: "".to_string(),
            output_ptr: "".to_string(),
            output_grad_ptr: "".to_string(),

            output_traverse_ptr: "".to_string(),

            backward_count: String::from("none"),
            backward_count_prev: String::from("none"),

            embedding_ptr_allocated: false,
            embedding_mat_initialised: false,
            zero_output: true,
            zero_input_grad: false,
            batch_size: 0.0,
        }
    }

    // supports batch matrix multiplication unlike cpu
    pub fn forward(&mut self, str_ptr_in: String) -> String
    {
        ////println!("{:?}", cuda_ptr_to_array(input.get_ptr(), input_shape));

        ////println!("{:?}, {:?}, {:?}, {:?}", input.get_ptr(), batch, rows, cols);

        if !self.embedding_ptr_allocated
        {   
            if !self.embedding_mat_initialised
            {   
                let mut embedding_mat: Array3<f32> = Array3::zeros((1, self.vocab_size, self.embedding_len));
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
                
                let range: f32 = (6.0 / ((self.embedding_len + self.embedding_len) as f32)).sqrt();
                embedding_mat.map_mut(|v: &mut f32| *v = rand::thread_rng().gen_range(-range..range));

                self.embedding_lookup_mat = embedding_mat.into_dyn();

                self.embedding_mat_initialised = true;
            }

            self.embedding_lookup_ptr = array_to_cuda_ptr_str(&mut self.embedding_lookup_mat);
            self.embedding_lookup_grad_ptr = new_cuda_ptr_str(&[1, self.vocab_size, self.embedding_len]);
            self.embedding_lookup_velocity_ptr = new_cuda_ptr_str(&[1, self.vocab_size, self.embedding_len]);
            self.embedding_lookup_momentum_ptr = new_cuda_ptr_str(&[1, self.vocab_size, self.embedding_len]);
            self.embedding_lookup_grad_count_ptr = new_cuda_ptr_str(&[1, self.vocab_size, self.embedding_len]);
            self.embedding_lookup_grad_temp_ptr = new_cuda_ptr_str(&[1, self.vocab_size, self.embedding_len]);

            init_layer_connections(
                &mut self.backward_count, &str_ptr_in, 
                &mut self.backward_count_prev, &mut self.input_ptr, 
                &mut self.input_grad_ptr, &mut self.output_ptr, 
                &mut self.output_grad_ptr, &mut self.output_traverse_ptr, 
                &[self.out_shape.0, self.out_shape.1, self.out_shape.2]
            );

            // initialise input pointer, set the input as the result pointer from previous layer
            // tensor struct at this stage will contain the result ptr of the previous layer
            ////////////////////////////////////////////////////////////////
            //self.input_ptr = input.get_ptr_as_str();
            //self.input_ptr = new_cuda_ptr_str(&[batch, rows, cols]);
            //////////////////////////////////////////////////////////////////

            // initialise the result tensor/pointer
            //self.result_ptr = new_cuda_ptr_str(&[1, input_shape[2], self.embedding_len]);
            //self.result_ptr_t = new_cuda_ptr_str(&[batch, self.n_out, rows]);
            
            //self.input_grads_ptr = new_cuda_ptr_str(input_shape);
            //self.output_grads_ptr = new_cuda_ptr_str(&[1, input_shape[2], self.embedding_len]);

            //self.in_shape = (input_shape[0], input_shape[1], input_shape[2]);
            //self.out_shape = (1, input_shape[2], self.embedding_len);

            self.embedding_ptr_allocated = true;
        }

        //let broadcast_buf: String = new_cuda_ptr_str(batch * rows * cols * self.n_out);

        // convert strings to pointers
        let input_ptr: *mut f32 = string_to_ptr(&self.input_ptr);
        let embedding_lookup_ptr: *mut f32 = string_to_ptr(&self.embedding_lookup_ptr);
        let result_ptr: *mut f32 = string_to_ptr(&self.output_ptr);

        ////println!("{:?}", cuda_ptr_to_array(input_ptr, &[self.in_shape.0, self.in_shape.1, self.in_shape.2]));
        ////println!("{:?}", cuda_ptr_to_array(embedding_lookup_ptr, &[1, self.vocab_size, self.embedding_len]));
        ////println!("{:?}", self.embedding_lookup_mat);
        ////println!("{:?}", cuda_ptr_to_array(result_ptr, &[self.out_shape.0, self.out_shape.1, self.out_shape.2]));

        // data does not need to be copied as result ptr from previous layer
        // is set as the input

        // copy data to the input pointer, prevent reallocation

        // parallel perform matrix multiplication
        // and sum with bias tensor
        // result pointer updated
        //let start: Instant = Instant::now();
        //if self.use_tiled || !self.use_tiled
        //{
        embedding_forward(
            input_ptr, self.seq_len, 
            embedding_lookup_ptr, self.vocab_size, self.embedding_len, 
            result_ptr
        );

        set_zero_counter(&self.backward_count);

        //println!("{:?}", cuda_ptr_to_array(input_ptr, &[self.in_shape.0, self.in_shape.1, self.in_shape.2]));
        //println!("{:?}", cuda_ptr_to_array(embedding_lookup_ptr, &[1, self.vocab_size, self.embedding_len]));
        //println!("{:?}", cuda_ptr_to_array(result_ptr, &[1, self.in_shape.2, self.embedding_len]));

        return self.output_traverse_ptr.clone();

        ////println!("{:?}", cuda_ptr_to_array(pos_encoding_ptr, &[self.in_shape.0, self.in_shape.1, self.in_shape.2]));
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
        ////println!("{:?}", cuda_ptr_to_array(input_ptr, &[self.in_shape.0, self.in_shape.1, self.in_shape.2]));
        ////println!("{:?}", cuda_ptr_to_array(embedding_lookup_ptr, &[1, self.vocab_size, self.embedding_len]));
        // overwrite the current pointer with result ptr, to be COPIED to input of next layer
        // current pointer is already recorded by previous layer, don't free
        //input.set_ptr(result_ptr, vec![self.out_shape.0, self.out_shape.1, self.out_shape.2]);

        ////println!("{:?}", cuda_ptr_to_array(result_ptr, &[1, self.seq_len, self.embedding_len]));
        ////println!("-----------------------------");
        //exit(1);
        // previous pointer will be recorded in previous layer

    }

    pub fn backward(&mut self)
    {
        let input_ptr: *mut f32 = string_to_ptr(&self.input_ptr);
        let input_grad_ptr: *mut f32 = string_to_ptr(&self.input_grad_ptr);
        let output_grad_ptr: *mut f32 = string_to_ptr(&self.output_grad_ptr);
        let embedding_lookup_ptr: *mut f32 = string_to_ptr(&self.embedding_lookup_ptr);
        let embedding_lookup_grad_ptr: *mut f32 = string_to_ptr(&self.embedding_lookup_grad_ptr);
        let embedding_lookup_grad_count_ptr: *mut f32 = string_to_ptr(&self.embedding_lookup_grad_count_ptr);
        let embedding_lookup_grad_temp_ptr: *mut f32 = string_to_ptr(&self.embedding_lookup_grad_temp_ptr);
        //let weight_grad_ptr: *mut f32 = string_to_ptr(&self.weight_gradients_ptr);
        //let bias_grad_ptr: *mut f32 = string_to_ptr(&self.bias_gradients_ptr);

        ////println!("{:?}", cuda_ptr_to_array(ptr.get_ptr(), &[self.out_shape.0, self.out_shape.1, self.out_shape.2]));
        ////println!("{:?}", cuda_ptr_to_array(embedding_lookup_grad_ptr, &[1, self.vocab_size, self.embedding_len]));

        embedding_backward(
            input_ptr, self.in_shape.2, 
            embedding_lookup_ptr, self.vocab_size, self.embedding_len, 
            embedding_lookup_grad_ptr, 
            embedding_lookup_grad_temp_ptr,
            embedding_lookup_grad_count_ptr,
            output_grad_ptr, 
            input_grad_ptr
        );

        self.batch_size += 1.0;

        //println!("{:?}", cuda_ptr_to_array(input_ptr, &[1, 1, self.seq_len]));
        //println!("{:?}", cuda_ptr_to_array(embedding_lookup_grad_ptr, &[1, self.vocab_size, self.embedding_len]));
        //exit(1);

        ////println!("{:?}", cuda_ptr_to_array(ptr.get_ptr(), ptr.get_shape()));
        ////println!("\noriginal_grads: {:?}", cuda_ptr_to_array(output_grad_ptr, &[1, self.seq_len, self.embedding_len]));
        //ptr.set_ptr(input_grad_ptr, vec![self.in_shape.0, self.in_shape.1, self.in_shape.2]);
        ////println!("\nchained_gradients: {:?}", cuda_ptr_to_array(ptr.get_ptr(), ptr.get_shape()));
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

    pub fn update_params(&mut self, optimizer_type: i32, lr: f32, l2: f32, alpha: f32, beta: f32)
    {   
        let embedding_lookup_ptr: *mut f32 = string_to_ptr(&self.embedding_lookup_ptr);
        let embedding_lookup_grad_ptr: *mut f32 = string_to_ptr(&self.embedding_lookup_grad_ptr);
        let embedding_lookup_velocity_ptr: *mut f32 = string_to_ptr(&self.embedding_lookup_velocity_ptr);
        let embedding_lookup_momentum_ptr: *mut f32 = string_to_ptr(&self.embedding_lookup_momentum_ptr);

        //scalar_op_3d_inplace(pos_encoding_grad_ptr, self.lr, 2, self.out_shape.0, self.out_shape.1, self.out_shape.2);
        //element_op_3d_inplace(pos_encoding_ptr, pos_encoding_grad_ptr, 1, self.out_shape.0, self.out_shape.1, self.out_shape.2);
        gradient_desc_3d(
            lr, l2,
            embedding_lookup_ptr, embedding_lookup_grad_ptr,
            embedding_lookup_velocity_ptr, embedding_lookup_momentum_ptr,
            1, self.vocab_size, self.embedding_len, 
            embedding_lookup_ptr, embedding_lookup_grad_ptr, 
            embedding_lookup_velocity_ptr, embedding_lookup_momentum_ptr,
            1, self.vocab_size, self.embedding_len,  
            true, false, self.batch_size, optimizer_type, alpha, beta
        );

        self.batch_size = 0.0;
        ////println!("{:?}", cuda_ptr_to_array(weight_grad_ptr, &[self.in_shape.0, self.in_shape.2, self.out_shape.2]))
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
        println!("Layer type: Embedding2D");
        println!("Shape: {:?}", self.in_shape);
        println!("Input ptr: {:?} | Input grad ptr: {:?}", &self.input_ptr, &self.input_grad_ptr);
        println!("Output ptr: {:?} | Output grad ptr: {:?}", &self.output_ptr, &self.output_grad_ptr);
        println!("Embedding matrix:");
        println!("{:?}", self.embedding_lookup_mat);
    }

    pub fn get_param_count(&self) -> usize
    {
        return self.vocab_size * self.embedding_len;
    }

    pub fn move_ptrs_to_arrays(&mut self)
    {
        self.embedding_lookup_mat = cuda_ptr_to_array(string_to_ptr(&self.embedding_lookup_ptr), &[1, self.vocab_size, self.embedding_len]);
        //free_cuda_array(string_to_ptr(&self.embedding_lookup_ptr));
        //free_cuda_array(string_to_ptr(&self.embedding_lookup_grad_ptr));

        self.embedding_ptr_allocated = false;
    }

    pub fn set_ptrs_allocated(&mut self)
    {
        self.embedding_ptr_allocated = true;
    }
}