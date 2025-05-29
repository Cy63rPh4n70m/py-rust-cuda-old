
use serde::{Deserialize, Serialize};

use crate::{cuda_bridge::{activation3d_cuda, activation3d_cuda_backward}, pointer_ops::{counter_is_zero, increment_counter, init_layer_connections, set_zero_counter, string_to_ptr}};

#[derive(Serialize, Deserialize)]
pub struct ActivationCuda
{
    //pub io_ptrs: HashMap<String, String>,

    //pub a: ArrayD<f32>,
    //pub a_ptr: String,
    //pub a_grads_ptr: String,
    //pub a_vel_ptr: String,
    //pub b: ArrayD<f32>,
    //pub b_ptr: String,
    //pub b_grads_ptr: String,
    //pub b_vel_ptr: String,

    pub backward_count: String,
    pub backward_count_prev: String,

    pub input_ptr: String,
    pub input_grad_ptr: String,
    pub output_ptr: String,
    pub output_grad_ptr: String,

    pub output_traverse_ptr: String,

    pub scale: f32,

    pub activation_str: String,
    //pub lr: f32,
    //pub l2: f32,
    pub shape: (usize, usize, usize),
    pub batch_size: f32,

    pub zero_output: bool,
    pub zero_input_grad: bool,

    pub ptrs_allocated: bool,
    pub arrays_allocated: bool
}
impl ActivationCuda
{
    pub fn new(activation_str: &str, batch: usize, rows: usize, cols: usize, scale: f32) -> Self
    {
        return Self {
            //io_ptrs,
            /*
            a: ArrayD::zeros(IxDyn(&[0])),
            a_ptr: String::from("NONE"),
            a_grads_ptr: String::from("NONE"),
            a_vel_ptr: String::from("NONE"),      
            b: ArrayD::zeros(IxDyn(&[0])),
            b_ptr: String::from("NONE"),
            b_grads_ptr: String::from("NONE"),
            b_vel_ptr: String::from("NONE"), 
            */

            input_ptr: "".to_string(),
            input_grad_ptr: "".to_string(),
            output_ptr: "".to_string(),
            output_grad_ptr: "".to_string(),

            output_traverse_ptr: "".to_string(),

            backward_count: String::from("none"),
            backward_count_prev: String::from("none"),
            
            activation_str: activation_str.to_string(),
            //lr,
            //l2,
            scale,
            batch_size: 0.0,
            shape: (batch, rows, cols),
            zero_output: true,
            zero_input_grad: false,

            ptrs_allocated: false,
            arrays_allocated: false,
        }
    }

    pub fn forward(&mut self, str_ptr_in: String) -> String
    {        
        if !self.ptrs_allocated
        {
            //self.input_ptr = input.get_ptr_as_str();
            //self.output_ptr = new_cuda_ptr_str(input_shape);

            //self.input_grads_ptr = new_cuda_ptr_str(input_shape);
            //self.output_grads_ptr = new_cuda_ptr_str(input_shape);

            /*
            if !self.arrays_allocated
            {
                self.a = ArrayD::from_shape_fn(
                    IxDyn(input_shape), 
                    |_| self.range
                );

                self.b = ArrayD::zeros(IxDyn(input_shape));
            }

            self.a_ptr = array_to_cuda_ptr_str(&mut self.a);
            self.a_grads_ptr = new_cuda_ptr_str(input_shape);
            self.a_vel_ptr = new_cuda_ptr_str(input_shape);
            self.b_ptr = array_to_cuda_ptr_str(&mut self.b);
            self.b_grads_ptr = new_cuda_ptr_str(input_shape);
            self.b_vel_ptr = new_cuda_ptr_str(input_shape);
            */

            init_layer_connections(
                &mut self.backward_count, &str_ptr_in, 
                &mut self.backward_count_prev, &mut self.input_ptr, 
                &mut self.input_grad_ptr, &mut self.output_ptr, 
                &mut self.output_grad_ptr, &mut self.output_traverse_ptr, 
                &[self.shape.0, self.shape.1, self.shape.2]
            );

            self.ptrs_allocated = true;
            self.arrays_allocated = true;
        }

        let input_ptr: *mut f32 = string_to_ptr(&self.input_ptr);
        let output_ptr: *mut f32 = string_to_ptr(&self.output_ptr);
        //let a: *mut f32 = string_to_ptr(&self.a_ptr);
        //let b: *mut f32 = string_to_ptr(&self.b_ptr);

        ////println!("{:?}, {:?}", input_ptr, output_ptr);
        //copy_cuda_to_cuda(input_ptr, input.get_ptr(), input_shape);
        activation3d_cuda(
            output_ptr, input_ptr, 
            self.shape.0 as u32, self.shape.1 as u32, self.shape.2 as u32, 
            &self.activation_str, self.scale, self.zero_output
            //a, b
        );
        set_zero_counter(&self.backward_count);

        //println!("input {:?}", cuda_ptr_to_array(input_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
        //println!("output {:?}\n", cuda_ptr_to_array(output_ptr, &[self.shape.0, self.shape.1, self.shape.2]));

        return self.output_traverse_ptr.clone();
        //round_3d_inplace(
        //    output_ptr, 5, 
        //    input_shape[0] as i32, input_shape[1] as i32, input_shape[2] as i32
        //);

        ////println!("-------------------------");
        ////println!("input {:?}\n", cuda_ptr_to_array(input_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
        ////println!("output {:?}\n", cuda_ptr_to_array(output_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
        //input.set_ptr(output_ptr, input_shape.to_vec());
        ////println!("forward activated{:?}", cuda_ptr_to_array(input.get_ptr(), input.get_shape()));
        ////println!("-------------------------");
        //exit(1);
    }

    pub fn backward(&mut self)
    {
        let input_ptr: *mut f32 = string_to_ptr(&self.input_ptr);
        let input_grads_ptr: *mut f32 = string_to_ptr(&self.input_grad_ptr);
        let output_grads_ptr: *mut f32 = string_to_ptr(&self.output_grad_ptr);
        //let a: *mut f32 = string_to_ptr(&self.a_ptr);
        //let b: *mut f32 = string_to_ptr(&self.b_ptr);
        //let a_grads: *mut f32 = string_to_ptr(&self.a_grads_ptr);
        //let b_grads: *mut f32 = string_to_ptr(&self.b_grads_ptr);

        ////println!("current_grads: {:?}", cuda_ptr_to_array(grad_ptr.get_ptr(), grad_ptr.get_shape()));
        ////println!("recored_inputs: {:?}", cuda_ptr_to_array(input_ptr, grad_ptr.get_shape()));

        //let start: Instant = Instant::now();
        if counter_is_zero(&self.backward_count_prev)
        {
            self.zero_input_grad = true;
        }
        activation3d_cuda_backward(
            input_grads_ptr, input_ptr, output_grads_ptr, 
            self.shape.0 as u32, self.shape.1 as u32, self.shape.2 as u32, 
            &self.activation_str, self.scale, self.zero_input_grad
        );
        increment_counter(&self.backward_count);

        self.batch_size += 1.0;

        ////println!("output grad: {:?}", cuda_ptr_to_array(input_grads_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
        ////println!("input grad: {:?}", cuda_ptr_to_array(input_grads_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
        //let end = start.elapsed();
        ////println!("backward: {:.6}", end.as_secs_f64());

        // do not free current pointer, new pointer to be used for next layer
        //grad_ptr.set_ptr(input_grads_ptr, shape.to_vec());
        ////println!("chained_grads: {:?}", cuda_ptr_to_array(grad_ptr.get_ptr(), grad_ptr.get_shape()));
        //exit(1);
    }

    pub fn update_params(&mut self)
    {
        //let a: *mut f32 = string_to_ptr(&self.a_ptr);
        //let b: *mut f32 = string_to_ptr(&self.b_ptr);
        //let a_grads: *mut f32 = string_to_ptr(&self.a_grads_ptr);
        //let b_grads: *mut f32 = string_to_ptr(&self.b_grads_ptr);
        //let a_vel_grads: *mut f32 = string_to_ptr(&self.a_vel_ptr);
        //let b_vel_grads: *mut f32 = string_to_ptr(&self.b_vel_ptr);
        
        //gradient_desc_3d_dense(
        //    self.lr, self.l2,
        //    a, a_grads, a_vel_grads, self.tensor_shape[0], self.tensor_shape[1], self.tensor_shape[2], 
        //    b, b_grads, b_vel_grads, self.tensor_shape[0], self.tensor_shape[1], self.tensor_shape[2], 
        //    false, false, self.batch_size
        //);
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
        println!("Layer type: ACTIVATION");
        println!("Activation function: {:?}", self.activation_str);
        println!("Input ptr: {:?} | Input grad ptr: {:?}", self.input_ptr, self.input_grad_ptr);
        println!("Output ptr: {:?} | Output grad ptr: {:?}", self.output_ptr, self.output_grad_ptr);
        println!("Shape: {:?}", self.shape);
        ////println!("Alpha constants: \n{:?}", self.a);
        ////println!("Beta constants: \n{:?}", self.b);
    }

    pub fn get_param_count(&self) -> usize
    {
        return 0_usize;
    }

    pub fn move_ptrs_to_arrays(&mut self)
    {
        //self.a = cuda_ptr_to_array(string_to_ptr(&self.a_ptr), self.tensor_shape.as_slice());
        //self.b = cuda_ptr_to_array(string_to_ptr(&self.b_ptr), self.tensor_shape.as_slice());

        //free_cuda_array(string_to_ptr(&self.a_ptr));
        //free_cuda_array(string_to_ptr(&self.a_grads_ptr));
        //free_cuda_array(string_to_ptr(&self.b_ptr));
        //free_cuda_array(string_to_ptr(&self.b_grads_ptr));
        //free_cuda_array(string_to_ptr(&self.input_grads_ptr));
        //free_cuda_array(string_to_ptr(&self.output_grads_ptr));

        self.ptrs_allocated = false;
    }

    pub fn set_ptrs_allocated(&mut self)
    {
        self.ptrs_allocated = true;
    }
}