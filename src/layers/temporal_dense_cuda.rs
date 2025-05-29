use std::process::exit;

use ndarray::{s, Array2, ArrayD, ArrayView1, ArrayView2, ArrayViewMut2, Axis, IxDyn};
use serde::{Deserialize, Serialize};

use crate::{cuda_bridge::{copy_cuda_to_cuda, element_op_3d_inplace, new_cuda_array}, neuralnet::CudaTensorPtr, pointer_ops::{cuda_ptr_to_array, new_cuda_ptr_str, ptr_to_string, string_to_ptr}};

use super::{activation_cuda::ActivationCuda, cuda_layer_enum::CudaLayer, dense_cuda::DenseCuda};

#[derive(Serialize, Deserialize)]
pub struct TemporalDenseCuda
{
    // returns only last vector in sequence if false
    pub return_sequence: bool,
    pub in_shape: usize,
    pub seq_len: usize,
    pub lr: f32,

    pub input: Array2<f32>,
    pub input_ptr_seq: Vec<String>,
    pub hidden_state_seq: Array2<f32>,

    pub sub_chain_n_layers: usize,
    pub sub_chain_layer_len: usize,
    pub hidden_state_len: usize,
    pub activation: String,
    pub hidden_state_activation: String,

    pub input_chain: Vec<Vec<CudaLayer>>,
    pub hidden_chain: Vec<Vec<CudaLayer>>,
    pub combiner_chain: Vec<Vec<CudaLayer>>,

    pub zeroed_hidden_state: ArrayD<f32>,

    // stores a zeroed hidden state to copy into current hidden state
    // current_hidden_ptr will be modified in place
    pub zeroed_hidden_ptr: String,
    pub current_hidden_ptr: String,
    pub input_slice_ptr: String,
    pub stored_hidden_ptr: String,
    pub output_ptr: String,

    pub return_grads_ptr: String,

    // the pointer to be used when backpropagating through the hidden/combiner subsections
    pub main_grad_ptr: String,
    pub stored_main_grad_ptr: String,
    pub main_grad_input_ptr: String,
}

impl TemporalDenseCuda {
    pub fn new(
        sub_chain_n_layers: usize, 
        sub_chain_layer_len: usize, 
        hidden_state_len: usize, 
        activation: &str,
        hidden_state_activation: &str,
        return_sequence: bool,
        lr: f32) -> Self
    {
        return Self
        {
            in_shape: 0,
            seq_len: 0,
            hidden_state_len,
            return_sequence,
            lr,

            input: Array2::zeros((0, 0)),
            input_ptr_seq: Vec::new(),
            hidden_state_seq: Array2::zeros((0, 0)),
            output_ptr: String::from("NONE"),

            sub_chain_layer_len,
            sub_chain_n_layers,
            activation: activation.to_string(),
            hidden_state_activation: hidden_state_activation.to_string(),

            input_chain: Vec::new(),
            hidden_chain: Vec::new(),
            combiner_chain: Vec::new(),

            zeroed_hidden_state: ArrayD::zeros(IxDyn(&[1, 1, hidden_state_len])),
            zeroed_hidden_ptr: String::from("NONE"),
            current_hidden_ptr: String::from("NONE"),
            input_slice_ptr: String::from("NONE"),
            stored_hidden_ptr: String::from("NONE"),
            return_grads_ptr: String::from("NONE"),
            main_grad_ptr: String::from("NONE"),
            stored_main_grad_ptr: String::from("NONE"),
            main_grad_input_ptr: String::from("NONE")
        }
    }

    pub fn forward(&mut self, input_ptr: &mut CudaTensorPtr, apply_dropout: bool)
    {
        if self.in_shape == 0
        {
            self.seq_len = input_ptr.get_shape()[1];
            self.in_shape = input_ptr.get_shape()[2];
            
            self.input = Array2::zeros((self.seq_len, self.in_shape));

            // initialise pointers
            self.current_hidden_ptr = new_cuda_ptr_str(&[1, 1, self.hidden_state_len]);
            self.zeroed_hidden_ptr = new_cuda_ptr_str(&[1, 1, self.hidden_state_len]);
            self.input_slice_ptr = new_cuda_ptr_str(&[1, 1, self.in_shape]);
            self.stored_hidden_ptr = new_cuda_ptr_str(&[1, 1, self.hidden_state_len]);

            if self.return_sequence
            {
                self.output_ptr = new_cuda_ptr_str(&[1, self.seq_len, self.hidden_state_len]);
            }
            else
            {
                self.output_ptr = new_cuda_ptr_str(&[1, 1, self.hidden_state_len]);
            }

            self.return_grads_ptr = new_cuda_ptr_str(&[1, self.seq_len, self.in_shape]);
            self.main_grad_ptr = new_cuda_ptr_str(&[1, 1, self.hidden_state_len]);
            self.stored_main_grad_ptr = new_cuda_ptr_str(&[1, 1, self.hidden_state_len]);
            self.main_grad_input_ptr = new_cuda_ptr_str(&[1, 1, self.hidden_state_len]);

            // create sequence of pointers, each to store length of in_shape
            for _ in 0..self.seq_len
            {
                let ptr: *mut f32 = unsafe { new_cuda_array(self.in_shape as u32) };
                self.input_ptr_seq.push(ptr_to_string(ptr));
                //let ptr: *mut f32 = unsafe { new_cuda_array(self.hidden_state_len as u32) };
                //self.output_ptr_seq.push(ptr_to_string(ptr));
            }

            // initialize the layer chains
            for _ in 0..self.seq_len
            {
                let mut input_sub_chain: Vec<CudaLayer> = Vec::new();
                let mut hidden_sub_chain: Vec<CudaLayer> = Vec::new();
                let mut combiner_sub_chain: Vec<CudaLayer> = Vec::new();
                
                // define number of hidden layers for input, hidden and combiner subsections
                for _ in 0..self.sub_chain_n_layers
                {
                    input_sub_chain.push(CudaLayer::DENSE_CUDA(DenseCuda::new(self.sub_chain_layer_len, 0.0, self.lr)));
                    input_sub_chain.push(CudaLayer::ACTIVATION_CUDA(ActivationCuda::new(&self.activation, self.lr)));

                    hidden_sub_chain.push(CudaLayer::DENSE_CUDA(DenseCuda::new(self.sub_chain_layer_len, 0.0, self.lr)));
                    hidden_sub_chain.push(CudaLayer::ACTIVATION_CUDA(ActivationCuda::new(&self.activation, self.lr)));

                    combiner_sub_chain.push(CudaLayer::DENSE_CUDA(DenseCuda::new(self.sub_chain_layer_len, 0.0, self.lr)));
                    combiner_sub_chain.push(CudaLayer::ACTIVATION_CUDA(ActivationCuda::new(&self.activation, self.lr)));
                }

                // add output layers for each subsection, allow data to transfer
                input_sub_chain.push(CudaLayer::DENSE_CUDA(DenseCuda::new(self.hidden_state_len, 0.0, self.lr)));
                input_sub_chain.push(CudaLayer::ACTIVATION_CUDA(ActivationCuda::new(&self.activation, self.lr)));

                hidden_sub_chain.push(CudaLayer::DENSE_CUDA(DenseCuda::new(self.hidden_state_len, 0.0, self.lr)));
                hidden_sub_chain.push(CudaLayer::ACTIVATION_CUDA(ActivationCuda::new(&self.activation, self.lr)));

                combiner_sub_chain.push(CudaLayer::DENSE_CUDA(DenseCuda::new(self.hidden_state_len, 0.0, self.lr)));
                combiner_sub_chain.push(CudaLayer::ACTIVATION_CUDA(ActivationCuda::new(&self.hidden_state_activation, self.lr)));

                // each timestep has its own set of subsections
                self.input_chain.push(input_sub_chain);
                self.hidden_chain.push(hidden_sub_chain);
                self.combiner_chain.push(combiner_sub_chain);
            }
        }

        //let input_slice_raw_ptr: *mut f32 = string_to_ptr(&self.input_slice_ptr);
        let current_hidden_raw_ptr: *mut f32 = string_to_ptr(&self.current_hidden_ptr);
        let zeroed_hidden_raw_ptr: *mut f32 = string_to_ptr(&self.zeroed_hidden_ptr);
        let stored_hidden_raw_ptr: *mut f32 = string_to_ptr(&self.stored_hidden_ptr);
        let original_input_ptr: *mut f32 = input_ptr.get_ptr();
        let output_ptr: *mut f32 = string_to_ptr(&self.output_ptr);

        unsafe
        {
            // initialise current hidden state ptr with zeros
            copy_cuda_to_cuda(
                current_hidden_raw_ptr, 
                zeroed_hidden_raw_ptr, 
                &[1, 1, self.hidden_state_len]
            );
        }
        
        // create the CudaTensorPtr for current hidden state
        let mut current_hidden_ptr_struct: CudaTensorPtr = CudaTensorPtr
        {
            tensor_ptr: ptr_to_string(current_hidden_raw_ptr),
            shape: vec![1, 1, self.hidden_state_len]
        };

        // create CudaTensorPtr for input slice
        let mut input_slice_ptr_struct: CudaTensorPtr = CudaTensorPtr
        {
            tensor_ptr: ptr_to_string(std::ptr::null_mut()),
            shape: vec![1, 1, self.in_shape]
        };
        
        // loop through each input in sequence
        for i in 0..self.seq_len
        {
            // get horizontal slice of input
            // perform pointer arithemtic and memory copy to input ptr i
            // in pointer vector
            let shifted_input_ptr: *mut f32 = unsafe { original_input_ptr.add(i * self.in_shape) };
            let input_ptr_i: *mut f32 = string_to_ptr(self.input_ptr_seq.get(i).unwrap());

            // copy input slice from shifted input ptr to input ptr
            unsafe { copy_cuda_to_cuda(input_ptr_i, shifted_input_ptr, &[1, 1, self.in_shape]) }

            // set pointer in pointer struct
            input_slice_ptr_struct.set_ptr(input_ptr_i, vec![1, 1, self.in_shape]);

            
            //println!("current hidden: {:?}", cuda_ptr_to_array(current_hidden_ptr_struct.get_ptr(), &[1, 1, self.hidden_state_len]));
            // store the current hidden state for residual connection
            // to be used after hidden and combined subsections
            unsafe {
                copy_cuda_to_cuda(
                    stored_hidden_raw_ptr, 
                    current_hidden_ptr_struct.get_ptr(), 
                    &[1, 1, self.hidden_state_len]
                )
            }
            
            // passing through input net
            // input and hidden chains have same length
            // combiner requires the summation of both chains to work
            for j in 0..self.input_chain[i].len()
            {
                let input_chain_layer: &mut CudaLayer = self.input_chain[i].get_mut(j).unwrap();
                let hidden_chain_layer: &mut CudaLayer = self.hidden_chain[i].get_mut(j).unwrap();
                input_chain_layer.forward(&mut input_slice_ptr_struct, apply_dropout);
                hidden_chain_layer.forward(&mut current_hidden_ptr_struct, apply_dropout);
            }
                        
            // sum the output from input and hidden subsections for input to combined subsection
            // use the hidden state struct to store the final result for combined input
            unsafe {
                element_op_3d_inplace(
                    current_hidden_ptr_struct.get_ptr(), 
                    input_slice_ptr_struct.get_ptr(), 0, 
                    1, 1, self.hidden_state_len
                );
            }
            //let mut output3: ArrayD<f32> = input_slice + current_hidden_state;


            // pass the combined input through combiner subsection using current hidden struct
            for layer in &mut self.combiner_chain[i]
            {
                layer.forward(&mut current_hidden_ptr_struct, apply_dropout);
            }

            // apply residual connection for final output
            // between combined output and the original hidden state
            unsafe {
                element_op_3d_inplace(
                    current_hidden_ptr_struct.get_ptr(), 
                    stored_hidden_raw_ptr, 0, 
                    1, 1, self.hidden_state_len
                );
            }
            //current_hidden_state = output3 + input_hidden_state;

            // get slice of hidden_state_seq and add current hidden state
            // to hidden_state_seq (also plays the role of inputs for next
            // temporal dense layer)

            if self.return_sequence // output ptr is (1, seq_len, hidden_len)
            {
                let shifted_output_ptr: *mut f32 = unsafe { output_ptr.add(i * self.hidden_state_len) };
                unsafe {
                    copy_cuda_to_cuda(
                        shifted_output_ptr, 
                        current_hidden_ptr_struct.get_ptr(), 
                        &[1, 1, self.hidden_state_len]
                    )
                }
            }
            else // output ptr is (1, 1, self.hidden_len), only copy at final timestep
            {
                if i == self.seq_len - 1
                {
                    unsafe {
                        copy_cuda_to_cuda(
                            output_ptr, 
                            current_hidden_ptr_struct.get_ptr(), 
                            &[1, 1, self.hidden_state_len]
                        )
                    }
                }
            }
            //println!("final: {:?}", cuda_ptr_to_array(output_ptr, &[1, self.seq_len, self.hidden_state_len]));
            //println!("current hidden: {:?}\n", cuda_ptr_to_array(current_hidden_ptr_struct.get_ptr(), &[1, 1, self.hidden_state_len]));

            //let mut hidden_state_seq_slice: ArrayViewMut2<f32> = 
            //    self.hidden_state_seq.slice_mut(s![i..i + 1, ..]);
            //hidden_state_seq_slice += &current_hidden_state;

            //println!("current hidden: {:?}", cuda_ptr_to_array(current_hidden_ptr_struct.get_ptr(), &[1, 1, self.hidden_state_len]));
            //println!("output_ptr: {:?}", cuda_ptr_to_array(output_ptr, &[1, 1, self.hidden_state_len]));
        }
        //exit(1);

        let shape: Vec<usize>;
        if self.return_sequence
        {
            shape = vec![1, self.seq_len, self.hidden_state_len];
        }
        else
        {
            shape = vec![1, 1, self.hidden_state_len];
        }

        input_ptr.set_ptr(output_ptr, shape);
        //println!("output_ptr: {:?}", cuda_ptr_to_array(input_ptr.get_ptr(), shape.as_slice()));
        //exit(1);

        /*
        let output: ArrayD<f32>;
        if self.return_sequence
        {
            output = self.hidden_state_seq.clone().into_dyn();
        }
        else
        {
            let last_hidden_state_slice: ArrayView2<f32> = 
                self.hidden_state_seq.slice(
                    s![self.seq_len - 1..self.seq_len, ..]
                );
            
            let last_hidden_state_slice: ArrayView1<f32> = 
                last_hidden_state_slice.slice(s![0, ..]);
            
            output = last_hidden_state_slice.into_owned().insert_axis(Axis(0)).into_dyn();
        }
        //println!("{:?}", output);
        //std::process::exit(1);
        return output;
        */
    }

    pub fn backward(&mut self, original_grad_ptr: &mut CudaTensorPtr, apply_dropout: bool)
    {   
        let main_grad_ptr: *mut f32 = string_to_ptr(&self.main_grad_ptr);
        let stored_main_grad_ptr: *mut f32 = string_to_ptr(&self.stored_main_grad_ptr);
        let return_grads: *mut f32 = string_to_ptr(&self.return_grads_ptr);
        let main_grad_input_ptr: *mut f32 = string_to_ptr(&self.main_grad_input_ptr);
        
        // grad will automatically have the correct shape
        // either (1, seq_len, hidden_len) or (1, 1, hidden_len)
        // depending on return sequence boolean

        /**/
        if self.return_sequence
        {
            unsafe 
            {
                // pointer arithmetic, shift to last gradient vector in sequence
                let original_grad_ptr_shifted: *mut f32 = 
                    original_grad_ptr.get_ptr().add(
                        (self.seq_len - 1) * self.hidden_state_len
                    );
                
                // copy memory to main_grad_ptr
                copy_cuda_to_cuda(
                    main_grad_ptr, original_grad_ptr_shifted, 
                    &[1, 1, self.hidden_state_len]
                );
            }
        }
        else
        {
            unsafe
            {
                copy_cuda_to_cuda(
                    main_grad_ptr, original_grad_ptr.get_ptr(), 
                    &[1, 1, self.hidden_state_len]
                );
            }
        }
        /**/
        //println!("{:?}", cuda_ptr_to_array(main_grad_ptr, &[1, 1, self.hidden_state_len]));
        //exit(1);

        // initialise pointer struct
        let mut main_grad_ptr_struct: CudaTensorPtr = CudaTensorPtr
        {
            tensor_ptr: ptr_to_string(main_grad_ptr),
            shape: vec![1, 1, self.hidden_state_len]
        };
        
        
        // iterate through sub chains in reverse
        for i in (0..self.seq_len).rev()
        {   
            // follow graph diagram for elaboration
            // obtain the gradients from the previous layer at timestep
            // t - 1
            if self.return_sequence && i < self.seq_len - 1
            {
                unsafe
                {
                    let original_grad_ptr_shifted: *mut f32 = 
                        original_grad_ptr.get_ptr().add(
                            i * self.hidden_state_len
                        );
                    
                    element_op_3d_inplace(
                        main_grad_ptr_struct.get_ptr(), original_grad_ptr_shifted, 
                        0, 
                        1, 1, self.hidden_state_len
                    );

                    //let loss_r_output_slice: ArrayView2<f32> = 
                    //    loss_r_outputs.slice(s![i..i + 1, ..]);
                    //let loss_r_output_slice: ArrayView1<f32> = loss_r_output_slice.slice(s![0, ..]);
                    //main_grads += &loss_r_output_slice;
                }
            }

            // for residual connection, store current main gradients
            unsafe 
            { 
                copy_cuda_to_cuda(
                    stored_main_grad_ptr, main_grad_ptr_struct.get_ptr(), 
                    &[1, 1, self.hidden_state_len]
                )
            }
            //let main_grads_temp: ArrayD<f32> = main_grads.clone();

            // backpropagate through combiner
            //println!("main grad before combine: {:?}\n", cuda_ptr_to_array(main_grad_ptr_struct.get_ptr(), &[1, 1, self.hidden_state_len]));
            for layer in self.combiner_chain[i].iter_mut().rev()
            {
                layer.backward(&mut main_grad_ptr_struct, apply_dropout);
            }
            //println!("main grad after combine: {:?}\n", cuda_ptr_to_array(main_grad_ptr_struct.get_ptr(), &[1, 1, self.hidden_state_len]));

            // backpropagate through input net and copy resulting
            // gradients to loss_r_inputs
            // create pointer struct

            let mut grads_for_input_subsection: CudaTensorPtr = CudaTensorPtr
            {
                tensor_ptr: ptr_to_string(main_grad_input_ptr),
                shape: vec![1, 1, self.hidden_state_len]
            };
            unsafe { 
                copy_cuda_to_cuda(
                    grads_for_input_subsection.get_ptr(), 
                    main_grad_ptr_struct.get_ptr(), 
                    &[1, 1, self.hidden_state_len]
                );
            }
            //println!("main grad copied before input: {:?}\n", cuda_ptr_to_array(grads_for_input_subsection.get_ptr(), &[1, 1, self.hidden_state_len]));
            // backpropagate using main grads copy
            for layer in self.input_chain[i].iter_mut().rev()
            {
                layer.backward(&mut grads_for_input_subsection, apply_dropout);
            }
            //println!("main grad copied after input: {:?}\n", cuda_ptr_to_array(grads_for_input_subsection.get_ptr(), &[1, 1, self.in_shape]));

            // shape of (1, 1, self.in_shape)
            // transfer main_grads_temp to i slice of loss_r_inputs
            //println!("current return grads: {:?}\n", cuda_ptr_to_array(return_grads, &[1, self.seq_len, self.in_shape]));
            unsafe
            {
                // obtain slice from return grads and copy input grads to it
                let return_grad_ptr_shifted: *mut f32 = return_grads.add(i * self.in_shape);
                copy_cuda_to_cuda(
                    return_grad_ptr_shifted, 
                    grads_for_input_subsection.get_ptr(),
                    &[1, 1, self.in_shape]
                )
            }
            //println!("added input grad to return grad: {:?}\n", cuda_ptr_to_array(return_grads, &[1, self.seq_len, self.in_shape]));

            // resume main grads through hidden state layers
            // and to next (previous) iteration
            //println!("resume main grad through hidden: {:?}\n", cuda_ptr_to_array(main_grad_ptr_struct.get_ptr(), &[1, 1, self.hidden_state_len]));
            for layer in self.hidden_chain[i].iter_mut().rev()
            {
                layer.backward(&mut main_grad_ptr_struct, apply_dropout);
            }
            //println!("main grad after hidden: {:?}\n", cuda_ptr_to_array(main_grad_ptr_struct.get_ptr(), &[1, 1, self.hidden_state_len]));

            // add the gradients from residual connection
            // main grads that were initially stored before passing through all subsections
            unsafe 
            { 
                element_op_3d_inplace(
                    main_grad_ptr_struct.get_ptr(), 
                    stored_main_grad_ptr, 
                    0, 
                    1, 1, self.hidden_state_len
                ); 
            }
            //println!("main grad added with residual {:?}\n", cuda_ptr_to_array(main_grad_ptr_struct.get_ptr(), &[1, 1, self.hidden_state_len]));
            //exit(1);
            //main_grads += &main_grads_temp;
        }

        original_grad_ptr.set_ptr(return_grads, vec![1, self.seq_len, self.in_shape]);
        //exit(1);
    }

    pub fn update_params(&mut self)
    {
        for i in 0..self.seq_len
        {
            for j in 0..self.input_chain[i].len()
            {
                self.input_chain[i][j].update_params();
                self.hidden_chain[i][j].update_params();
                self.combiner_chain[i][j].update_params();
            }
        }
    }

    pub fn zero_grads(&mut self)
    {
        for i in 0..self.seq_len
        {
            for j in 0..self.input_chain[i].len()//self.sub_chain_n_layers + 2
            {
                self.input_chain[i][j].zero_grads();
                self.hidden_chain[i][j].zero_grads();
                self.combiner_chain[i][j].zero_grads();
            }
        }
    }

    pub fn details(&self)
    {
        for layer_chain in &self.input_chain
        {
            for layer in layer_chain
            {
                layer.details();
            }
        }
        for layer_chain in &self.hidden_chain
        {
            for layer in layer_chain
            {
                layer.details();
            }
        }
        for layer_chain in &self.combiner_chain
        {
            for layer in layer_chain
            {
                layer.details();
            }
        }
        //println!("{:?}", self.biases);

    }
}