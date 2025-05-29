use ndarray::{s, Array2, ArrayD, ArrayView1, ArrayView2, ArrayViewMut2, Axis, IxDyn};
use serde::{Deserialize, Serialize};

use super::{activation::Activation, dense::Dense, layer_enum::Layer};

#[derive(Serialize, Deserialize)]
pub struct TemporalDense
{
    // returns only last vector in sequence if false
    pub return_sequence: bool,
    pub in_shape: usize,
    pub seq_len: usize,
    pub lr: f32,

    pub input_vec_seq: Array2<f32>,
    pub hidden_state_seq: Array2<f32>,

    pub sub_chain_n_layers: usize,
    pub sub_chain_layer_len: usize,
    pub hidden_state_len: usize,
    pub activation: String,
    pub hidden_state_activation: String,

    pub input_chain: Vec<Vec<Layer>>,
    pub hidden_chain: Vec<Vec<Layer>>,
    pub combiner_chain: Vec<Vec<Layer>>,

    pub zeroed_hidden_state: ArrayD<f32>,
}

impl TemporalDense {
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

            input_vec_seq: Array2::zeros((0, 0)),
            hidden_state_seq: Array2::zeros((0, 0)),

            sub_chain_layer_len,
            sub_chain_n_layers,
            activation: activation.to_string(),
            hidden_state_activation: hidden_state_activation.to_string(),

            input_chain: Vec::new(),
            hidden_chain: Vec::new(),
            combiner_chain: Vec::new(),

            zeroed_hidden_state: ArrayD::zeros(IxDyn(&[1, hidden_state_len]))
        }
    }

    pub fn forward(&mut self, input_seq: ArrayD<f32>, apply_dropout: bool) -> ArrayD<f32>
    {
        if self.in_shape == 0
        {
            self.seq_len = input_seq.shape()[0];
            self.in_shape = input_seq.shape()[1];
            
            self.input_vec_seq = Array2::zeros((self.seq_len, self.in_shape));
            self.hidden_state_seq = Array2::zeros((self.seq_len, self.hidden_state_len));

            // initialize the layer chains based on received seq len
            for _ in 0..self.seq_len
            {
                let mut input_sub_chain: Vec<Layer> = Vec::new();
                let mut hidden_sub_chain: Vec<Layer> = Vec::new();
                let mut combiner_sub_chain: Vec<Layer> = Vec::new();

                for _ in 0..self.sub_chain_n_layers
                {
                    input_sub_chain.push(Layer::DENSE(Dense::new(self.sub_chain_layer_len, 0.0, self.lr)));
                    input_sub_chain.push(Layer::ACTIVATION(Activation::new(&self.activation, self.lr)));

                    hidden_sub_chain.push(Layer::DENSE(Dense::new(self.sub_chain_layer_len, 0.0, self.lr)));
                    hidden_sub_chain.push(Layer::ACTIVATION(Activation::new(&self.activation, self.lr)));

                    combiner_sub_chain.push(Layer::DENSE(Dense::new(self.sub_chain_layer_len, 0.0, self.lr)));
                    combiner_sub_chain.push(Layer::ACTIVATION(Activation::new(&self.activation, self.lr)));
                }

                // add output layers
                input_sub_chain.push(Layer::DENSE(Dense::new(self.hidden_state_len, 0.0, self.lr)));
                input_sub_chain.push(Layer::ACTIVATION(Activation::new(&self.activation, self.lr)));

                hidden_sub_chain.push(Layer::DENSE(Dense::new(self.hidden_state_len, 0.0, self.lr)));
                hidden_sub_chain.push(Layer::ACTIVATION(Activation::new(&self.activation, self.lr)));

                combiner_sub_chain.push(Layer::DENSE(Dense::new(self.hidden_state_len, 0.0, self.lr)));
                combiner_sub_chain.push(Layer::ACTIVATION(Activation::new(&self.hidden_state_activation, self.lr)));

                self.input_chain.push(input_sub_chain);
                self.hidden_chain.push(hidden_sub_chain);
                self.combiner_chain.push(combiner_sub_chain);
            }
        }

        let input_seq: Array2<f32> = 
            input_seq.into_shape((self.seq_len, self.in_shape)).unwrap();
        
        self.input_vec_seq.fill(0.0);
        self.hidden_state_seq.fill(0.0);
        self.input_vec_seq += &input_seq;

        // initialize first hidden state
        let mut current_hidden_state: ArrayD<f32> = 
            self.zeroed_hidden_state.clone();
        
        // loop through each input in sequence
        for i in 0..self.seq_len
        {
            // get horizontal slice of input
            let mut input_slice: ArrayD<f32> = 
                self.input_vec_seq.slice(s![i..i + 1, ..]).into_owned().into_dyn();
            
            //println!("{}", self.input_chain.len());
            //println!("{}", self.hidden_chain.len());
            //std::process::exit(1);
            // passing through input net
            for layer in &mut self.input_chain[i]
            {
                input_slice = layer.forward(input_slice, apply_dropout);
            }
            
            // temporary hidden state for residual connection
            let input_hidden_state: ArrayD<f32> = current_hidden_state.clone();
            // passing through hidden net
            for layer in &mut self.hidden_chain[i]
            {
                current_hidden_state = layer.forward(current_hidden_state, apply_dropout);
            }

            // combined output passing through combiner
            let mut output3: ArrayD<f32> = input_slice + current_hidden_state;
            for layer in &mut self.combiner_chain[i]
            {
                output3 = layer.forward(output3, apply_dropout);
            }

            // apply residual connection for final output
            current_hidden_state = output3 + input_hidden_state;

            // get slice of hidden_state_seq and add current hidden state
            // to hidden_state_seq (also plays the role of inputs for next
            // temporal dense layer)
            let mut hidden_state_seq_slice: ArrayViewMut2<f32> = 
                self.hidden_state_seq.slice_mut(s![i..i + 1, ..]);
            hidden_state_seq_slice += &current_hidden_state;
        }
        // pass to final layers
        //for layer in &mut self.final_layers
        //{
        //    current_hidden_state = layer.forward(current_hidden_state, apply_dropout);
        //}
        //println!("{:?}", current_hidden_state);
        //std::process::exit(1);

        // now final output
        //return current_hidden_state;

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
    }

    pub fn backward(&mut self, loss_r_outputs: ArrayD<f32>, apply_dropout: bool) -> ArrayD<f32>
    {
        //println!("{:?}", loss_r_outputs);
        //println!("{:?}", 2);
        //for layer in self.final_layers.iter_mut().rev()
        //{
            //println!("{:?}", 1);
        //    loss_r_outputs = layer.backward(loss_r_outputs, apply_dropout);
        //}
        //println!("{:?}", loss_r_outputs);

        let mut loss_r_inputs: ArrayD<f32> = ArrayD::zeros(IxDyn(&[self.seq_len, self.in_shape]));
        let mut main_grads: ArrayD<f32>;
        if self.return_sequence
        {
            // just obtain gradients at last
            let loss_r_outputs_last: ArrayView2<f32> = 
                loss_r_outputs.slice(
                    s![self.seq_len - 1..self.seq_len, ..]
                );
            
            let loss_r_outputs_last: ArrayView1<f32> = 
                loss_r_outputs_last.slice(s![0, ..]);
            
            main_grads = loss_r_outputs_last.into_owned().into_dyn();
        }
        else
        {
            main_grads = loss_r_outputs.clone();
        }
        
        /**/
        // iterate through sub chains in reverse
        for i in (0..self.seq_len).rev()
        {   
            // follow graph diagram for elaboration
            // obtain the gradients from the previous layer at timestep
            // t - 1
            if self.return_sequence && i < self.seq_len - 1
            {
                let loss_r_output_slice: ArrayView2<f32> = 
                    loss_r_outputs.slice(s![i..i + 1, ..]);
                let loss_r_output_slice: ArrayView1<f32> = loss_r_output_slice.slice(s![0, ..]);
                main_grads += &loss_r_output_slice;
            }

            // for residual connection
            let main_grads_temp: ArrayD<f32> = main_grads.clone();

            // backpropagate through combiner
            for layer in self.combiner_chain[i].iter_mut().rev()
            {
                main_grads = layer.backward(main_grads, apply_dropout);
            }

            // backpropagate through input net and copy resulting
            // gradients to loss_r_inputs
            let mut main_grads_inputs: ArrayD<f32> = main_grads.clone();
            for layer in self.input_chain[i].iter_mut().rev()
            {
                main_grads_inputs = layer.backward(main_grads_inputs, apply_dropout);
            }

            // transfer main_grads_temp to i slice of loss_r_inputs
            let mut loss_r_inputs_slice: ArrayViewMut2<f32> = 
                loss_r_inputs.slice_mut(s![i..i + 1, ..]);
            loss_r_inputs_slice += &main_grads_inputs;

            // continue main grads through hidden state layers
            // and to next (previous) iteration
            for layer in self.hidden_chain[i].iter_mut().rev()
            {
                main_grads = layer.backward(main_grads, apply_dropout);
            }

            // gradients from residual connection
            main_grads += &main_grads_temp;
        }
        /**/
        //println!("\x1b[32m-------------------------------------------\x1b[0m");
        //println!("{:?}", loss_r_inputs);
        //std::process::exit(1);
        return loss_r_inputs;
    }

    pub fn update_params(&mut self, lr: f32, min_lr: f32, max_lr: f32)
    {
        for i in 0..self.seq_len
        {
            //println!("{:?}, {:?}, {:?}", self.input_chain[i].len(), self.hidden_chain[i].len(), self.combiner_chain[i].len());
            //println!("{:?}", self.sub_chain_n_layers);
            for j in 0..self.input_chain[i].len()
            {
                self.input_chain[i][j].update_params(lr, min_lr, max_lr);
                self.hidden_chain[i][j].update_params(lr, min_lr, max_lr);
                self.combiner_chain[i][j].update_params(lr, min_lr, max_lr);
            }
        }
        //exit(1);
    }

    pub fn zero_grads(&mut self)
    {
        for i in 0..self.seq_len
        {
            for j in 0..self.input_chain[i].len()
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