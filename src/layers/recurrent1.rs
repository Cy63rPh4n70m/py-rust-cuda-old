use std::process::exit;

use ndarray::{s, Array1, Array2, Array3, ArrayD, ArrayView1, ArrayView2, ArrayView3, ArrayViewD, ArrayViewMut2, ArrayViewMut3, Axis, Ix2, Ix3, IxDyn};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::math_functions::{silu, silu_deriv, tanh, tanh_deriv};

#[derive(Serialize, Deserialize)]
pub struct RecurrentDense
{
    // returns only last vector in sequence if false
    pub return_sequence: bool,
    pub in_shape: usize,
    pub hidden_out_shape: usize,
    pub middle_len: usize,
    pub seq_len: usize,
    pub lr: f64,

    // holds the input vector
    pub input_vec_seq: Array2<f64>,

    // stores the inputs/outputs of all layers in each timestep
    pub timestep_hist: Vec<TimeStepState>,

    // history vector
    /*
    pub input_vec_hist: Vec<Array2<f64>>,
    pub input_middle_dot_hist: Vec<Array2<f64>>,
    pub input_middle_dot_silu_hist: Vec<Array2<f64>>,

    pub hidden_middle_dot_hist: Vec<Array2<f64>>,
    pub hidden_middle_dot_silu_hist: Vec<Array2<f64>>,

    pub combined_hist: Vec<Array2<f64>>,
    pub combined_dot_hist: Vec<Array2<f64>>,
    pub combined_dot_silu_hist: Vec<Array2<f64>>,

    pub output_dot_hist: Vec<Array2<f64>>,
    pub output_dot_tanh_hist: Vec<Array2<f64>>,

    pub prev_hidden_state_hist: Vec<Array2<f64>>,
    */

    // layer weight definitions
    pub input_weights2d: Array2<f64>,
    pub input_weights2d_grads: Array2<f64>,
    pub input_biases: Array2<f64>,
    pub input_biases_grads: Array2<f64>,
    
    pub hidden_weights2d: Array2<f64>,
    pub hidden_weights2d_grads: Array2<f64>,
    pub hidden_biases: Array2<f64>,
    pub hidden_biases_grads: Array2<f64>,
    
    // silu activated
    pub combine_weights2d: Array2<f64>,
    pub combine_weights2d_grads: Array2<f64>,
    pub combine_biases: Array2<f64>,
    pub combine_biases_grads: Array2<f64>,
    
    // tanh activated
    pub output_weights2d: Array2<f64>,
    pub output_weights2d_grads: Array2<f64>,
    pub output_biases: Array2<f64>,
    pub output_biases_grads: Array2<f64>,
}

impl RecurrentDense {
    pub fn new(hidden_state_len: usize, middle_len: usize, return_sequence: bool, lr: f64) -> Self
    {
        return Self
        {
            in_shape: 0,
            seq_len: 0,
            hidden_out_shape: hidden_state_len,
            middle_len,
            return_sequence,
            lr,

            input_vec_seq: Array2::zeros((0, 0)),

            timestep_hist: Vec::new(),
            /*
            input_vec_hist: Vec::new(),
            input_middle_dot_hist: Vec::new(),
            input_middle_dot_silu_hist: Vec::new(),
            hidden_middle_dot_hist: Vec::new(),
            hidden_middle_dot_silu_hist: Vec::new(),
            combined_hist: Vec::new(),
            combined_dot_hist: Vec::new(),
            combined_dot_silu_hist: Vec::new(),
            output_dot_hist: Vec::new(),
            output_dot_tanh_hist: Vec::new(),
            prev_hidden_state_hist: Vec::new(),
            */
            
            // initialise arrays
            input_weights2d: Array2::zeros((0, 0)),
            input_weights2d_grads: Array2::zeros((0, 0)),
            input_biases: Array2::zeros((0, 0)),
            input_biases_grads: Array2::zeros((0, 0)),

            hidden_weights2d: Array2::zeros((0, 0)),
            hidden_weights2d_grads: Array2::zeros((0, 0)),
            hidden_biases: Array2::zeros((0, 0)),
            hidden_biases_grads: Array2::zeros((0, 0)),

            combine_weights2d: Array2::zeros((0, 0)),
            combine_weights2d_grads: Array2::zeros((0, 0)),
            combine_biases: Array2::zeros((0, 0)),
            combine_biases_grads: Array2::zeros((0, 0)),

            output_weights2d: Array2::zeros((0, 0)),
            output_weights2d_grads: Array2::zeros((0, 0)),
            output_biases: Array2::zeros((0, 0)),
            output_biases_grads: Array2::zeros((0, 0)),
        }
    }

    pub fn forward(&mut self, input_seq: ArrayD<f64>) -> ArrayD<f64>
    {
        if self.in_shape == 0
        {
            self.in_shape = input_seq.shape()[1];
            self.seq_len = input_seq.shape()[0];
            
            self.input_vec_seq = Array2::zeros((self.seq_len, self.in_shape));

            self.input_weights2d = Array2::zeros((self.in_shape, self.middle_len));
            self.input_weights2d_grads = Array2::zeros((self.in_shape, self.middle_len));
            self.input_biases = Array2::zeros((1, self.middle_len));
            self.input_biases_grads = Array2::zeros((1, self.middle_len));
            
            self.hidden_weights2d = Array2::zeros((self.hidden_out_shape, self.middle_len));
            self.hidden_weights2d_grads = Array2::zeros((self.hidden_out_shape, self.middle_len));
            self.hidden_biases = Array2::zeros((1, self.middle_len));
            self.hidden_biases_grads = Array2::zeros((1, self.middle_len));

            self.combine_weights2d = Array2::zeros((self.middle_len, self.hidden_out_shape));
            self.combine_weights2d_grads = Array2::zeros((self.middle_len, self.hidden_out_shape));
            self.combine_biases = Array2::zeros((1, self.hidden_out_shape));
            self.combine_biases_grads = Array2::zeros((1, self.hidden_out_shape));

            self.output_weights2d = Array2::zeros((self.hidden_out_shape, self.hidden_out_shape));
            self.output_weights2d_grads = Array2::zeros((self.hidden_out_shape, self.hidden_out_shape));
            self.output_biases = Array2::zeros((1, self.hidden_out_shape));
            self.output_biases_grads = Array2::zeros((1, self.hidden_out_shape));
            
            // glorot uniform initialisation
            let input_weights_range: f64 = (6.0 / (self.in_shape + self.middle_len) as f64).sqrt();
            let hidden_weights_range: f64 = (6.0 / (self.hidden_out_shape + self.middle_len) as f64).sqrt();
            let combine_weights_range: f64 = (6.0 / (self.middle_len + self.hidden_out_shape) as f64).sqrt();
            let output_weights_range: f64 = (6.0 / (self.hidden_out_shape + self.hidden_out_shape) as f64).sqrt();

            self.input_weights2d.map_mut(
                |v: &mut f64|
                *v = rand::thread_rng().gen_range(-input_weights_range..input_weights_range)
            );
            self.hidden_weights2d.map_mut(
                |v: &mut f64|
                *v = rand::thread_rng().gen_range(-hidden_weights_range..hidden_weights_range)
            );
            self.combine_weights2d.map_mut(
                |v: &mut f64|
                *v = rand::thread_rng().gen_range(-combine_weights_range..combine_weights_range)
            );
            self.output_weights2d.map_mut(
                |v: &mut f64|
                *v = rand::thread_rng().gen_range(-output_weights_range..output_weights_range)
            );
        }

        self.timestep_hist.clear();

        // convert dynamic to static array
        let input_seq: Array2<f64> = input_seq.into_shape((self.seq_len, self.in_shape)).unwrap();
        
        // output sequence (hidden states)
        let mut output_seq: Array2<f64> = Array2::zeros((self.seq_len, self.hidden_out_shape));
        
        // store input vectors
        self.input_vec_seq.fill(0.0);
        self.input_vec_seq += &input_seq;

        // initialize first hidden state
        let mut current_hidden_state: Array2<f64> = Array2::zeros((1, self.hidden_out_shape));
        
        //println!("{}, {}, {}", self.in_shape, self.hidden_out_shape, self.seq_len);
        // loop through each input in sequence
        for i in 0..self.seq_len
        {
            // get horizontal slice of input
            let input_slice: ArrayView2<f64> = 
                self.input_vec_seq.slice(s![i..i + 1, ..]);
            
            // calculate result from middle
            let input_middle_dot: Array2<f64> = input_slice.dot(&self.input_weights2d) + &self.input_biases;
            let input_middle_dot_silu: Array2<f64> = input_middle_dot.mapv(|v: f64| silu(v));
            let hidden_middle_dot: Array2<f64> = current_hidden_state.dot(&self.hidden_weights2d) + &self.hidden_biases;
            let hidden_middle_dot_silu: Array2<f64> = hidden_middle_dot.mapv(|v: f64| silu(v));

            // calculate result of combined
            let combined: Array2<f64> = &input_middle_dot_silu + &hidden_middle_dot_silu;
            let combined_dot: Array2<f64> = combined.dot(&self.combine_weights2d) + &self.combine_biases;
            let mut combined_dot_silu: Array2<f64> = combined_dot.mapv(|v: f64| silu(v));

            // residual connection
            combined_dot_silu += &current_hidden_state;

            // convert combined to tanh (output)
            let output_dot: Array2<f64> = combined_dot_silu.dot(&self.output_weights2d) + &self.output_biases;
            let output_dot_tanh: Array2<f64> = output_dot.mapv(|v: f64| tanh(v)); // next hidden state
            
            // save state of timestep
            self.timestep_hist.push(
                TimeStepState::new(
                    input_slice.into_owned(), input_middle_dot, input_middle_dot_silu, 
                    hidden_middle_dot, hidden_middle_dot_silu, 
                    combined, combined_dot, combined_dot_silu, 
                    output_dot, output_dot_tanh.clone(), 
                    current_hidden_state.clone()
                )
            );

            // add new hidden state to output sequence
            let mut output_seq_slice: ArrayViewMut2<f64> = output_seq.slice_mut(s![i..i + 1, ..]);
            output_seq_slice += &output_dot_tanh;

            // update current hidden state for next input in sequence
            current_hidden_state = output_dot_tanh;
        }

        if self.return_sequence
        {
            return output_seq.into_dyn()
        }
        else
        {
            let output_seq_last_slice: ArrayView2<f64> = 
                output_seq.slice(
                    s![self.seq_len - 1..self.seq_len, ..]
                );
                                    
            return output_seq_last_slice.into_owned().into_dyn();
        }
    }

    pub fn backward(&mut self, loss_r_outputs: ArrayD<f64>) -> ArrayD<f64>
    {
        // add extra dimension if gradients is one dimensional only last of sequence)
        let seq_len: usize = loss_r_outputs.shape()[0];
        let input_len: usize = loss_r_outputs.shape()[1];
        let mut main_grads: Array2<f64>;
        if self.return_sequence
        {
            main_grads = loss_r_outputs.slice(s![seq_len - 1..seq_len, ..]).into_owned();
        }
        else 
        { 
            //println!("{:?}", length);
            //println!("{:?}", loss_r_outputs);
            main_grads = loss_r_outputs.clone().into_shape((1, input_len)).unwrap();
        }
        //println!("{:?}", main_grads);
        
        // matrix to return from recurrent layer
        let cols: usize = self.input_vec_seq.shape()[0];
        let rows: usize = self.input_vec_seq.shape()[1];
        let mut loss_r_inputs: Array2<f64> = Array2::zeros((cols, rows));
        //let mut main_grads: Array2<f64> = loss_r_outputs.clone();

        //println!("{:?}", self.timestep_hist.len());
        // iterate each timestep in reverse
        
        let last_idx: usize = self.timestep_hist.len() - 1;
        for i in (0..self.timestep_hist.len()).rev()
        {
            // obtain timestep and pop
            let timestep: TimeStepState = self.timestep_hist.pop().unwrap();

            if self.return_sequence && i < last_idx
            {
                let loss_r_output_slice: ArrayView2<f64> = loss_r_outputs.slice(s![i..i + 1, ..]);
                main_grads += &loss_r_output_slice;
            }

            // backpropagate through the output layer
            main_grads = backward_through_layer(
                main_grads,  
                tanh_deriv, &timestep.output_dot, 
                &timestep.combined_dot_silu, &self.output_weights2d,
                &mut self.output_weights2d_grads, &mut self.output_biases_grads
            );

            let residual_grads: Array2<f64> = main_grads.clone();
            //println!("1");
            
            // backprop through combine layer
            main_grads = backward_through_layer(
                main_grads, silu_deriv, &timestep.combined_dot, 
                &timestep.combined, &self.combine_weights2d, 
                &mut self.combine_weights2d_grads, &mut self.combine_biases_grads
            );
            //println!("2");

            // create copy of main_grads for input section
            let main_grads_input: Array2<f64> = main_grads.clone();

            // backpropagate through input section
            let input_grads: Array2<f64> = backward_through_layer(
                main_grads_input, silu_deriv, &timestep.input_middle_dot, 
                &timestep.input_vec, &self.input_weights2d, 
                &mut self.input_weights2d_grads, &mut self.input_biases_grads
            );
            //println!("3");

            // backpropagate through hidden state section
            main_grads = backward_through_layer(
                main_grads, silu_deriv, &timestep.hidden_middle_dot, 
                &timestep.prev_hidden_state, &self.hidden_weights2d, 
                &mut self.hidden_weights2d_grads, &mut self.hidden_biases_grads
            );

            // add residual
            main_grads += &residual_grads;

            //println!("{:?}", loss_r_inputs);
            // write input grads to the loss respect to inputs row
            //println!("loop: {}", i);
            let mut loss_r_inputs_slice: ArrayViewMut2<f64> = loss_r_inputs.slice_mut(s![i..i + 1, ..]);
            //println!("{:?}", loss_r_inputs_slice);
            loss_r_inputs_slice += &input_grads;
            //println!("5");
            //std::process::exit(1);
        }
        //println!("{:?}", loss_r_inputs);
        return loss_r_inputs.into_dyn();
    }

    pub fn update_params(&mut self)
    {
        // ADD REGULARIZATION
        /**/
        self.input_weights2d -= &(self.lr * (&self.input_weights2d_grads + 0.0001 * &self.input_weights2d));
        self.hidden_weights2d -= &(self.lr * (&self.hidden_weights2d_grads + 0.0001 * &self.hidden_weights2d));
        self.combine_weights2d -= &(self.lr * (&self.combine_weights2d_grads + 0.0001 * &self.combine_weights2d));
        self.output_weights2d -= &(self.lr * (&self.output_weights2d_grads + 0.0001 * &self.output_weights2d));
        
        self.input_biases -= &(self.lr * &self.input_biases_grads);
        self.hidden_biases -= &(self.lr * &self.hidden_biases_grads);
        self.combine_biases -= &(self.lr * &self.combine_biases_grads);
        self.output_biases -= &(self.lr * &self.output_biases_grads);
        /**/
    }

    pub fn zero_grads(&mut self)
    {
        self.input_weights2d_grads.fill(0.0);
        self.hidden_weights2d_grads.fill(0.0);
        self.combine_weights2d_grads.fill(0.0);
        self.output_weights2d_grads.fill(0.0);

        self.input_biases_grads.fill(0.0);
        self.hidden_biases_grads.fill(0.0);
        self.combine_biases_grads.fill(0.0);
        self.output_biases_grads.fill(0.0);
        //self.timestep_hist.clear();

        /*
        self.input_vec_hist.clear();
        self.input_middle_dot_hist.clear();
        self.input_middle_dot_silu_hist.clear();
        self.hidden_middle_dot_hist.clear();
        self.hidden_middle_dot_silu_hist.clear();
        self.combined_hist.clear();
        self.combined_dot_hist.clear();
        self.combined_dot_silu_hist.clear();
        self.output_dot_hist.clear();
        self.output_dot_tanh_hist.clear();
        self.prev_hidden_state_hist.clear();
        */

    }

    pub fn details(&self)
    {
        println!("Input weights:");
        println!("{:?}", self.input_weights2d);
        println!("------------------------------------");
        println!("Hidden weights:");
        println!("{:?}", self.hidden_weights2d);
        println!("------------------------------------");
    }
}

// a layer consists of a single weight2d followed by activation
fn backward_through_layer(
    mut main_grads: Array2<f64>, 
    act_deriv_fn: fn(f64) -> f64, input_dot: &Array2<f64>, // for activation layer
    weight_input: &Array2<f64>, weights2d: &Array2<f64>, // for weight layer
    weights2d_grads: &mut Array2<f64>, biases_grads: &mut Array2<f64>, // to update
) -> Array2<f64>
{
    //////////////////////////////////////////////////////////////
    // backprop through output activation
    // a = activation(x_dot)
    // calculate da/dx_dot = activation_derivative(x_dot)
    let act_grads: Array2<f64> = backward_for_act(
        act_deriv_fn, 
        input_dot);

    // chain rule
    main_grads = main_grads * act_grads;
            
    //////////////////////////////////////////////////////////
    // backprop through weights
    // x_dot = x * w
    // dx_dot/dw
    let weight_grads: Array2<f64> = backward_for_weights(&main_grads, weight_input);
    
    // update the gradients for layer
    *weights2d_grads += &(&main_grads * &weight_grads);
    *biases_grads += &main_grads; // bias derivative is 1.0
            
    // dx_dot/dx
    let input_grads: Array2<f64> = 
        backward_for_inputs(weights2d);
    
    // chain rule
    main_grads = main_grads.dot(&input_grads);

    return main_grads;
}

// y = w * x
// dy/dw = x
fn backward_for_weights(
    grads: &Array2<f64>, input: &Array2<f64>) -> Array2<f64>
{
    let new_cols: usize = input.shape()[1];
    let new_rows: usize = grads.shape()[1];

    // transpose and broadcast input vector
    let input_t: ArrayView2<f64> = input.t();
    let input_t_broadcast: ArrayView2<f64> = input_t.broadcast((new_cols, new_rows)).unwrap();

    // broadcast multiply grads with broacasted inputs
    return input_t_broadcast.into_owned();
}

// y = w * x
// dy/dx = w
fn backward_for_inputs(
    weights: &Array2<f64>
) -> Array2<f64>
{
    return weights.t().into_owned();
}

fn backward_for_act(
    act_deriv_fn: fn(f64) -> f64, input_dot: &Array2<f64>) -> Array2<f64>
{
    let act_r_unact: Array2<f64> = input_dot.mapv(|v: f64| act_deriv_fn(v));
    // chain rule
    return act_r_unact;
}

#[derive(Serialize, Deserialize)]
pub struct TimeStepState
{
    input_vec: Array2<f64>,
    input_middle_dot: Array2<f64>,
    input_middle_dot_silu: Array2<f64>,
    hidden_middle_dot: Array2<f64>,
    hidden_middle_dot_silu: Array2<f64>,
    combined: Array2<f64>,
    combined_dot: Array2<f64>,
    combined_dot_silu: Array2<f64>,
    output_dot: Array2<f64>,
    output_dot_tanh: Array2<f64>,
    prev_hidden_state: Array2<f64>,
}
impl TimeStepState
{
    fn new(
        input_vec: Array2<f64>, input_middle_dot: Array2<f64>, input_middle_dot_silu: Array2<f64>,
        hidden_middle_dot: Array2<f64>, hidden_middle_dot_silu: Array2<f64>,
        combined: Array2<f64>, combined_dot: Array2<f64>, combined_dot_silu: Array2<f64>,
        output_dot: Array2<f64>, output_dot_tanh: Array2<f64>, prev_hidden_state: Array2<f64>
    ) -> Self
    {
        return Self
        {
            input_vec, input_middle_dot, input_middle_dot_silu,
            hidden_middle_dot, hidden_middle_dot_silu,
            combined, combined_dot, combined_dot_silu,
            output_dot, output_dot_tanh, prev_hidden_state,
        };
    }
}