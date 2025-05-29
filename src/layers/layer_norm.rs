use ndarray::{ArrayD, Axis, IxDyn};
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct NormLayer
{
    pub input_tensor: ArrayD<f32>,
    pub mean_tensor: ArrayD<f32>,
    pub input_sub_mean: ArrayD<f32>,
    pub input_sub_mean_sqr: ArrayD<f32>,
    pub input_sub_mean_sqr_sum: ArrayD<f32>,
    pub var_tensor: ArrayD<f32>,
    pub z_score_denom: ArrayD<f32>,
    pub normed_tensor: ArrayD<f32>,
    pub normed_ab_tensor: ArrayD<f32>,

    pub alpha_tensor: ArrayD<f32>,
    pub alpha_grads: ArrayD<f32>,
    pub beta_tensor: ArrayD<f32>,
    pub beta_grads: ArrayD<f32>,
    
    pub epsilon: f32,
    pub lr: f32,
    pub axis: usize,
    pub length: f32,
    // e.g. for 2d matrix, axis 0 is layer norm, axis 1 is batch norm
    // where (seq_len, features)
}
impl NormLayer
{
    pub fn new(lr: f32, axis: usize) -> Self
    {
        return Self
        {
            input_tensor: ArrayD::zeros(IxDyn(&[0])),
            mean_tensor: ArrayD::zeros(IxDyn(&[0])),
            input_sub_mean: ArrayD::zeros(IxDyn(&[0])),
            input_sub_mean_sqr: ArrayD::zeros(IxDyn(&[0])),
            input_sub_mean_sqr_sum: ArrayD::zeros(IxDyn(&[0])),
            var_tensor: ArrayD::zeros(IxDyn(&[0])),
            z_score_denom: ArrayD::zeros(IxDyn(&[0])),
            normed_tensor: ArrayD::zeros(IxDyn(&[0])),
            normed_ab_tensor: ArrayD::zeros(IxDyn(&[0])),

            alpha_tensor: ArrayD::zeros(IxDyn(&[0])),
            alpha_grads: ArrayD::zeros(IxDyn(&[0])),
            beta_tensor: ArrayD::zeros(IxDyn(&[0])),
            beta_grads: ArrayD::zeros(IxDyn(&[0])),

            axis,
            epsilon: 1.0e-5,
            lr,
            length: 0.0
        }
    }

    pub fn forward(&mut self, input_tensor: ArrayD<f32>) -> ArrayD<f32>
    {
        if self.input_tensor.shape() == &[0]
        {
            let input_shape: &[usize] = input_tensor.shape();
            let mut shape_prod: f32 = 1.0;
            for dim in input_shape
            {
                shape_prod *= *dim as f32;
            }

            self.alpha_tensor = ArrayD::ones(IxDyn(input_shape));
            self.alpha_grads = ArrayD::zeros(IxDyn(input_shape));
            self.beta_tensor = ArrayD::zeros(IxDyn(input_shape));
            self.beta_grads = ArrayD::zeros(IxDyn(input_shape));
            
            self.alpha_tensor.map_mut(
                |v: &mut f32|
                *v = rand::thread_rng().gen_range(-1.0..1.0)
            );
            self.beta_tensor.map_mut(
                |v: &mut f32|
                *v = rand::thread_rng().gen_range(-1.0..1.0)
            );
        }

        self.length = input_tensor.shape()[self.axis] as f32;
        self.mean_tensor = input_tensor.mean_axis(Axis(self.axis)).unwrap().insert_axis(Axis(self.axis));
        self.input_sub_mean = &input_tensor - &self.mean_tensor;
        self.input_sub_mean_sqr = &self.input_sub_mean * &self.input_sub_mean;
        self.input_sub_mean_sqr_sum = self.input_sub_mean_sqr.sum_axis(Axis(self.axis)).insert_axis(Axis(self.axis));
        self.var_tensor = &self.input_sub_mean_sqr_sum / self.length;
        self.z_score_denom = (&self.var_tensor + self.epsilon).mapv(|v: f32| v.sqrt());
        self.normed_tensor = &self.input_sub_mean / &self.z_score_denom;
        self.normed_ab_tensor = &self.alpha_tensor * &self.normed_tensor + &self.beta_tensor;
        self.input_tensor = input_tensor;

        return self.normed_ab_tensor.clone();

    }

    pub fn backward(&mut self, mut grads: ArrayD<f32>) -> ArrayD<f32>
    {
        // backpropagate through output = alpha * l_norm + beta
        // dlnorm/dscaled = alpha
        // dlnorm/dalpha = scaled
        // dlnorm/dbeta = 1.0
        let output_r_lnorm: ArrayD<f32> = self.alpha_tensor.clone();
        let output_r_alpha: ArrayD<f32> = self.normed_tensor.clone();
        let output_r_beta: f32 = 1.0;

        // update normalization layer parameters
        self.alpha_grads += &(&grads * &output_r_alpha);
        self.beta_grads += &(&grads * output_r_beta);

        grads = grads * output_r_lnorm; // chain rule

        // backpropagate through l_norm = x / denom
        // calculate dl_norm/x = 1 / denom
        // calculate dl_norm/denom = x / denom^2
        let l_norm_r_input_sub_mean: ArrayD<f32> = 1.0 / &self.z_score_denom;
        let l_norm_r_denom: ArrayD<f32> = -(&self.input_sub_mean / (&self.z_score_denom * &self.z_score_denom));

        let mut grads1: ArrayD<f32> = &grads * l_norm_r_input_sub_mean; // chain rule (grads1)
        let mut grads2: ArrayD<f32> = grads * l_norm_r_denom; // chain rule (grads2)

        // uses grads1
        {
            // backpropagate through x = inputs - mean
            // calculate dx/dinputs = 1.0;
            //let x_r_inputs: f32 = 1.0;
            //grads1 = grads1 * x_r_inputs;
        }

        // uses grads2
        {
            // backpropagate through denom = sqrt(var + epsilon)
            // calculate ddenom/dvar = 1/(2 * sqrt(var + epsilon))
            let denom_r_var: ArrayD<f32> = 1.0 / (2.0 * (&self.var_tensor + self.epsilon).mapv(|v: f32| v.sqrt()));
            grads2 = grads2 * denom_r_var; // chain rule

            // backpropagate through var = input_sqr_sum / length
            // calculate dvar/dinput_sqr_sum = 1 / length
            let var_r_input_sqr_sum: f32 = 1.0 / self.length;
            grads2 = grads2 * var_r_input_sqr_sum; // chain rule

            // backpropagate through input_sqr_sum = sum(input_sqr)
            // calculate dinput_sqr_sum/dinput_sqr = 1.0 (broadcast gradients)

            // backpropagate through input_sqr = inputs_sqr ** 2
            // calculate dinput_sqr/dinputs = 2 * inputs_sqr
            let input_sqr_r_x: ArrayD<f32> = 2.0 * &self.input_sub_mean;
            grads2 = grads2 * input_sqr_r_x;

            // backpropagate through x = inputs - mean
            // calculate dx/dinputs = 1.0;
            //let x_r_inputs: f32 = 1.0;
            //grads2 = grads2 * x_r_inputs;
        }

        // sum grads1 and grads 2 for derivative of x
        grads = grads1 + grads2;

        // backpropagate x = inputs - mean
        // calculate dx/dinputs = 1.0;
        grads1 = grads.clone();
        // calculate dx/mean = -1.0;
        grads2 = -grads.clone();

        // backpropagate mean = inputs / length
        // calculate dmean/dinput = 1 / length
        let mean_r_inputs: f32 = 1.0 / self.length;
        grads2 = grads2 * mean_r_inputs;

        grads = grads1 + grads2;

        return grads;
    }

    pub fn update_params(&mut self)
    {
        self.alpha_tensor -= &(self.lr * &self.alpha_grads);
        self.beta_tensor -= &(self.lr * &self.beta_grads);
    }

    pub fn zero_grads(&mut self)
    {
        self.alpha_grads.fill(0.0);
        self.beta_grads.fill(0.0);
    }

    pub fn details(&self)
    {
        println!("Layer type: LAYER_NORM");
        println!("Input shape: {:?}", self.input_tensor.shape());
        println!("Alphas: {:?}", self.alpha_tensor);
        println!("--------------------------------------");
        println!("Betas: {:?}", self.beta_tensor);
    }

}