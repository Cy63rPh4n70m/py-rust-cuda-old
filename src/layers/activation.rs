use ndarray::{ArrayD, IxDyn};
use serde::{Deserialize, Serialize};

use crate::{math_functions::
    {get_activation_deriv_from_str, get_activation_from_str}, 
    types::{ActivationFn, ActivationFnDeriv}};

#[derive(Serialize, Deserialize)]
pub struct Activation
{
    pub summed_tensor: ArrayD<f32>,
    pub a: ArrayD<f32>,
    pub a_grads: ArrayD<f32>,
    pub b: ArrayD<f32>,
    pub b_grads: ArrayD<f32>,

    pub activation_str: String,
    pub lr: f32
}
impl Activation
{
    pub fn new(activation_str: &str, lr: f32) -> Self
    {
        return Self {
            summed_tensor: ArrayD::zeros(IxDyn(&[0])),
            
            a: ArrayD::zeros(IxDyn(&[0])),
            a_grads: ArrayD::zeros(IxDyn(&[0])),            
            b: ArrayD::zeros(IxDyn(&[0])),
            b_grads: ArrayD::zeros(IxDyn(&[0])),
            
            activation_str: activation_str.to_string(),
            lr
        }
    }

    pub fn forward(&mut self, input_tensor: ArrayD<f32>) -> ArrayD<f32>
    {
        let activation_fn: ActivationFn = 
            get_activation_from_str(&self.activation_str).unwrap();

        if self.summed_tensor.shape() == &[0]
        {
            let input_shape: &[usize] = input_tensor.shape();
            self.summed_tensor = ArrayD::zeros(IxDyn(input_shape));
            self.a = ArrayD::ones(IxDyn(input_shape));
            self.a_grads = ArrayD::zeros(IxDyn(input_shape));
            self.b = ArrayD::zeros(IxDyn(input_shape));
            self.b_grads = ArrayD::zeros(IxDyn(input_shape));
        }

        let input_tensor1: ArrayD<f32> = &input_tensor * &self.a;
        let input_tensor2: ArrayD<f32> = input_tensor1 - &self.b;
        let activated_tensor: ArrayD<f32> = input_tensor2.mapv(|val: f32| activation_fn(val));
        self.summed_tensor = input_tensor;

        return activated_tensor;
    }

    pub fn backward(&mut self, loss_r_activated: ArrayD<f32>) -> ArrayD<f32>
    {
        let activation_fn_deriv: ActivationFnDeriv = 
            get_activation_deriv_from_str(&self.activation_str).unwrap();
        
        // calculate output respect to a
        let output_r_a: ArrayD<f32> = 
            &self.summed_tensor * (&self.summed_tensor * &self.a - &self.b).mapv(|v: f32| activation_fn_deriv(v));
        // calculate output respect to b
        let output_r_b: ArrayD<f32> = 
            -1.0 * (&self.summed_tensor * &self.a - &self.b).mapv(|v: f32| activation_fn_deriv(v));
        // calculate output respect to inputs
        let output_r_inputs: ArrayD<f32> = 
            &self.a * (&self.summed_tensor * &self.a - &self.b).mapv(|v: f32| activation_fn_deriv(v));
        
        // chain rule update alpha gradients
        self.a_grads += &(&output_r_a * &loss_r_activated);
        self.b_grads += &(&output_r_b * &loss_r_activated);

        let to_return: ArrayD<f32> = output_r_inputs * loss_r_activated;
        return to_return;
    }

    pub fn update_params(&mut self)
    {
        /*
        self.a_grads_vec.push(
            self.a_grads.mapv(|v: f32| v.powf(2.0))
        );
        self.b_grads_vec.push(
            self.b_grads.mapv(|v: f32| v.powf(2.0))
        );
        self.a_grads_sum += &self.a_grads.mapv(|v: f32| v.powf(2.0));
        self.b_grads_sum += &self.b_grads.mapv(|v: f32| v.powf(2.0));

        if self.a_grads_vec.len() > 1000
        {
            self.a_grads_sum -= &self.a_grads_vec.remove(0);
            self.b_grads_sum -= &self.b_grads_vec.remove(0);
        }

        let denominator_a: ArrayD<f32> = self.a_grads_sum.mapv(|v: f32| (v + 1e-10).powf(2.0));
        let denominator_b: ArrayD<f32> = self.b_grads_sum.mapv(|v: f32| (v + 1e-10).powf(2.0));
        let adaptive_lr_a: ArrayD<f32> = (lr / denominator_a).mapv(|v: f32| v.min(0.00001));
        let adaptive_lr_b: ArrayD<f32> = (lr / denominator_b).mapv(|v: f32| v.min(0.00001));
        */
        //self.a -= &(self.lr * &self.a_grads);
        //self.b -= &(self.lr * &self.b_grads);
    }
    
    pub fn zero_grads(&mut self)
    {
        self.a_grads *= 0.0;
        self.b_grads *= 0.0;
    }

    pub fn details(&self)
    {
        println!("Layer type: ACTIVATION");
        println!("Activation function: {:?}", self.activation_str);
        println!("Shape: {:?}", self.a.shape());
        println!("Alpha constants: \n{:?}", self.a);
        println!("Beta constants: \n{:?}", self.b);
    }
}