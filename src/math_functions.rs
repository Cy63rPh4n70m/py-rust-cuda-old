use ndarray::ArrayD;
use rand::Rng;

use crate::types::*;

// loss functions
pub fn squared_error(pred: ArrayD<f32>, target: ArrayD<f32>) -> f32
{
    let loss_ave: f32 = (pred - target).mapv(|x: f32| x * x).sum();
    return loss_ave;
}

pub fn squared_error_deriv(pred: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32>
{
    return 2.0 * (pred - target);
}


pub fn abs_error(pred: Vec<f32>, target: Vec<f32>) -> f32
{
    let mut total: f32 = 0.0;
    for i in 0..pred.len()
    {
        total += (pred[i] - target[i]).abs()
    }
    return total;
}

pub fn abs_error_deriv(pred: f32, target: f32) -> f32
{
    if pred > target
    {
        return 1.0;
    }
    else if pred < target
    {
        return -1.0;
    }
    else
    {
        return 0.0;
    }
}


pub fn log_loss_error(pred: ArrayD<f32>, target: ArrayD<f32>) -> f32
{
    // epsilon added to prevent log(0)
    let epsilon: f32 = 1.0e-8;
    // log(p)
    let mut segment1: ArrayD<f32> = (&pred + epsilon).mapv(|v: f32| v.ln());
    // y * log(p)
    segment1 = &segment1 * &target;
    // (1 - p) (abs called to prevent 1.0 - 1.0 + epsilon, results in
    // negative value)
    let mut segment2: ArrayD<f32> = (1.0 - &pred + epsilon).mapv(|v: f32| v.abs());
    // log(1 - p)
    segment2 = segment2.mapv(|v: f32| v.ln());
    // (1 - y) * log(1 - p)
    segment2 = &segment2 * (1.0 - &target);
    // -(y * log(p) + (1 - y) * log(1 - p))
    let segment3: ArrayD<f32> = -(&segment1 + &segment2);
    // sum of loss values
    let loss_val: f32 = segment3.sum();

    return loss_val;
}

pub fn log_loss_deriv(pred: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32>
{
    //let shape: &[usize] = pred.shape();
    let epsilon: f32 = 1.0e-8;
    let gradient_tensor: ArrayD<f32> = (&pred - &target) / (&pred * (1.0 - &pred) + epsilon);
    //let gradient_tensor: ArrayD<f32> = pred - target;
    //let gradient_mean: f32 = gradient_tensor.mean().unwrap();

    return gradient_tensor;
}

pub fn cat_cross_entropy(pred: ArrayD<f32>, target: ArrayD<f32>) -> f32
{
    let pred_ln: ArrayD<f32> = pred.mapv(|v: f32| (v + 1e-8).ln());
    let loss_values: ArrayD<f32> = - target * pred_ln;

    return loss_values.sum();
}

pub fn cat_cross_entropy_deriv(pred: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32>
{
    // already calculates cross entropy respect to logits
    let gradient_tensor: ArrayD<f32> = pred - target;
    return gradient_tensor;
}

pub fn log_cosh_loss(pred: ArrayD<f32>, target: ArrayD<f32>) -> f32
{
    let total: f32 = (pred - target).mapv(|v: f32| (v.cosh()).ln()).sum();
    return total;
}

pub fn log_cosh_loss_deriv(pred: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32>
{
    return (pred - target).mapv(|v: f32| v.tanh());
}

// activation functions

// sigmoid
pub fn sigmoid(val: f32) -> f32
{
    return (1.0 + val.tanh()) / 2.0;
}

pub fn sigmoid_deriv(val: f32) -> f32
{
    return (1.0 - val.tanh().powf(2.0)) / 2.0;
}

// relu
pub fn relu(val: f32) -> f32
{
    if val > 0.0
    {
        return val;
    }
    else
    {
        return 0.0;
    }
}

pub fn relu_deriv(val: f32) -> f32
{
    if val > 0.0
    {
        return 1.0;
    }
    else
    {
        return 0.0;
    }
}

// tanh
pub fn tanh(val: f32) -> f32
{
    return val.tanh();
}

pub fn tanh_deriv(val: f32) -> f32
{
    return 1.0 - tanh(val).powi(2);
}

// silu
pub fn silu(val: f32) -> f32
{
    return val * sigmoid(val)
}

pub fn silu_deriv(val: f32) -> f32
{
    return ((1.0 + val.tanh()) + val * (1.0 - val.tanh().powf(2.0))) / 2.0;
    //return sigmoid(val) + val * sigmoid(val) * (1.0 - sigmoid(val));
    //return silu(val) + sigmoid(val) * (1.0 - silu(val));
}

// linear
pub fn linear(val: f32) -> f32
{
    return val;
}

pub fn linear_deriv(_val: f32) -> f32
{
    return 1.0;
}

// parabolic
pub fn parabolic(val: f32) -> f32
{
    return val.powf(2.0);
}

pub fn parabolic_deriv(val: f32) -> f32
{
    return 2.0 * val;
}

pub fn sign(val: f32) -> f32
{
    if val > 0.0
    {
        return 1.0;
    }
    else if val < 0.0
    {
        return -1.0;
    }
    else
    {
        return 0.0;
    }
}

// softplus
pub fn softplus(val: f32) -> f32
{
    return (1.0 + val.exp()).ln();
}

pub fn softplus_deriv(val: f32) -> f32
{
    return sigmoid(val);
}

////////////////////////////////////////////////////////////////////////////

pub fn get_activation_from_str(name: &str) -> Option<ActivationFn>
{
    let activation_fn: Option<ActivationFn>;
    match name 
    {
        "tanh" => activation_fn = Some(tanh),
        "silu" => activation_fn = Some(silu),
        "relu" => activation_fn = Some(relu),
        "linear" => activation_fn = Some(linear),
        "sigmoid" => activation_fn = Some(sigmoid),
        "softplus" => activation_fn = Some(softplus),
        "parabolic" => activation_fn = Some(parabolic),
        &_ => activation_fn = None
    }

    return activation_fn;
}

pub fn get_activation_deriv_from_str(name: &str) -> Option<ActivationFnDeriv>
{
    let activation_fn_deriv: Option<ActivationFnDeriv>;
    match name 
    {
        "tanh" => activation_fn_deriv = Some(tanh_deriv),
        "silu" => activation_fn_deriv = Some(silu_deriv),
        "relu" => activation_fn_deriv = Some(relu_deriv),
        "linear" => activation_fn_deriv = Some(linear_deriv),
        "sigmoid" => activation_fn_deriv = Some(sigmoid_deriv),
        "softplus" => activation_fn_deriv = Some(softplus_deriv),
        "parabolic" => activation_fn_deriv = Some(parabolic_deriv),
        &_ => activation_fn_deriv = None
    }

    return activation_fn_deriv;
}

pub fn get_loss_from_str(name: &str) -> Option<LossFn>
{
    let loss_fn: Option<LossFn>;
    match name 
    {
        "bce_loss" => loss_fn = Some(log_loss_error),
        "ce_loss" => loss_fn = Some(cat_cross_entropy),
        "mse_loss" => loss_fn = Some(squared_error),
        "log_cosh_loss" => loss_fn = Some(log_cosh_loss),
        &_ => loss_fn = None
    }

    return loss_fn;
}

pub fn get_loss_deriv_from_str(name: &str) -> Option<LossFnDeriv>
{
    let loss_fn_deriv: Option<LossFnDeriv>;
    match name 
    {
        "bce_loss" => loss_fn_deriv = Some(log_loss_deriv),
        "ce_loss" => loss_fn_deriv = Some(cat_cross_entropy_deriv),
        "mse_loss" => loss_fn_deriv = Some(squared_error_deriv),
        "log_cosh_loss" => loss_fn_deriv = Some(log_cosh_loss_deriv),
        &_ => loss_fn_deriv = None
    }

    return loss_fn_deriv;
}


pub fn matmul(vector: &Vec<f32>, weights2d: &Vec<Vec<f32>>) -> Vec<f32>
{
    let n_out: usize = weights2d[0].len();
    let n_in: usize = vector.len();

    let mut output_vec: Vec<f32> = Vec::new();
    for o in 0..n_out
    {
        let mut total: f32 = 0.0;
        for i in 0..n_in
        {
            total += vector[i] * weights2d[i][o];
        }
        output_vec.push(total);
    }

    return output_vec;
}

pub fn random_float_vec(length: usize, lower_range: f32, upper_range: f32) -> Vec<f32>
{
    let mut vector: Vec<f32> = Vec::new();
    let mut rand_gen: rand::prelude::ThreadRng = rand::thread_rng();
    for _ in 0..length
    {
        if lower_range == 0.0 && upper_range == 0.0
        {
            vector.push(0.0);
        }
        else
        {
            vector.push(rand_gen.gen_range(lower_range..=upper_range));
        }
    }

    return vector;
}

// normal distribution equation
pub fn normal_dist(_x: f32, _mean: f32, _std: f32) -> f32
{
    return 1.0
}