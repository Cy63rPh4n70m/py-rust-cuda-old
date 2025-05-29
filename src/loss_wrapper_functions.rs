use ndarray::ArrayD;

use crate::{math_functions::{
    get_loss_deriv_from_str, get_loss_from_str}, 
    LossFn, LossFnDeriv};

pub fn loss_fn(loss_fn_str: &str, pred: ArrayD<f64>, actual: ArrayD<f64>) -> f64
{
    let loss_function: LossFn = get_loss_from_str(loss_fn_str).unwrap();
    let loss_val: f64 = loss_function(pred, actual);
    return loss_val;
    
}

pub fn loss_grad_fn(loss_fn_str: &str, pred: ArrayD<f64>, actual: ArrayD<f64>) -> ArrayD<f64>
{
    let loss_function_deriv: LossFnDeriv = get_loss_deriv_from_str(loss_fn_str).unwrap();
    let loss_gradients: ArrayD<f64> = loss_function_deriv(pred, actual);
    return loss_gradients;
}