//! Custom type definitions here are specifically for loss
//! and loss_grads FFI functions, subject to removal in the future
use ndarray::ArrayD;

pub type LossFn = fn(ArrayD<f32>, ArrayD<f32>) -> f32;
pub type LossFnDeriv = fn(ArrayD<f32>, ArrayD<f32>) -> ArrayD<f32>;
