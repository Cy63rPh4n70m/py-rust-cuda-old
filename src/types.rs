use ndarray::ArrayD;

pub type Matrix2D = Vec<Vec<f32>>;
pub type Window2D = Vec<(usize, usize)>;

pub type LossFn = fn(ArrayD<f32>, ArrayD<f32>) -> f32;
pub type LossFnDeriv = fn(ArrayD<f32>, ArrayD<f32>) -> ArrayD<f32>;

pub type ActivationFn = fn(f32) -> f32;
pub type ActivationFnDeriv = fn(f32) -> f32;
