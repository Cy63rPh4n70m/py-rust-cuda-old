use ndarray::ArrayD;

pub type LossFn = fn(ArrayD<f32>, ArrayD<f32>) -> f32;
pub type LossFnDeriv = fn(ArrayD<f32>, ArrayD<f32>) -> ArrayD<f32>;
