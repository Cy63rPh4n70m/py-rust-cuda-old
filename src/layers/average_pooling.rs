use ndarray::ArrayD;

pub struct AvePooling
{
    pub input_tensor: ArrayD<f64>,
    pub pooled_tensor: ArrayD<f64>,
    pub input_shape: Vec<f64>,
    pub pool_dim: usize
}