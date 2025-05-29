use ndarray::{ArrayD, IxDyn};

pub struct Rounding
{
    input_tensor: ArrayD<f64>,
    nearest: f64
}
impl Rounding 
{
    pub fn new(nearest: f64) -> Self
    {
        return Self
        {
            input_tensor: ArrayD::zeros(IxDyn(&[0])),
            nearest
        };
    }

    pub fn forward(input_tensor: ArrayD<f64>)
    {
        
    }
}