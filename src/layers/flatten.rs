use ndarray::{Array2, ArrayD, IxDyn};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Flatten
{
    pub input_tensor: ArrayD<f32>,
    pub flattened_tensor: Array2<f32>,
    pub input_shape: Vec<usize>,
    pub flatten_len: usize
}
impl Flatten
{
    pub fn new() -> Self
    {
        return Self
        {
            input_tensor: ArrayD::zeros(IxDyn(&[0])),
            flattened_tensor: Array2::zeros((0, 0)),
            input_shape: Vec::new(),
            flatten_len: 1
        }
    }

    pub fn forward(&mut self, input_tensor: ArrayD<f32>) -> ArrayD<f32>
    {
        if self.input_tensor.shape() == &[0]
        {
            let input_shape: &[usize] = input_tensor.shape();
            self.input_tensor = ArrayD::zeros(IxDyn(input_shape));
            for val in input_shape
            {
                self.flatten_len *= val;
            }
            //println!("Input shape: {:?}", input_shape);
            //println!("Flattened length: {:?}", self.flatten_len);
            self.flattened_tensor = Array2::zeros((1, self.flatten_len));
            self.input_shape = input_shape.to_vec();
        }

        self.input_tensor.fill(0.0);
        self.flattened_tensor.fill(0.0);

        self.input_tensor += &input_tensor;
        //println!("{:?}", input_tensor);
        let flattened_tensor: Array2<f32> = 
            input_tensor.into_shape((1, self.flatten_len)).unwrap();
        
        //println!("{:?}", flattened_tensor);
        //std::process::exit(1);
        
        self.flattened_tensor += &flattened_tensor;
        //println!("{:?}", self.flattened_tensor);

        return self.flattened_tensor.clone().into_dyn();
    }

    pub fn backward(&mut self, flattened_grads: ArrayD<f32>) -> ArrayD<f32>
    {
        let reshaped_grads: ArrayD<f32> = 
            flattened_grads.into_shape(self.input_shape.clone()).unwrap();

        return reshaped_grads.clone();
    }

    pub fn details(&self)
    {
        println!("Input Shape: {:?}", self.input_shape);
        println!("Flattened length: {}", self.flatten_len);
        println!("=======================================");
    }

}