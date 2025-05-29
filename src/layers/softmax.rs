use ndarray::{Array2, ArrayD, Ix2};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Softmax
{
    // where input is shape (1, length)
    pub input_tensor: Array2<f32>,
    pub activated_tensor: Array2<f32>,
    pub input_e_tensor: Array2<f32>,
}

impl Softmax
{
    pub fn new() -> Self
    {
        return Self
        {
            input_tensor: Array2::zeros((0, 0)),
            activated_tensor: Array2::zeros((0, 0)),
            input_e_tensor: Array2::zeros((0, 0)),
        }
    }

    pub fn forward(&mut self, input_tensor: ArrayD<f32>) -> ArrayD<f32>
    {
        let input_tensor: Array2<f32> = 
            input_tensor.into_dimensionality::<Ix2>().unwrap();
        let shape: usize = input_tensor.shape()[1];

        if self.input_tensor.shape() == &[0]
        {
            self.input_tensor = Array2::zeros((1, shape));
            self.activated_tensor = Array2::zeros((1, shape));
            self.input_e_tensor = Array2::zeros((1, shape));
        }

        self.input_tensor = input_tensor;

        // x - max to ensure numerical stability
        // gradient respect to x is 1.0, thus no backpropagation changes needed
        let (_, max) = __get_vector_max_and_min(&self.input_tensor);
        self.input_tensor -= max;
        self.input_e_tensor = self.input_tensor.mapv(|v: f32| v.exp() as f32);
        self.activated_tensor = &self.input_e_tensor / self.input_e_tensor.sum();

        //println!("input: {:?}", self.input_tensor);
        //println!("input e: {:?}", self.input_e_tensor);
        let activated_tensor: Array2<f32> = self.activated_tensor.mapv(|v: f32| v as f32);

        return activated_tensor.clone().into_dyn();
    }

    pub fn backward(&mut self, loss_r_softmax: ArrayD<f32>) -> ArrayD<f32>
    {
        let n_classes: usize = self.input_tensor.len();
        let mut softmax_r_inputs: Array2<f32> = Array2::zeros((1, n_classes));

        for i in 0..n_classes
        {
            let mut input_e_tensor_clone: Array2<f32> = self.input_e_tensor.clone();
            let current_exp: f32 = input_e_tensor_clone[(0, i)];

            input_e_tensor_clone[(0, i)] = 0.0;
            let numerator: f32 = (current_exp * input_e_tensor_clone).sum();

            let softmax_r_i: f32 = (numerator / (self.input_e_tensor.sum()).powf(2.0)) as f32;
            softmax_r_inputs[(0, i)] = softmax_r_i;
        }

        let loss_r_inputs: ArrayD<f32> = &loss_r_softmax * &softmax_r_inputs;
        return loss_r_inputs;

    }

    pub fn details(&self)
    {
        println!("Layer type: SOFTMAX");
        println!("=======================================");
    }
}

fn __get_vector_max_and_min(vector: &Array2<f32>,) -> (f32, f32)
{
    let mut max_val: f32 = -1.0e+20;
    //let mut max_idx: usize = 0;
    let mut min_val: f32 = 1.0e+20;
    //let mut min_idx: usize = 0;
    for i in 0..vector.shape()[1]
    {
        if vector[(0, i)] > max_val
        {
            //max_idx = i;
            max_val = vector[(0, i)];
        }

        if vector[(0, i)] < min_val
        {
            //min_idx = i;
            min_val = vector[(0, i)];
        }
    }

    return (min_val, max_val);
}