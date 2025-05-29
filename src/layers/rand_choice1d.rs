use ndarray::{Array1, ArrayD};
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct RandChoice1D
{
    pub input_tensor: Array1<f32>,
    pub output_tensor: Array1<f32>,
    pub choice_mask: Array1<f32>
}

impl RandChoice1D
{
    pub fn new() -> Self
    {
        return Self
        {
            input_tensor: Array1::zeros(0),
            output_tensor: Array1::zeros(0),
            choice_mask: Array1::zeros(0)
        };
    }

    pub fn forward(&mut self, input_tensor: ArrayD<f32>) -> ArrayD<f32>
    {
        if self.input_tensor.shape() == &[0]
        {
            let input_shape: usize = input_tensor.shape()[0];
            self.input_tensor = Array1::zeros(input_shape);
            self.output_tensor = Array1::zeros(input_shape);
            self.choice_mask = Array1::zeros(input_shape);
        }

        self.input_tensor.fill(0.0);
        self.output_tensor.fill(0.0);
        self.choice_mask.fill(0.0); // prevent division by zero

        self.input_tensor += &input_tensor;

        println!("input: {:?}", input_tensor);

        // set choice mask based on cumulative probability
        self.set_choice_mask();

        println!("mask: {:?}", self.choice_mask);

        // divide input tensor by mask to create one hot selection
        self.output_tensor = &self.input_tensor * &self.choice_mask;

        println!("output: {:?}", self.output_tensor);
        println!("-------------------------------------");

        return self.output_tensor.clone().into_dyn();
    }

    pub fn backward(&mut self, loss_r_one_hot: ArrayD<f32>) -> ArrayD<f32>
    {
        // calculate one hot respect to input
        // y = mask * x
        // dy/dx = mask
        
        /*
        println!("mask: {:?}", self.choice_mask);
        println!("input: {:?}", self.input_tensor);
        println!("output: {:?}", self.output_tensor);
        println!("-------------------------------------");
        */

        // chain rule
        let loss_r_inputs: ArrayD<f32> = 
            &self.choice_mask * &loss_r_one_hot;
        
        //println!("{:?}", loss_r_inputs);
        
        return loss_r_inputs;
    }

    fn set_choice_mask(&mut self)
    {
        let mut total: f32 = 0.0;
        let mut cumul_dist: Vec<f32> = vec![0.0];
        for i in 0..self.input_tensor.shape()[0]
        {
            total += self.input_tensor[[i]];
            cumul_dist.push(total);
        }

        //println!("{:?}", cumul_dist);
        let random_val: f32 = rand::thread_rng().gen_range(0.0..total);
        let mut chosen_idx: usize = 0;
        let mut chosen_val: f32 = 0.0;
        //let mut chosen: bool = false;
        for i in 1..cumul_dist.len()
        {
            if cumul_dist[i - 1] <= random_val && random_val <= cumul_dist[i]
            {
                chosen_idx = i - 1;
                chosen_val = self.input_tensor[chosen_idx];
        //        chosen = true;
                break;
            }
        }

        self.choice_mask[[chosen_idx]] = 1.0 / chosen_val;


    }
}