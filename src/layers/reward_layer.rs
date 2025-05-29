use ndarray::{Array1, ArrayD};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct RewardLayer
{
    pub input_tensor: Array1<f32>,
    pub lowest_reward: f32,
    pub highest_reward: f32
}
impl RewardLayer
{
    pub fn new(lowest_reward: f32, highest_reward: f32) -> Self
    {
        return Self
        {
            input_tensor: Array1::zeros(0),
            lowest_reward,
            highest_reward
        }
    }

    pub fn forward(&mut self, input_tensor: ArrayD<f32>) -> ArrayD<f32>
    {
        if self.input_tensor.shape() == &[0]
        {
            let shape: &[usize] = input_tensor.shape();
            self.input_tensor = Array1::zeros(shape[0]);
        }

        self.input_tensor.fill(0.0);
        self.input_tensor += &input_tensor;
        
        let mut output_tensor: Array1<f32> = 
            self.input_tensor.mapv(|v: f32| v.tanh());
        
        output_tensor = 
            ((self.highest_reward - self.lowest_reward) / 2.0) * 
            (output_tensor + 1.0);

        output_tensor += self.lowest_reward;

        return output_tensor.into_dyn();
    }

    pub fn backward(&mut self, loss_r_outputs: ArrayD<f32>) -> ArrayD<f32>
    {
        let output_r_inputs: Array1<f32> = 
            ((self.highest_reward - self.lowest_reward) / 2.0) * 
            (1.0 - self.input_tensor.mapv(|v: f32| (v.tanh()).powf(2.0)));
        
        return loss_r_outputs * output_r_inputs;
    }

    pub fn details(&self)
    {
        println!("Layer type: REWARD_LAYER");
        println!("Lowest reward: {}", self.lowest_reward);
        println!("Highest reward: {}", self.highest_reward);
        println!("=======================================");
    }
}