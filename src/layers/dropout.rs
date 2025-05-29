use ndarray::{ArrayD, IxDyn};
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Dropout
{
    pub original_tensor: ArrayD<f32>,
    pub dropout_mask: ArrayD<f32>,
    pub dropped_tensor: ArrayD<f32>,
    pub input_shape: Vec<usize>,
    pub dropout_rate: f32
}
impl Dropout
{
    pub fn new(rate: f32) -> Self
    {
        return Self
        {
            original_tensor: ArrayD::zeros(IxDyn(&[0])),
            dropped_tensor: ArrayD::zeros(IxDyn(&[0])),
            dropout_mask: ArrayD::zeros(IxDyn(&[0])),
            input_shape: Vec::new(),
            dropout_rate: rate
        }
    }

    pub fn forward(&mut self, input_tensor: ArrayD<f32>, apply_dropout: bool) -> ArrayD<f32>
    {
        if self.original_tensor.shape() == &[0]
        {
            let input_shape: &[usize] = input_tensor.shape();
            self.original_tensor = ArrayD::zeros(IxDyn(input_shape));
            self.dropped_tensor = ArrayD::zeros(IxDyn(input_shape));
            self.dropout_mask = ArrayD::zeros(IxDyn(input_shape));
            self.input_shape = input_shape.to_vec();
        }

        self.original_tensor.fill(0.0);
        self.dropped_tensor.fill(0.0);
        self.dropout_mask.fill(0.0);

        self.original_tensor += &input_tensor;
        if apply_dropout
        {
            self.dropout_mask.map_mut(
                |v: &mut f32| 
                *v = init_dropout_mask(self.dropout_rate)
            );

            self.dropped_tensor += &(&self.original_tensor * &self.dropout_mask);
            self.dropped_tensor = &self.dropped_tensor / (1.0 - self.dropout_rate);
        }
        else
        {
            self.dropped_tensor += &input_tensor;
        }

        return self.dropped_tensor.clone().into_dyn();
    }

    pub fn backward(&mut self, loss_r_dropped: ArrayD<f32>, apply_dropout: bool) -> ArrayD<f32>
    {
        let loss_r_activated: ArrayD<f32>;
        if apply_dropout
        {
            let dropped_r_activated: ArrayD<f32> = 
                &self.dropout_mask / (1.0 - self.dropout_rate);
            
            loss_r_activated = loss_r_dropped * dropped_r_activated;
        }
        else
        {
            loss_r_activated = loss_r_dropped;
        }
        return loss_r_activated;
    }

    pub fn details(&self)
    {
        println!("Layer type: DROPOUT");
        println!("Input shape: {:?}", self.input_shape);
        println!("Dropout rate: {}", self.dropout_rate);
        println!("=======================================");
    }

}

pub fn init_dropout_mask(rate: f32) -> f32
{
    let new_val: f32;
    if rand::thread_rng().gen_bool(rate as f64)
    {
        new_val = 0.0;
    }
    else
    {
        new_val = 1.0;
    }

    return new_val;
}