use ndarray::ArrayD;
use serde::{Deserialize, Serialize};

use super::{
    activation::Activation, adaptive::{AdaptiveReLU, AdaptiveSwish, AdaptiveTanh}, attention::SelfAttention, conv2d::Conv2d, dense::Dense, dropout::Dropout, flatten::Flatten, layer_norm::NormLayer, rand_choice1d::RandChoice1D, recurrent::RecurrentDense, reward_layer::RewardLayer, softmax::Softmax, temporal_dense::TemporalDense};

#[derive(Serialize, Deserialize)]
pub enum Layer
{
    DENSE(Dense),
    CONV2D(Conv2d),
    ACTIVATION(Activation),
    FLATTEN(Flatten),
    DROPOUT(Dropout),
    LAYER_NORM(NormLayer),
    RAND_CHOICE1D(RandChoice1D),
    SOFTMAX(Softmax),
    ADAPTIVE_SWISH(AdaptiveSwish),
    ADAPTIVE_RELU(AdaptiveReLU),
    ADAPTIVE_TANH(AdaptiveTanh),
    REWARD_LAYER(RewardLayer),
    RECURRENT(RecurrentDense),
    TEMPORAL_DENSE(TemporalDense),
    SELF_ATTENTION(SelfAttention),
}
impl Layer
{
    pub fn forward(&mut self, input_tensor: ArrayD<f32>, apply_dropout: bool) -> ArrayD<f32>
    {
        let output_tensor: ArrayD<f32>;
        match self 
        {
            Layer::DENSE(layer) =>
            {                   
                output_tensor = layer.forward(input_tensor);
            }
            Layer::CONV2D(layer) => 
            {
                output_tensor = layer.forward(input_tensor);
            }
            Layer::ACTIVATION(layer) => 
            {
                output_tensor = layer.forward(input_tensor);
            }
            Layer::FLATTEN(layer) =>
            {
                output_tensor = layer.forward(input_tensor);
            }
            Layer::DROPOUT(layer) =>
            {
                output_tensor = layer.forward(input_tensor, apply_dropout);
            }
            Layer::LAYER_NORM(layer) =>
            {
                output_tensor = layer.forward(input_tensor);
            }
            Layer::RAND_CHOICE1D(layer) =>
            {
                output_tensor = layer.forward(input_tensor);
            }
            Layer::SOFTMAX(layer) =>
            {
                output_tensor = layer.forward(input_tensor);
            }
            Layer::ADAPTIVE_SWISH(layer) =>
            {
                output_tensor = layer.forward(input_tensor);
            }
            Layer::ADAPTIVE_RELU(layer) =>
            {
                output_tensor = layer.forward(input_tensor);
            }
            Layer::ADAPTIVE_TANH(layer) =>
            {
                output_tensor = layer.forward(input_tensor);
            }
            Layer::REWARD_LAYER(layer) =>
            {
                output_tensor = layer.forward(input_tensor);
            }
            Layer::RECURRENT(layer) =>
            {
                output_tensor = layer.forward(input_tensor);
            }
            Layer::TEMPORAL_DENSE(layer) =>
            {
                output_tensor = layer.forward(input_tensor, apply_dropout);
            }
            Layer::SELF_ATTENTION(layer) =>
            {
                output_tensor = layer.forward(input_tensor);
            }
        }

        return output_tensor;
    }

    pub fn backward(&mut self, loss_grad_prod: ArrayD<f32>, apply_dropout: bool) -> ArrayD<f32>
    {
        let loss_r_inputs: ArrayD<f32>;
        match self 
        {
            Layer::DENSE(layer) =>
            {
                loss_r_inputs = layer.backward(loss_grad_prod);
            }
            Layer::CONV2D(layer) => 
            {
                loss_r_inputs = layer.backward(loss_grad_prod);
            }
            Layer::ACTIVATION(layer) => 
            {
                loss_r_inputs = layer.backward(loss_grad_prod);
            }
            Layer::FLATTEN(layer) =>
            {
                loss_r_inputs = layer.backward(loss_grad_prod);
            }
            Layer::DROPOUT(layer) =>
            {
                loss_r_inputs = layer.backward(loss_grad_prod, apply_dropout);
            }
            Layer::LAYER_NORM(layer) =>
            {
                loss_r_inputs = layer.backward(loss_grad_prod);
            }
            Layer::RAND_CHOICE1D(layer) =>
            {
                loss_r_inputs = layer.backward(loss_grad_prod);
            }
            Layer::SOFTMAX(layer) =>
            {
                loss_r_inputs = layer.backward(loss_grad_prod);
            }
            Layer::ADAPTIVE_SWISH(layer) =>
            {
                loss_r_inputs = layer.backward(loss_grad_prod);
            }
            Layer::ADAPTIVE_RELU(layer) =>
            {
                loss_r_inputs = layer.backward(loss_grad_prod);
            }
            Layer::ADAPTIVE_TANH(layer) =>
            {
                loss_r_inputs = layer.backward(loss_grad_prod);
            }
            Layer::REWARD_LAYER(layer) =>
            {
                loss_r_inputs = layer.backward(loss_grad_prod);
            }
            Layer::RECURRENT(layer) =>
            {
                loss_r_inputs = layer.backward(loss_grad_prod);
            }
            Layer::TEMPORAL_DENSE(layer) =>
            {
                loss_r_inputs = layer.backward(loss_grad_prod, apply_dropout);
            }
            Layer::SELF_ATTENTION(layer) =>
            {
                loss_r_inputs = layer.backward(loss_grad_prod);
            }
        }

        return loss_r_inputs;
    }

    pub fn update_params(&mut self, lr: f32, min_lr: f32, max_lr: f32)
    {
        match self
        {
            Layer::DENSE(layer) =>
            {
                layer.update_params();
            }
            Layer::CONV2D(layer) => 
            {
                layer.update_params(lr);
            }
            Layer::ACTIVATION(layer) => 
            {
                layer.update_params();
            }
            Layer::FLATTEN(_) =>
            {
                // pass
            }
            Layer::DROPOUT(_) =>
            {
                // pass
            }
            Layer::LAYER_NORM(layer) =>
            {
                layer.update_params();
            }
            Layer::RAND_CHOICE1D(_) =>
            {
                // pass
            }
            Layer::SOFTMAX(_) =>
            {
                // pass
            }
            Layer::ADAPTIVE_SWISH(layer) =>
            {
                layer.update_params(lr);
            }
            Layer::ADAPTIVE_RELU(layer) =>
            {
                layer.update_params(lr);
            }
            Layer::ADAPTIVE_TANH(layer) =>
            {
                layer.update_params(lr);
            }
            Layer::REWARD_LAYER(_) =>
            {
                // pass
            }
            Layer::RECURRENT(layer) =>
            {
                layer.update_params();
            }
            Layer::TEMPORAL_DENSE(layer) =>
            {
                layer.update_params(lr, min_lr, max_lr);
            }
            Layer::SELF_ATTENTION(layer) =>
            {
                layer.update_params();
            }
        }
    }

    pub fn zero_grads(&mut self)
    {
        match self 
        {
            Layer::DENSE(layer) =>
            {
                layer.zero_grads();
            }
            Layer::CONV2D(layer) => 
            {
                layer.zero_grads();
            }
            Layer::ACTIVATION(layer) => 
            {
                layer.zero_grads();
            }
            Layer::FLATTEN(_) =>
            {
                // pass
            }
            Layer::DROPOUT(_) =>
            {
                // pass
            }
            Layer::LAYER_NORM(layer) =>
            {
                layer.zero_grads();
            }
            Layer::RAND_CHOICE1D(_) =>
            {
                // pass
            }
            Layer::SOFTMAX(_) =>
            {
                // pass
            }
            Layer::ADAPTIVE_SWISH(layer) =>
            {
                layer.zero_grads();
            }
            Layer::ADAPTIVE_RELU(layer) =>
            {
                layer.zero_grads();
            }
            Layer::ADAPTIVE_TANH(layer) =>
            {
                layer.zero_grads();
            }
            Layer::REWARD_LAYER(_) =>
            {
                // pass
            }
            Layer::RECURRENT(layer) =>
            {
                layer.zero_grads();
            }
            Layer::TEMPORAL_DENSE(layer) =>
            {
                layer.zero_grads();
            }
            Layer::SELF_ATTENTION(layer) =>
            {
                layer.zero_grads();
            }
        }
    }

    pub fn details(&self)
    {
        match self 
        {
            Layer::DENSE(layer) =>
            {
                layer.details();
            }
            Layer::CONV2D(layer) => 
            {
                layer.details();
            }
            Layer::ACTIVATION(layer) => 
            {
                layer.details();
            }
            Layer::FLATTEN(layer) =>
            {
                layer.details();
            }
            Layer::DROPOUT(layer) =>
            {
                layer.details();
            }
            Layer::LAYER_NORM(layer) =>
            {
                layer.details();
            }
            Layer::RAND_CHOICE1D(_) =>
            {
                // pass
            }
            Layer::SOFTMAX(layer) =>
            {
                layer.details();
            }
            Layer::ADAPTIVE_SWISH(layer) =>
            {
                layer.details();
            }
            Layer::ADAPTIVE_RELU(layer) =>
            {
                layer.details();
            }
            Layer::ADAPTIVE_TANH(layer) =>
            {
                layer.details();
            }
            Layer::REWARD_LAYER(layer) =>
            {
                layer.details();
            }
            Layer::RECURRENT(layer) =>
            {
                layer.details();
            }
            Layer::TEMPORAL_DENSE(layer) =>
            {
                layer.details();
            }
            Layer::SELF_ATTENTION(layer) =>
            {
                layer.details();
            }
        }
    }
}