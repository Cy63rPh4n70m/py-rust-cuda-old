
use serde::{Deserialize, Serialize};

use super::{activation_cuda::ActivationCuda, broadcast_cuda::BroadcastCuda, cls_cuda::CLSCuda, conv2d_cuda::Conv2dCuda, dense_cuda::DenseCuda, dropout_cuda::DropoutCuda, elementwise_cuda::ElementwiseCuda, embedding_cuda::Embedding2DCuda, l2norm_cuda::L2NormCuda, softmax_cuda::SoftmaxCuda, sum_cuda::SumCuda, transpose_cuda::BatchTransposeCuda};

#[derive(Serialize, Deserialize)]
pub enum CudaLayer
{
    DENSE_CUDA(DenseCuda),
    ACTIVATION_CUDA(ActivationCuda),
    //SELF_ATTENTION_CUDA(SelfAttentionCuda),
    //NAVIGATION_CUDA(NavigationCuda),
    L2_NORM_CUDA(L2NormCuda),
    //POS_ENCODE_2D_CUDA(PosEncoding2DCuda),
    EMBEDDING_2D_CUDA(Embedding2DCuda),
    CONV_2D_CUDA(Conv2dCuda),
    SOFTMAX_CUDA(SoftmaxCuda),
    DROPOUT_CUDA(DropoutCuda),
    ELEMENTWISE_CUDA(ElementwiseCuda),
    BROADCAST_CUDA(BroadcastCuda),
    SUM_CUDA(SumCuda),
    TRANSPOSE_CUDA(BatchTransposeCuda),
    CLS_CUDA(CLSCuda)
}
impl CudaLayer
{
    pub fn forward(&mut self, str_ptr_in: String, str_ptr_weight: String, apply_dropout: bool) -> String
    {
        let mut new_traverse_ptr: String = String::new();
        match self 
        {
            CudaLayer::DENSE_CUDA(layer) =>
            {
                new_traverse_ptr = layer.forward(str_ptr_in, str_ptr_weight);
            }
            CudaLayer::ACTIVATION_CUDA(layer) =>
            {
                new_traverse_ptr = layer.forward(str_ptr_in);
            }
            //CudaLayer::SELF_ATTENTION_CUDA(layer) =>
            //{
            //    layer.forward(input_ptr);
            //}
            //CudaLayer::NAVIGATION_CUDA(layer) =>
            //{
            //    layer.forward(apply_dropout);
            //}
            CudaLayer::L2_NORM_CUDA(layer) =>
            {
                new_traverse_ptr = layer.forward(str_ptr_in);
            }
            CudaLayer::EMBEDDING_2D_CUDA(layer) =>
            {
                new_traverse_ptr = layer.forward(str_ptr_in);
            }
            CudaLayer::CONV_2D_CUDA(layer) =>
            {
                new_traverse_ptr = layer.forward(str_ptr_in);
            }
            CudaLayer::SOFTMAX_CUDA(layer) =>
            {
                new_traverse_ptr = layer.forward(str_ptr_in);
            }
            CudaLayer::DROPOUT_CUDA(layer) =>
            {
                new_traverse_ptr = layer.forward(str_ptr_in, apply_dropout);
            }
            CudaLayer::ELEMENTWISE_CUDA(layer) =>
            {
                new_traverse_ptr = layer.forward(str_ptr_in, str_ptr_weight, apply_dropout);
            }
            CudaLayer::BROADCAST_CUDA(layer) =>
            {
                new_traverse_ptr = layer.forward(str_ptr_in);
            }
            CudaLayer::SUM_CUDA(layer) =>
            {
                new_traverse_ptr = layer.forward(str_ptr_in);
            }
            CudaLayer::TRANSPOSE_CUDA(layer) =>
            {
                new_traverse_ptr = layer.forward(str_ptr_in);
            }
            CudaLayer::CLS_CUDA(layer) =>
            {
                new_traverse_ptr = layer.forward(str_ptr_in);
            }
        }

        return new_traverse_ptr;
    }

    pub fn backward(&mut self, apply_dropout: bool)
    {
        match self 
        {
            CudaLayer::DENSE_CUDA(layer) =>
            {
                layer.backward();
            }
            CudaLayer::ACTIVATION_CUDA(layer) =>
            {
                layer.backward();
            }
            //CudaLayer::SELF_ATTENTION_CUDA(layer) =>
            //{
            //    layer.backward(gradient_ptr);
            //}
            //CudaLayer::NAVIGATION_CUDA(layer) =>
            //{
            //    layer.backward( apply_dropout);
            //}
            CudaLayer::L2_NORM_CUDA(layer) =>
            {
                layer.backward();
            }
            CudaLayer::EMBEDDING_2D_CUDA(layer) =>
            {
                layer.backward();
            }
            CudaLayer::CONV_2D_CUDA(layer) =>
            {
                layer.backward();
            }
            CudaLayer::SOFTMAX_CUDA(layer) =>
            {
                layer.backward();
            }
            CudaLayer::DROPOUT_CUDA(layer) =>
            {
                layer.backward(apply_dropout);
            }
            CudaLayer::ELEMENTWISE_CUDA(layer) =>
            {
                layer.backward(apply_dropout);
            }
            CudaLayer::BROADCAST_CUDA(layer) =>
            {
                layer.backward();
            }
            CudaLayer::SUM_CUDA(layer) =>
            {
                layer.backward();
            }
            CudaLayer::TRANSPOSE_CUDA(layer) =>
            {
                layer.backward();
            }
            CudaLayer::CLS_CUDA(layer) =>
            {
                layer.backward();
            }
        }
    }

    pub fn update_params(&mut self, optimizer_type: i32, lr: f32, l2: f32, alpha: f32, beta: f32)
    {
        match self
        {
            CudaLayer::DENSE_CUDA(layer) =>
            {
                layer.update_params(optimizer_type, lr, l2, alpha, beta, false);
            }
            CudaLayer::ACTIVATION_CUDA(layer) =>
            {
                layer.update_params();
            }
            //CudaLayer::SELF_ATTENTION_CUDA(layer) =>
            //{
            //    layer.update_params();
            //}
            //CudaLayer::NAVIGATION_CUDA(layer) =>
            //{
            //    layer.update_params();
            //}
            CudaLayer::L2_NORM_CUDA(layer) =>
            {
                layer.update_params();
            }
            CudaLayer::EMBEDDING_2D_CUDA(layer) =>
            {
                layer.update_params(optimizer_type, lr, l2, alpha, beta)
            }
            CudaLayer::CONV_2D_CUDA(layer) =>
            {
                layer.update_params(optimizer_type, lr, l2, alpha, beta, false)
            }
            CudaLayer::SOFTMAX_CUDA(layer) =>
            {
                layer.update_params();
            }
            CudaLayer::DROPOUT_CUDA(layer) =>
            {
                layer.update_params();
            }
            CudaLayer::ELEMENTWISE_CUDA(layer) =>
            {
                layer.update_params(optimizer_type, lr, l2, alpha, beta);
            }
            CudaLayer::BROADCAST_CUDA(layer) =>
            {
                layer.update_params();
            }
            CudaLayer::SUM_CUDA(layer) =>
            {
                layer.update_params();
            }
            CudaLayer::TRANSPOSE_CUDA(layer) =>
            {
                layer.update_params();
            }
            CudaLayer::CLS_CUDA(layer) =>
            {
                layer.update_params();
            }
        }
    }

    pub fn details(&self)
    {
        match self 
        {
            CudaLayer::DENSE_CUDA(layer) =>
            {
                layer.details();
            }
            CudaLayer::ACTIVATION_CUDA(layer) =>
            {
                layer.details();
            }
            //CudaLayer::SELF_ATTENTION_CUDA(layer) =>
            //{
            //    layer.details();
            //}
            //CudaLayer::NAVIGATION_CUDA(layer) =>
            //{
            //    layer.details();
            //}
            CudaLayer::L2_NORM_CUDA(layer) =>
            {
                layer.details();
            }
            CudaLayer::EMBEDDING_2D_CUDA(layer) =>
            {
                layer.details();
            }
            CudaLayer::CONV_2D_CUDA(layer) =>
            {
                layer.details();
            }
            CudaLayer::SOFTMAX_CUDA(layer) =>
            {
                layer.details();
            }
            CudaLayer::DROPOUT_CUDA(layer) =>
            {
                layer.details();
            }
            CudaLayer::ELEMENTWISE_CUDA(layer) =>
            {
                layer.details();
            }
            CudaLayer::BROADCAST_CUDA(layer) =>
            {
                layer.details();
            }
            CudaLayer::SUM_CUDA(layer) =>
            {
                layer.details();
            }
            CudaLayer::TRANSPOSE_CUDA(layer) =>
            {
                layer.details();
            }
            CudaLayer::CLS_CUDA(layer) =>
            {
                layer.details();
            }
        }
    }

    pub fn get_param_count(&self) -> usize
    {
        let param_count: usize;
        match self 
        {
            CudaLayer::DENSE_CUDA(layer) =>
            {
                param_count = layer.get_param_count();
            }
            CudaLayer::ACTIVATION_CUDA(layer) =>
            {
                param_count = layer.get_param_count();
            }
            //CudaLayer::SELF_ATTENTION_CUDA(layer) =>
            //{
            //    param_count = layer.get_param_count();
            //}
            //CudaLayer::NAVIGATION_CUDA(layer) =>
            //{
            //    param_count = layer.get_param_count();
            //}
            CudaLayer::L2_NORM_CUDA(layer) =>
            {
                param_count = layer.get_param_count();
            }
            CudaLayer::EMBEDDING_2D_CUDA(layer) =>
            {
                param_count = layer.get_param_count();
            }
            CudaLayer::CONV_2D_CUDA(layer) =>
            {
                param_count = layer.get_param_count();
            }
            CudaLayer::SOFTMAX_CUDA(layer) =>
            {
                param_count = layer.get_param_count();
            }
            CudaLayer::DROPOUT_CUDA(layer) =>
            {
                param_count = layer.get_param_count();
            }
            CudaLayer::ELEMENTWISE_CUDA(layer) =>
            {
                param_count = layer.get_param_count();
            }
            CudaLayer::BROADCAST_CUDA(layer) =>
            {
                param_count = layer.get_param_count();
            }
            CudaLayer::SUM_CUDA(layer) =>
            {
                param_count = layer.get_param_count();
            }
            CudaLayer::TRANSPOSE_CUDA(layer) =>
            {
                param_count = layer.get_param_count();
            }
            CudaLayer::CLS_CUDA(layer) =>
            {
                param_count = layer.get_param_count();
            }
        }

        return param_count;
    }

    pub fn move_ptrs_to_arrays(&mut self)
    {
        match self 
        {
            CudaLayer::DENSE_CUDA(layer) =>
            {
                layer.move_ptrs_to_arrays(false);
            }
            CudaLayer::ACTIVATION_CUDA(layer) =>
            {
                layer.move_ptrs_to_arrays();
            }
            //CudaLayer::SELF_ATTENTION_CUDA(layer) =>
            //{
            //    layer.move_ptrs_to_arrays();
            //}
            //CudaLayer::NAVIGATION_CUDA(layer) =>
            //{
            //    layer.move_ptrs_to_arrays();
            //}
            CudaLayer::L2_NORM_CUDA(layer) =>
            {
                layer.move_ptrs_to_arrays();
            }
            CudaLayer::EMBEDDING_2D_CUDA(layer) =>
            {
                layer.move_ptrs_to_arrays();
            }
            CudaLayer::CONV_2D_CUDA(layer) =>
            {
                layer.move_ptrs_to_arrays(false);
            }
            CudaLayer::SOFTMAX_CUDA(layer) =>
            {
                layer.move_ptrs_to_arrays();
            }
            CudaLayer::DROPOUT_CUDA(layer) =>
            {
                layer.move_ptrs_to_arrays();
            }
            CudaLayer::ELEMENTWISE_CUDA(layer) =>
            {
                layer.move_ptrs_to_arrays();
            }
            CudaLayer::BROADCAST_CUDA(layer) =>
            {
                layer.move_ptrs_to_arrays();
            }
            CudaLayer::SUM_CUDA(layer) =>
            {
                layer.move_ptrs_to_arrays();
            }
            CudaLayer::TRANSPOSE_CUDA(layer) =>
            {
                layer.move_ptrs_to_arrays();
            }
            CudaLayer::CLS_CUDA(layer) =>
            {
                layer.move_ptrs_to_arrays();
            }
        }
    }

    pub fn set_ptrs_allocated(&mut self)
    {
        match self 
        {
            CudaLayer::DENSE_CUDA(layer) =>
            {
                layer.set_ptrs_allocated();
            }
            CudaLayer::ACTIVATION_CUDA(layer) =>
            {
                layer.set_ptrs_allocated();
            }
            //CudaLayer::SELF_ATTENTION_CUDA(layer) =>
            //{
            //    layer.move_ptrs_to_arrays();
            //}
            //CudaLayer::NAVIGATION_CUDA(layer) =>
            //{
            //    layer.move_ptrs_to_arrays();
            //}
            CudaLayer::L2_NORM_CUDA(layer) =>
            {
                layer.set_ptrs_allocated();
            }
            CudaLayer::EMBEDDING_2D_CUDA(layer) =>
            {
                layer.set_ptrs_allocated();
            }
            CudaLayer::CONV_2D_CUDA(layer) =>
            {
                layer.set_ptrs_allocated();
            }
            CudaLayer::SOFTMAX_CUDA(layer) =>
            {
                layer.set_ptrs_allocated();
            }
            CudaLayer::DROPOUT_CUDA(layer) =>
            {
                layer.set_ptrs_allocated();
            }
            CudaLayer::ELEMENTWISE_CUDA(layer) =>
            {
                layer.set_ptrs_allocated();
            }
            CudaLayer::BROADCAST_CUDA(layer) =>
            {
                layer.set_ptrs_allocated();
            }
            CudaLayer::SUM_CUDA(layer) =>
            {
                layer.set_ptrs_allocated();
            }
            CudaLayer::TRANSPOSE_CUDA(layer) =>
            {
                layer.set_ptrs_allocated();
            }
            CudaLayer::CLS_CUDA(layer) =>
            {
                layer.set_ptrs_allocated();
            }
        }
    }
}