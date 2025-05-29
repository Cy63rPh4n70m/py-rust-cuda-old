
use serde::{Deserialize, Serialize};

use crate::{cuda_bridge::{broadcast_2d_to_3d, copy_cuda_to_cuda, sum_axis, transpose_2d, zeroes_3d_inplace}, neuralnet::CudaTensorPtr, pointer_ops::{new_cuda_ptr_str, ptr_to_string, string_to_ptr}};

use super::{activation_cuda::ActivationCuda, broadcast_cuda::BroadcastMulCuda, dense_cuda::DenseCuda, dropout_cuda::DropoutCuda, elementwise_cuda::ScalingCuda, l2norm_cuda::L2NormCuda, layer_norm_cuda::LayerNormCuda, pos_encoding_cuda::PosEncoding2DCuda};

#[derive(Serialize, Deserialize)]
pub struct NavigationCuda
{
    navigator_layer: DenseCuda, // produces vectors that move through high dim space
    navigator_scaler_layer: ScalingCuda,
    //navigator_pos_layer: PosEncoding2DCuda,
    //navigator_pre_scaler_layer: ScalingCuda,
    //navigator_scaler_layer: BroadcastInvMulCuda,
    //navigator_act_layer: ActivationCuda,
    dropout_layer: DropoutCuda,
    //navigator_output_layer: DenseCuda,
    
    //input_scaler_layer: ScalingCuda,
    input_resizer_layer: DenseCuda,
    //input_resizer_pre_scaler_layer: ScalingCuda,
    input_resizer_act_layer: ActivationCuda,
    //input_resizer_norm_layer: LayerNormCuda,
    latent_broadcast_layer: BroadcastMulCuda,
    //scaler_layer: ScalingCuda,
    //aggregator_layer: DenseCuda, // combines output from sum of navigator and transpose of receptor
    //aggregator_norm_layer: LayerNormCuda,
    //value_norm_layer: LayerNormCuda,
    //value_act_layer: ActivationCuda,
    aggregator_act_layer: ActivationCuda,
    aggregator_resizer_layer: DenseCuda,
    //aggregator_scaler_layer: ScalingCuda,

    //value_layer: DenseCuda, value_layer1: DenseCuda,
    //value_softmax: SoftmaxCuda,

    //latent_norm_layer: LayerNormCuda,
    latent_act_layer: ActivationCuda,
    //output_embedding_norm_layer: LayerNormCuda,
    //output_act_layer: ActivationCuda,

    //input_resizer_bottleneck_layer: DenseCuda,
    //navigator_bottleneck_layer: DenseCuda,
    //navigator_l2norm_layer: L2NormCuda,

    //latent_modifier_layer: DenseCuda,
    //latent_modifier_act_layer: ActivationCuda,
    //latent_modifier_norm_layer: LayerNormCuda,
    //latent_modifier_scaler_layer: ScalingCuda,
    //value_bottleneck_layer: DenseCuda,
    //value_expand_layer: DenseCuda,
    //head_out_dense_layer: DenseCuda,

    //final_act_layer: ActivationCuda,

    //extra_dense_layer: DenseCuda,
    //extra_act_layer: ActivationCuda,
    //extra_dense_layer1: DenseCuda,
    //extra_act_layer1: ActivationCuda,
    //extra_dense_layer2: DenseCuda,
    //extra_norm_layer: LayerNormCuda,

    pub input_embedding_len: usize,
    pub latent_embedding_len: usize,
    pub seq_len: usize,
    pub n_heads: usize,
    pub l2: f32, pub return_sequence: bool,

    pub main_input_ptr: String,
    pub main_input_grads_ptr: String,
    pub main_output_ptr: String,
    pub main_output_grads_ptr: String,

    //pub input_embedding_ptr: String,
    pub input_embedding_broadcasted_ptr: String,
    //pub input_embedding_broadcasted_t_ptr: String,
    pub latent_ptr: String,
    //pub latent_resized_ptr: String,
    pub latent_ptr_t: String,
    //pub latent_expanded_horz_ptr: String,
    //pub value_modifier_ptr: String,
    //pub input_resizer_ptr: String,
    pub output_embedding_ptr: String,
    pub output_embedding_vector_ptr: String,
    //pub residual_input: String,

    // gradient ptrs
    //pub residual_grads: String,
    pub output_grads_broadcasted_ptr: String, // (1, seq_len, output)
    pub output_grads_broadcasted_head_ptr: String, // (heads, seq_len, output)
    //pub grads_input_resizer: String,
    //pub grads_latent: String,
    //pub latent_expanded_grad_ptr: String,
    //pub value_modifier_grad_ptr: String,
    pub latent_grad_ptr: String,
    pub latent_grad_broadcasted_ptr: String,
    //pub val_grads_t_ptr: String,
    //pub return_grads_val: String,
    pub return_grads: String, // (1, seq_len, input)

    pub ptrs_allocated: bool,
    pub sub_layers_allocated: bool,
}
impl NavigationCuda
{
    pub fn new(
        n_heads: usize, 
        input_embedding_len: usize,
        latent_embedding_len: usize, 
        seq_len: usize,
        dropout_rate: f32,
        append_scaler: f32,
        return_sequence: bool, l2: f32, lr: f32,
        ) -> Self
    {
        /* 
        let input_scaler_layer: ScalingCuda = ScalingCuda::new(
            (1.0 / (n_heads as f32)).sqrt(), 2, lr, l2, dropout_rate
        );
        //let navigator_bottleneck_layer: DenseCuda = DenseCuda::new(input_resized_len, l2, lr, true, "navigator_bottleneck");
        let navigator_pre_scaler_layer: ScalingCuda = ScalingCuda::new(
            (6.0 / (n_heads as f32)).sqrt(), 2, lr, l2, dropout_rate
        );
        */
        let mut navigator_layer: DenseCuda = DenseCuda::new(
            input_embedding_len, latent_embedding_len, 
            n_heads, seq_len, l2, lr, true, "navigator");

        navigator_layer.use_bias = false;
        //let navigator_act_layer: ActivationCuda = ActivationCuda::new("tanh", n_heads, seq_len, latent_embedding_len, 1.0);
        //let navigator_l2norm_layer: L2NormCuda = L2NormCuda::new();
        let navigator_scaler_layer: ScalingCuda = ScalingCuda::new(
            //(6.0 / ((seq_len + latent_embedding_len) as f32)).sqrt(), 
            1.0 / (seq_len as f32).sqrt(),
            2, lr, l2, dropout_rate
        );
        //let navigator_pos_layer: PosEncoding2DCuda = PosEncoding2DCuda::new(
        //    lr, l2
        //);
        let dropout_layer: DropoutCuda = DropoutCuda::new(dropout_rate, "dropout");
        //let navigator_scaler_layer: BroadcastInvMulCuda = BroadcastInvMulCuda::new(
        //    (n_heads, seq_len, 1), 1.0 / (seq_len as f32).sqrt(), 2, lr
        //);
        //let navigator_output_layer: DenseCuda = DenseCuda::new(latent_embedding_len, l2, lr, true);

        let mut latent_modifier_layer: DenseCuda = DenseCuda::new(
            input_embedding_len, latent_embedding_len, 
            n_heads, seq_len, l2, lr, true, "latent_modifier");

        latent_modifier_layer.use_bias = false;
        //let latent_modifier_act_layer: ActivationCuda = ActivationCuda::new("tanh", 1.0);
        //let latent_modifier_norm_layer: LayerNormCuda = LayerNormCuda::new(lr, l2, 1);
        //let latent_modifier_scaler_layer: ScalingCuda = ScalingCuda::new(
        //    (6.0 / (seq_len as f32 + 1.0)).sqrt(), 
        //    //1.0 / (seq_len as f32).sqrt(),
        //    2, lr, l2, dropout_rate
        //);

        //let input_resizer_bottleneck_layer: DenseCuda = DenseCuda::new(input_resized_len, l2, lr, true, "input_resizer_bottleneck");
        //let input_resizer_pre_scaler_layer: ScalingCuda = ScalingCuda::new(
        //    (6.0 / (n_heads as f32)).sqrt(), 2, lr, l2, dropout_rate
        //);
        let mut input_resizer_layer: DenseCuda = DenseCuda::new(
            input_embedding_len, latent_embedding_len, 
            n_heads, seq_len, l2, lr, true, "input_resizer_layer");
        input_resizer_layer.use_bias = false;
        let input_resizer_act_layer: ActivationCuda = ActivationCuda::new("tanh", n_heads, seq_len, input_embedding_len, 1.0);
        //let input_resizer_norm_layer: LayerNormCuda = LayerNormCuda::new(lr, l2, 2);

        //let scaler_layer: ScalingCuda = ScalingCuda::new(1.0 / (seq_len as f32).sqrt(), 2, lr, l2, dropout_rate);
        
        //let aggregator_layer: DenseCuda = DenseCuda::new(input_resized_len, l2, lr, true);
        //let aggregator_norm_layer: LayerNormCuda = LayerNormCuda::new(lr, l2, 2);
        let aggregator_act_layer: ActivationCuda = ActivationCuda::new("tanh", n_heads, seq_len, input_embedding_len, append_scaler);
        let mut aggregator_resizer_layer: DenseCuda = DenseCuda::new(
            latent_embedding_len, input_embedding_len,
            n_heads, seq_len, l2, lr, true, "aggregator_resizer"
        );
        aggregator_resizer_layer.use_bias = false;
        
        //let value_bottleneck_layer: DenseCuda = DenseCuda::new(input_resized_len, l2, lr, true);
        //let value_expand_layer: DenseCuda = DenseCuda::new(input_embedding_len, l2, lr, true);
        //let mut value_layer: DenseCuda = DenseCuda::new(input_embedding_len, l2, lr, true, "value");
        //value_layer.use_bias = false;
        //let value_layer1: DenseCuda = DenseCuda::new(latent_embedding_len, l2, lr, true, "value1");
        //let value_act_layer: ActivationCuda = ActivationCuda::new("tanh", 1.0);
        //let value_norm_layer: LayerNormCuda = LayerNormCuda::new(lr, l2, 1);

        //let aggregator_scaler_layer: ScalingCuda = ScalingCuda::new(1.0 / (n_heads as f32).sqrt(), 2, lr, l2, dropout_rate);
        //let value_softmax: SoftmaxCuda = SoftmaxCuda::new();

        //let final_act_layer: ActivationCuda = ActivationCuda::new("silu", lr);
        
        //let latent_norm_layer: LayerNormCuda = LayerNormCuda::new(lr, l2, 2);
        let latent_act_layer: ActivationCuda = ActivationCuda::new("tanh", n_heads, 1, latent_embedding_len, 1.0);
        let latent_broadcast_layer: BroadcastMulCuda = BroadcastMulCuda::new(
            (n_heads, latent_embedding_len, input_embedding_len), 
            (6.0 / (latent_embedding_len + input_embedding_len) as f32).sqrt(),
            2, lr, l2
        );
        //let output_embedding_norm_layer: LayerNormCuda = LayerNormCuda::new(lr, l2, 2);
        //let output_act_layer: ActivationCuda = ActivationCuda::new("tanh", 1.0);
        
        //let extra_dense_layer: DenseCuda = DenseCuda::new(latent_embedding_len, l2, lr, true, "extra_dense");
        //let extra_dense_layer1: DenseCuda = DenseCuda::new(input_embedding_len, l2, lr, true, "extra_dense1");
        //let extra_dense_layer2: DenseCuda = DenseCuda::new(input_embedding_len, l2, lr, true, "extra_dense2");
        //let extra_act_layer: ActivationCuda = ActivationCuda::new("silu", 1.0);
        //let extra_act_layer1: ActivationCuda = ActivationCuda::new("silu", 1.0);
        //let extra_norm_layer: LayerNormCuda = LayerNormCuda::new(lr, l2, 2);

        //let head_out_dense_layer: DenseCuda = DenseCuda::new(input_embedding_len, l2, lr, true);

        // initialize the pointers
        //let input_embedding_ptr: String = new_cuda_ptr_str(&[1, seq_len, input_embedding_len])

        return Self
        {
            navigator_layer,
            //navigator_pre_scaler_layer,
            //navigator_act_layer,
            //navigator_l2norm_layer,
            navigator_scaler_layer,
            //navigator_pos_layer,
            dropout_layer,
            //navigator_output_layer,

            //latent_modifier_layer,
            //latent_modifier_act_layer,
            //latent_modifier_scaler_layer,
            //latent_modifier_norm_layer,

            //input_scaler_layer,
            input_resizer_layer,
            //input_resizer_pre_scaler_layer,
            input_resizer_act_layer,
            //input_resizer_norm_layer,

            //scaler_layer,

            //aggregator_layer,
            aggregator_act_layer,
            //aggregator_norm_layer,
            aggregator_resizer_layer,
            //aggregator_scaler_layer,
            //latent_norm_layer,
            latent_act_layer,
            latent_broadcast_layer,

            //value_layer, value_layer1,
            //value_norm_layer,
            //value_act_layer,
            //value_softmax,

            //output_embedding_norm_layer,
            //output_act_layer,

            //navigator_bottleneck_layer,
            //input_resizer_bottleneck_layer,
            //latent_bottleneck_layer,
            //value_bottleneck_layer,
            //value_expand_layer,
            //final_act_layer,
            //head_out_dense_layer,
            //extra_dense_layer, extra_act_layer,
            //extra_dense_layer1, extra_act_layer1,
            //extra_norm_layer, extra_dense_layer2,

            //head_embeddings3d: Array3::zeros((n_attention_heads, seq_len, output_embedding_len)),

            //input_tensor: Array3::zeros((0, 0, 0)),
            input_embedding_len,
            latent_embedding_len,
            n_heads,
            seq_len,
            l2, return_sequence,
            //input_resized_len,

            main_input_ptr: "".to_string(),
            main_input_grads_ptr: "".to_string(),
            main_output_ptr: "".to_string(), 
            main_output_grads_ptr: "".to_string(),

            //input_embedding_ptr,
            input_embedding_broadcasted_ptr: "".to_string(),
            //input_embedding_broadcasted_t_ptr: "".to_string(),

            latent_ptr: "".to_string(),
            //latent_resized_ptr: "".to_string(),
            latent_ptr_t: "".to_string(),
            //latent_expanded_horz_ptr: "".to_string(),
            //value_modifier_ptr: "".to_string(),
            
            //input_resizer_ptr: "".to_string(),

            output_embedding_ptr: "".to_string(),
            output_embedding_vector_ptr: "".to_string(),
            //residual_input: "".to_string(),

            // gradient pointers
            //residual_grads: "".to_string(),
            output_grads_broadcasted_ptr: "".to_string(),
            output_grads_broadcasted_head_ptr: "".to_string(),
            //grads_input_resizer: "".to_string(),
            //grads_latent: "".to_string(),
            //latent_expanded_grad_ptr: "".to_string(),
            //value_modifier_grad_ptr: "".to_string(),
            latent_grad_ptr: "".to_string(),
            latent_grad_broadcasted_ptr: "".to_string(),

            //val_grads_t_ptr: "".to_string(),
            //return_grads_val: "".to_string(),
            return_grads: "".to_string(),

            ptrs_allocated: false,
            sub_layers_allocated: false,
        };
    }

    pub fn initialize_ptrs(&mut self)
    {
        self.input_embedding_broadcasted_ptr = new_cuda_ptr_str(&[self.n_heads, self.seq_len, self.input_embedding_len]);
        //self.input_embedding_broadcasted_t_ptr = new_cuda_ptr_str(&[self.n_heads, self.input_resized_len, self.seq_len]);
        self.latent_ptr = new_cuda_ptr_str(&[self.n_heads, 1, self.latent_embedding_len]);
        //self.latent_resized_ptr = new_cuda_ptr_str(&[self.n_heads, 1, self.latent_embedding_len]);
        self.latent_ptr_t = new_cuda_ptr_str(&[self.n_heads, self.latent_embedding_len, 1]);
        //self.latent_expanded_horz_ptr = new_cuda_ptr_str(&[self.n_heads, self.input_embedding_len, self.latent_embedding_len]);
        //self.value_modifier_ptr = new_cuda_ptr_str(&[self.n_heads, self.latent_embedding_len, self.input_embedding_len]);
        //self.input_resizer_ptr = new_cuda_ptr_str(&[self.n_heads, self.seq_len, self.input_embedding_len]);

        self.output_embedding_ptr = new_cuda_ptr_str(&[1, self.seq_len, self.input_embedding_len]);
        self.output_embedding_vector_ptr = new_cuda_ptr_str(&[1, 1, self.input_embedding_len]);
        
        //self.residual_input = new_cuda_ptr_str(&[1, self.seq_len, self.input_embedding_len]);

        //self.residual_grads = new_cuda_ptr_str(&[1, self.seq_len, self.input_embedding_len]);
        self.output_grads_broadcasted_ptr = new_cuda_ptr_str(&[1, self.seq_len, self.input_embedding_len]);
        self.output_grads_broadcasted_head_ptr = new_cuda_ptr_str(&[self.n_heads, self.seq_len, self.input_embedding_len]);
        //self.latent_expanded_grad_ptr = new_cuda_ptr_str(&[self.n_heads, self.latent_embedding_len, self.input_embedding_len]);
        //self.value_modifier_grad_ptr = new_cuda_ptr_str(&[self.n_heads, self.input_embedding_len, self.latent_embedding_len]);
        self.latent_grad_ptr = new_cuda_ptr_str(&[self.n_heads, 1, self.latent_embedding_len]);
        self.latent_grad_broadcasted_ptr = new_cuda_ptr_str(&[self.n_heads, self.seq_len, self.latent_embedding_len]);
        self.return_grads = new_cuda_ptr_str(&[1, self.seq_len, self.input_embedding_len]);
    }

    pub fn link_layers(&mut self)
    {
        //self.navigator_layer.input_ptr = self.input_embedding_broadcasted_ptr.clone();
        //self.navigator_layer.input_grads_ptr = self.
    }

    // input_embeddings is 3D (1, seq_len, input_embedding_len)
    pub fn forward(&mut self, use_dropout: bool)
    {
        if !self.ptrs_allocated
        {
            self.initialize_ptrs();
            self.ptrs_allocated = true;
        }

        let main_input_ptr: *mut f32 = string_to_ptr(&self.main_input_ptr);
        let main_input_grad_ptr: *mut f32 = string_to_ptr(&self.main_input_grads_ptr);
        let main_output_ptr: *mut f32 = string_to_ptr(&self.main_output_ptr);
        let main_output_grad_ptr: *mut f32 = string_to_ptr(&self.main_output_grads_ptr);


        let input_embedding_broadcasted_ptr: *mut f32 = string_to_ptr(&self.input_embedding_broadcasted_ptr);
        //let input_embedding_broadcasted_t_ptr: *mut f32 = string_to_ptr(&self.input_embedding_broadcasted_t_ptr);
        let latent_ptr: *mut f32 = string_to_ptr(&self.latent_ptr);
        //let latent_resized_ptr: *mut f32 = string_to_ptr(&self.latent_resized_ptr);
        let latent_ptr_t: *mut f32 = string_to_ptr(&self.latent_ptr_t);
        //let latent_expanded_horz_ptr: *mut f32 = string_to_ptr(&self.latent_expanded_horz_ptr);
        //let value_modifier_ptr: *mut f32 = string_to_ptr(&self.value_modifier_ptr);
        //let input_resizer_ptr: *mut f32 = string_to_ptr(&self.input_resizer_ptr);
        let output_embedding_ptr: *mut f32 = string_to_ptr(&self.output_embedding_ptr);
        let output_embedding_vector_ptr: *mut f32 = string_to_ptr(&self.output_embedding_vector_ptr);
        //let residual_input: *mut f32 = string_to_ptr(&self.residual_input);

        //println!("{:?}", cuda_ptr_to_array(input_embeddings_struct.get_ptr(), input_embeddings_struct.get_shape()));
        // broadcast along z axis
        //let start = Instant::now();
        //self.input_scaler_layer.forward(input_embeddings_struct);

        broadcast_2d_to_3d(
            input_embedding_broadcasted_ptr, 
            self.n_heads, self.seq_len, self.input_embedding_len, 
            input_ptr, 0, 0, true
        );

        // bottleneck the value layer
        /*
        let mut val_ptr_struct: CudaTensorPtr = CudaTensorPtr {
            tensor_ptr: ptr_to_string(input_embedding_broadcasted_ptr),
            shape: vec![self.n_heads, self.seq_len, self.input_embedding_len]
        };

        self.value_bottleneck_layer.forward(&mut val_ptr_struct);

        // for value matrix
        transpose_2d(
            input_embedding_broadcasted_t_ptr, 
            val_ptr_struct.get_ptr(), 
            self.n_heads, self.seq_len, self.input_resized_len
        );
        */
        //let end = start.elapsed();
        //println!("broadcast_2d_to_3d: {:?}", end.as_secs_f64());
        //println!("{:?}", cuda_ptr_to_array(input_embedding_broadcasted_ptr, &[self.n_heads, self.seq_len, self.input_embedding_len]));

        //exit(1);
        // define pointer structs for passing through query and key layers
        // make a second copy for passing through hidden layer and transformation layer
        //let mut navigator_ptr_struct: CudaTensorPtr = CudaTensorPtr {
        //    tensor_ptr: ptr_to_string(input_embedding_broadcasted_ptr),
        //    shape: vec![self.n_heads, self.seq_len, self.input_embedding_len]
        //};
        //let mut latent_modifier_ptr_struct: CudaTensorPtr = navigator_ptr_struct.clone();
        //let mut aggregator_ptr_struct: CudaTensorPtr = navigator_ptr_struct.clone();

        //println!("{:?}", cuda_ptr_to_array(aggregator_ptr_struct.get_ptr(), aggregator_ptr_struct.get_shape()));

        //val_ptr_struct.set_ptr(input_embedding_broadcasted_t_ptr, vec![self.n_heads, self.input_resized_len, self.seq_len]);

        //let mut original_input_ptr_struct: CudaTensorPtr = key_ptr_struct.clone();

        // make a copy of the broadcast 
        //////////////////////////////////////////////////
        // batch matrix multiplication to create key and query embeddings

        //println!("{:?}", cuda_ptr_to_array(key_ptr_struct.get_ptr(), key_ptr_struct.get_shape()));
        //println!("{:?}", cuda_ptr_to_array(query_ptr_struct.get_ptr(), query_ptr_struct.get_shape()));
        //println!("{:?}", cuda_ptr_to_array(value_ptr_struct.get_ptr(), value_ptr_struct.get_shape()));

        /*
        self.latent_modifier_layer.forward(&mut latent_modifier_ptr_struct);
        self.latent_modifier_act_layer.forward(&mut latent_modifier_ptr_struct);
        scalar_op_3d_inplace(
            latent_modifier_ptr_struct.get_ptr(), 
            (self.seq_len as f32).sqrt(), 3, 
            self.n_heads, self.seq_len, self.latent_embedding_len
        );
        */
        //println!("{:?}", cuda_ptr_to_array(latent_modifier_ptr_struct.get_ptr(), latent_modifier_ptr_struct.get_shape()));
        //println!("{:?}", cuda_ptr_to_array(latent_modifier_ptr_struct.get_ptr(), latent_modifier_ptr_struct.get_shape()));
        //self.latent_modifier_scaler_layer.forward(&mut latent_modifier_ptr_struct);

        //println!("---------------------------------");
        //self.navigator_bottleneck_layer.forward(&mut navigator_ptr_struct);
        //println!("query_norm: {:?}", cuda_ptr_to_array(navigator_ptr_struct.get_ptr(), navigator_ptr_struct.get_shape()));
        //self.navigator_pre_scaler_layer.forward(&mut navigator_ptr_struct);
        self.navigator_layer.forward(&mut navigator_ptr_struct);
        //self.navigator_l2norm_layer.forward(&mut navigator_ptr_struct);

        //println!("query_norm: {:?}", cuda_ptr_to_array(navigator_ptr_struct.get_ptr(), navigator_ptr_struct.get_shape()));
        /*
        let latent_modifier_shape: &[usize] = latent_modifier_ptr_struct.get_shape();
        //println!("{:?}", latent_modifier_shape);
        self.navigator_scaler_layer.set_weights_from_ptr(
            latent_modifier_ptr_struct.get_ptr(),
            (latent_modifier_shape[0], latent_modifier_shape[1], latent_modifier_shape[2])
        );
        */
        //self.navigator_act_layer.forward(&mut navigator_ptr_struct);
        self.navigator_scaler_layer.forward(&mut navigator_ptr_struct, use_dropout);
        //self.dropout_layer.forward(&mut navigator_ptr_struct, use_dropout);
        //self.navigator_pos_layer.forward(&mut navigator_ptr_struct);
        //scalar_op_3d_inplace(
        //    navigator_ptr_struct.get_ptr(), 
        //    (self.seq_len as f32).sqrt(), 3, 
        //    self.n_heads, self.seq_len, self.latent_embedding_len
        //);
        //println!("{:?}", cuda_ptr_to_array(string_to_ptr(&self.navigator_scaler_layer.weight_ptr), navigator_ptr_struct.get_shape()));
        //println!("{:?}", cuda_ptr_to_array(navigator_ptr_struct.get_ptr(), navigator_ptr_struct.get_shape()));
        //println!("==================================================================");
        //println!("query_norm: {:?}", cuda_ptr_to_array(navigator_ptr_struct.get_ptr(), navigator_ptr_struct.get_shape()));
        //exit(1);
        //scalar_op_3d_inplace(navigator_ptr_struct.get_ptr(), 10.0, 0, self.n_heads, self.seq_len, self.latent_embedding_len);
        //self.navigator_output_layer.forward(&mut navigator_ptr_struct);
        //self.input_resizer_bottleneck_layer.forward(&mut aggregator_ptr_struct);
        if self.input_embedding_len != self.latent_embedding_len
        {
            self.input_resizer_layer.forward(&mut aggregator_ptr_struct);
            //self.input_resizer_act_layer.forward(&mut aggregator_ptr_struct);
        }
        //self.input_resizer_pre_scaler_layer.forward(&mut aggregator_ptr_struct);
        //self.input_resizer_norm_layer.forward(&mut aggregator_ptr_struct);
        ////self.input_resizer_act_layer.forward(&mut aggregator_ptr_struct);
        //copy_cuda_to_cuda(input_resizer_ptr, aggregator_ptr_struct.get_ptr(), &[self.n_heads, self.seq_len, self.latent_embedding_len]);
        
        //println!("query_norm: {:?}", cuda_ptr_to_array(navigator_ptr_struct.get_ptr(), navigator_ptr_struct.get_shape()));
        //exit(1);
        //let start = Instant::now();
        //self.kqv_layer.forward(&mut key_ptr_struct);
        //let end = start.elapsed();
        //println!("kqv: {:?}", end.as_secs_f64());
        //println!("k_matrix: {:?}", cuda_ptr_to_array(string_to_ptr(&self.kqv_layer.k_weight_ptr), &[self.kqv_layer.k_in_shape.0, self.kqv_layer.k_in_shape.2, self.kqv_layer.k_out_shape.2]));
        //println!("k_out: {:?}", cuda_ptr_to_array(string_to_ptr(&self.kqv_layer.k_result_ptr), &[self.kqv_layer.k_out_shape.0, self.kqv_layer.k_out_shape.1, self.kqv_layer.k_out_shape.2]));
        //println!("q_matrix: {:?}", cuda_ptr_to_array(string_to_ptr(&self.kqv_layer.q_weight_ptr), &[self.kqv_layer.q_in_shape.0, self.kqv_layer.q_in_shape.2, self.kqv_layer.q_out_shape.2]));
        //println!("q_out: {:?}", cuda_ptr_to_array(string_to_ptr(&self.kqv_layer.q_result_ptr), &[self.kqv_layer.q_out_shape.0, self.kqv_layer.q_out_shape.1, self.kqv_layer.q_out_shape.2]));
        //println!("v_matrix: {:?}", cuda_ptr_to_array(string_to_ptr(&self.kqv_layer.v_weight_ptr), &[self.kqv_layer.v_in_shape.0, self.kqv_layer.v_in_shape.2, self.kqv_layer.v_out_shape.2]));
        //println!("v_out: {:?}", cuda_ptr_to_array(string_to_ptr(&self.kqv_layer.v_result_ptr), &[self.kqv_layer.v_out_shape.0, self.kqv_layer.v_out_shape.1, self.kqv_layer.v_out_shape.2]));
        
        //println!("{:?}", cuda_ptr_to_array(query_ptr_struct.get_ptr(), query_ptr_struct.get_shape()));
        //println!("{:?}", cuda_ptr_to_array(value_ptr_struct.get_ptr(), value_ptr_struct.get_shape()));

        //exit(1);

        //println!("{:?}", cuda_ptr_to_array(key_ptr_struct.get_ptr(), key_ptr_struct.get_shape()));
        //println!("{:?}", cuda_ptr_to_array(query_ptr_struct.get_ptr(), query_ptr_struct.get_shape()));
        //println!("{:?}", cuda_ptr_to_array(value_ptr_struct.get_ptr(), value_ptr_struct.get_shape()));

        // sum navigation output along axis 1
        // move in the high dimensional space

        let shape: &[usize] = navigator_ptr_struct.get_shape();
        //println!("{:?}", shape);

        //println!("{:?}", cuda_ptr_to_array(navigator_ptr_struct.get_ptr(), navigator_ptr_struct.get_shape()));
        sum_axis(
            latent_ptr, 
            navigator_ptr_struct.get_ptr(), 
            shape[0], shape[1], shape[2],
            1, true
        );
        //println!("{:?}", cuda_ptr_to_array(latent_ptr, &[self.n_heads, 1, self.latent_embedding_len]));

        //scalar_op_3d_inplace(
        //    latent_ptr, 
        //    (self.seq_len as f32).sqrt(), 3, 
        //    self.n_heads, 1, self.latent_embedding_len
        //);
        //println!("{:?}", cuda_ptr_to_array(latent_ptr, &[self.n_heads, 1, self.latent_embedding_len]));
        //exit(1);

        //println!("latent_coord: {:?}", cuda_ptr_to_array(latent_ptr, &[self.n_heads, 1, self.input_resized_len]));
        //exit(1);
        // overwrite the existing result of navigator_ptr_struct
        navigator_ptr_struct.set_ptr(latent_ptr, vec![self.n_heads, 1, self.latent_embedding_len]);
        //self.scaler_layer.forward(&mut navigator_ptr_struct);
        ////self.dropout_layer.forward(&mut navigator_ptr_struct, use_dropout);
        self.latent_act_layer.forward(&mut navigator_ptr_struct);
        //println!("{:?}", cuda_ptr_to_array(string_to_ptr(&//self.scaler_layer.weight_ptr), &[self.n_heads, 1, self.latent_embedding_len]));
        //self.latent_norm_layer.forward(&mut navigator_ptr_struct);
        //self.latent_bottleneck_layer.forward(&mut navigator_ptr_struct);

        // store resized latent coordinate for backprop
        //copy_cuda_to_cuda(latent_resized_ptr, navigator_ptr_struct.get_ptr(), &[self.n_heads, 1, self.latent_embedding_len]);
        
        // transpose to (resized_latent, 1), matrix multiply with (1, input_embedding) to produce (resized_latent, input_embedding)
        //println!("{:?}", cuda_ptr_to_array(latent_resized_ptr, &[self.n_heads, 1, self.input_resized_len]));
        
        transpose_2d(latent_ptr_t, navigator_ptr_struct.get_ptr(), self.n_heads, 1, self.latent_embedding_len);
        let mut val_ptr_struct: CudaTensorPtr = CudaTensorPtr {
            tensor_ptr: ptr_to_string(latent_ptr_t),
            shape: vec![self.n_heads, self.latent_embedding_len, 1]
        };
        //println!("{:?}", cuda_ptr_to_array(val_ptr_struct.get_ptr(),val_ptr_struct.get_shape()));
        self.latent_broadcast_layer.forward(&mut val_ptr_struct);
        //self.value_layer.forward(&mut val_ptr_struct);
        //self.final_act_layer.forward(&mut val_ptr_struct);
        //println!("{:?}", cuda_ptr_to_array(val_ptr_struct.get_ptr(),val_ptr_struct.get_shape()));
        //exit(1);
        /*
        transpose_2d(
            latent_expanded_horz_ptr, val_ptr_struct.get_ptr(), 
            self.n_heads, self.latent_embedding_len, self.input_embedding_len
        );
        //println!("{:?}", cuda_ptr_to_array(val_ptr_struct.get_ptr(),val_ptr_struct.get_shape()));
        val_ptr_struct.set_ptr(
            latent_expanded_horz_ptr, 
            vec![self.n_heads, self.input_embedding_len, self.latent_embedding_len]
        );
        //println!("{:?}", cuda_ptr_to_array(val_ptr_struct.get_ptr(),val_ptr_struct.get_shape()));
        //exit(1);

        self.value_layer1.forward(&mut val_ptr_struct);
        */
        //println!("{:?}", cuda_ptr_to_array(val_ptr_struct.get_ptr(),&[self.n_heads, self.input_embedding_len, self.latent_embedding_len]));
        //exit(1);
        //scalar_op_3d_inplace(
        //    val_ptr_struct.get_ptr(), 
        //    (self.latent_embedding_len as f32).sqrt(), 3, 
        //    self.n_heads, self.input_embedding_len, self.latent_embedding_len
        //);
        //self.value_softmax.forward(&mut val_ptr_struct);
        //self.value_norm_layer.forward(&mut val_ptr_struct);
        //self.value_act_layer.forward(&mut val_ptr_struct);
        /*
        transpose_2d(
            value_modifier_ptr, val_ptr_struct.get_ptr(), 
            self.n_heads, self.input_embedding_len, self.latent_embedding_len
        );
        val_ptr_struct.set_ptr(
            value_modifier_ptr, 
            vec![self.n_heads, self.latent_embedding_len, self.input_embedding_len]
        );
        */
        //println!("{:?}", cuda_ptr_to_array(val_ptr_struct.get_ptr(),&[self.n_heads, self.latent_embedding_len, self.input_embedding_len]));
        //exit(1);

        // transfer (resized, input_embedding) to aggregator_resized
        self.aggregator_resizer_layer.set_weights_from_ptr(
            val_ptr_struct.get_ptr(),
            (self.n_heads, self.latent_embedding_len, self.input_embedding_len), 
            false,
        );

        //key_ptr_struct.set_ptr(
        //    string_to_ptr(&self.kqv_layer.k_result_ptr), 
        //    vec![shape.0, shape.1, shape.2]
        //);
        //let end = start.elapsed();
        //println!("set_weights: {:?}", end.as_secs_f64());
        //    tensor_ptr: ptr_to_string(input_embedding_broadcasted_ptr),
        //    shape: vec![self.n_attention_heads, self.seq_len, self.input_embedding_len]
        //};

        // calculate attention scores and record key/query matrices
        //println!("{:?}", cuda_ptr_to_array(key_ptr_struct.get_ptr(), &[shape.0, shape.1, shape.2]));
        //println!("{:?}", cuda_ptr_to_array(query_ptr_struct.get_ptr(), query_ptr_struct.get_shape()));
        //println!("{:?}", cuda_ptr_to_array(string_to_ptr(&self.attention_layer.weight_ptr), &[shape.0, shape.2, shape.1]));
        //let start = Instant::now();
        //self.aggregator_layer.forward(&mut aggregator_ptr_struct);
        //println!("{:?}", cuda_ptr_to_array(aggregator_ptr_struct.get_ptr(), aggregator_ptr_struct.get_shape()));
        self.aggregator_resizer_layer.forward(&mut aggregator_ptr_struct);
        ////self.aggregator_act_layer.forward(&mut aggregator_ptr_struct);
        //self.aggregator_norm_layer.forward(&mut aggregator_ptr_struct);
        //println!("{:?}", cuda_ptr_to_array(string_to_ptr(&self.aggregator_resizer_layer.weight_ptr), &[self.n_heads, self.input_embedding_len, self.input_embedding_len]));
        //println!("{:?}", cuda_ptr_to_array(input_embeddings_struct.get_ptr(), input_embeddings_struct.get_shape()));
        //exit(1);
        //scalar_op_3d_inplace(
        //    aggregator_ptr_struct.get_ptr(), 
        //    (self.input_resized_len as f32).sqrt(), 3, 
        //    self.n_heads, self.seq_len, self.input_embedding_len
        //);
        
        //println!("{:?}", cuda_ptr_to_array(aggregator_ptr_struct.get_ptr(), aggregator_ptr_struct.get_shape()));
        //println!("{:?}", cuda_ptr_to_array(key_ptr_struct.get_ptr(), key_ptr_struct.get_shape()));
        //exit(1);

        //zeroes_3d_inplace(output_embedding_ptr, 1, self.seq_len, self.input_embedding_len);

        //let start = Instant::now();
        // concatenate the navigation heads
        sum_axis(
            output_embedding_ptr, aggregator_ptr_struct.get_ptr(), 
            self.n_heads, self.seq_len, self.input_embedding_len, 
            0, true
        );

        let mut head_sum_struct: CudaTensorPtr = CudaTensorPtr
        {
            tensor_ptr: self.output_embedding_ptr.clone(),
            shape: vec![1, self.seq_len, self.input_embedding_len]
        };

        //scalar_op_3d_inplace(
        //    head_sum_struct.get_ptr(), self.n_heads as f32, 
        //    3, 1, self.seq_len, self.input_embedding_len
        //);

        self.aggregator_act_layer.forward(&mut head_sum_struct);
        //self.aggregator_scaler_layer.forward(&mut head_sum_struct);

        //self.head_sum_scaler_layer.forward(input_embeddings_struct);

        //println!("{:?}", cuda_ptr_to_array(output_embedding_ptr, &[1, self.seq_len, self.input_embedding_len]));
        //exit(1);

        //scalar_op_3d_inplace(
        //    output_embedding_ptr, 
        //    (self.n_heads as f32).sqrt(), 3, 
        //    1, self.seq_len, self.input_embedding_len
        //);
        //let end = start.elapsed();
        //println!("sum: {:?}", end.as_secs_f64());
        //let end: Instant = Instant::now();
        //let duration = end.duration_since(start);
        //let secs = duration.as_secs_f64();

        //println!("time: {:?}", secs);

        //println!("{:?}", cuda_ptr_to_array(output_embedding_ptr, &[1, self.seq_len, self.input_embedding_len]));
        //exit(1);
        //println!("{:?}", cuda_ptr_to_array(input_embeddings_struct.get_ptr(), input_embeddings_struct.get_shape()));
        
        // add the original input embeddings (transforms the input embedding)
        //let start = Instant::now(); 
        sum_axis(
            head_sum_struct.get_ptr(), 
            input_embeddings_struct.get_ptr(), 
            1, self.seq_len, self.input_embedding_len, 
            0, false
        );

        //scalar_op_3d_inplace(
        //    output_embedding_ptr, (self.n_heads + 1) as f32, 3, 
        //    1, self.seq_len, self.input_embedding_len
        //);

        //let end = start.elapsed();
        //println!("sum: {:?}", end.as_secs_f64());

        //println!("{:?}", cuda_ptr_to_array(output_embedding_ptr, &[1, self.seq_len, self.input_embedding_len]));
        //exit(1);
        //println!("{:?}", cuda_ptr_to_array(output_embedding_ptr, &[1, self.seq_len, self.output_embedding_len]));
        //let embeddings_sum: Array3<f32> = (self.head_embeddings3d.sum_axis(Axis(0))).insert_axis(Axis(0));

        //let mut output_embeddings: ArrayD<f32> = embeddings_sum.into_dyn();
        
        //let input_tensors3d: Array3<f32> = input_embeddings.into_dimensionality().unwrap();
        //self.input_tensor = input_tensors3d;

        input_embeddings_struct.set_ptr(
            head_sum_struct.get_ptr(), 
            vec![1, self.seq_len, self.input_embedding_len]
        );

        //////self.aggregator_act_layer.forward(input_embeddings_struct);
        //self.output_act_layer.forward(input_embeddings_struct);

        //self.output_embedding_norm_layer.forward(input_embeddings_struct);

        //let start = Instant::now(); 
        if self.return_sequence
        {
            //let mut residual_input: CudaTensorPtr = input_embeddings_struct.clone();

            //self.extra_dense_layer.forward(&mut residual_input);
            //self.extra_act_layer.forward(input_embeddings_struct);
            //self.extra_dense_layer1.forward(&mut residual_input);
            //self.extra_act_layer1.forward(&mut residual_input);
            //self.extra_dense_layer2.forward(&mut residual_input);

            //println!("{:?}", cuda_ptr_to_array(input_embeddings_struct.get_ptr(), input_embeddings_struct.get_shape()));
            //exit(1);
            //sum_axis(
            //    input_embeddings_struct.get_ptr(), 
            //    residual_input.get_ptr(), 
            //   1, self.seq_len, self.input_embedding_len, 
            //    0, false
            //);
            /**/

            //scalar_op_3d_inplace(
            //    input_embeddings_struct.get_ptr(), 
            //    self.n_heads as f32 + 1.0, 3, 
            //    1, self.seq_len, self.input_embedding_len
            //);
        }
        else
        {
            // zeroes_3d_inplace(output_embedding_vector_ptr, 1, 1, self.input_embedding_len);
            //sum_axis(
            //    output_embedding_vector_ptr, input_embeddings_struct.get_ptr(), 
            //    1, self.seq_len, self.input_embedding_len, 
            //    1, true
            //);

            let last_embedding_ptr: *mut f32 = unsafe {input_embeddings_struct.get_ptr().add((self.seq_len - 1) * self.input_embedding_len)};
            // only get the last embedding, each embedding all shared the same info
            copy_cuda_to_cuda(
                output_embedding_vector_ptr, last_embedding_ptr, 
                &[1, 1, self.input_embedding_len]
            );


            //println!("{:?}", cuda_ptr_to_array(input_embeddings_struct.get_ptr(), input_embeddings_struct.get_shape()));
            //println!("{:?}", cuda_ptr_to_array(output_embedding_vector_ptr, &[1, 1, self.input_embedding_len]));
            //exit(1);
            input_embeddings_struct.set_ptr(
                output_embedding_vector_ptr, 
                vec![1, 1, self.input_embedding_len]
            );

            //scalar_op_3d_inplace(
            //    input_embeddings_struct.get_ptr(), 
            //    self.seq_len as f32, 3, 
            //    1, 1, self.input_embedding_len
            //);
            
            //self.extra_norm_layer.forward(input_embeddings_struct);
        }
        
        //self.extra_norm_layer.forward(input_embeddings_struct);

        //let end = start.elapsed();
        //println!("sum_and_norm: {:?}", end.as_secs_f64());

        //println!("{:?}", cuda_ptr_to_array(input_embeddings_struct.get_ptr(), input_embeddings_struct.get_shape()));
        //exit(1);
    }

    pub fn backward(&mut self, grads: &mut CudaTensorPtr, use_dropout: bool)
    {
        let output_grads_broadcasted_ptr: *mut f32 = string_to_ptr(&self.output_grads_broadcasted_ptr);
        let output_grads_broadcasted_head_ptr: *mut f32 = string_to_ptr(&self.output_grads_broadcasted_head_ptr);
        //let grads_input_resizer: *mut f32 = string_to_ptr(&self.grads_input_resizer);
        //let grads_latent: *mut f32 = string_to_ptr(&self.grads_latent);
        //let latent_expanded_grad_ptr: *mut f32 = string_to_ptr(&self.latent_expanded_grad_ptr);
        //let value_modifier_grad_ptr: *mut f32 = string_to_ptr(&self.value_modifier_grad_ptr);
        let latent_grad_ptr: *mut f32 = string_to_ptr(&self.latent_grad_ptr);
        let latent_grad_broadcasted_ptr: *mut f32 = string_to_ptr(&self.latent_grad_broadcasted_ptr);
        //let val_grads_t_ptr: *mut f32 = string_to_ptr(&self.val_grads_t_ptr);
        //let return_grads_val: *mut f32 = string_to_ptr(&self.return_grads_val);
        let return_grads: *mut f32 = string_to_ptr(&self.return_grads);

        //let latent_ptr: *mut f32 = string_to_ptr(&self.latent_ptr);
        //let latent_resized_ptr: *mut f32 = string_to_ptr(&self.latent_resized_ptr);
        //let input_resizer_ptr: *mut f32 = string_to_ptr(&self.input_resizer_ptr);
        
        //self.extra_norm_layer.backward(grads);

        //println!("{:?}", cuda_ptr_to_array(grads.get_ptr(), grads.get_shape()));
        if self.return_sequence
        {
            // already (1, seq_len, output_embedding)
            // pass

            //let mut residual_grads: CudaTensorPtr = grads.clone();
            //println!("{:?}", cuda_ptr_to_array(grads.get_ptr(), grads.get_shape()));
            //self.extra_dense_layer2.backward(&mut residual_grads);
            //self.extra_act_layer1.backward(&mut residual_grads);
            //self.extra_dense_layer1.backward(&mut residual_grads);
            //self.extra_act_layer.backward(grads);
            //self.extra_dense_layer.backward(&mut residual_grads);
            //println!("{:?}", cuda_ptr_to_array(residual_grads.get_ptr(), residual_grads.get_shape()));
            //exit(1);

            //sum_axis(
            //    grads.get_ptr(), 
            //    residual_grads.get_ptr(), 
            //    1, self.seq_len, self.input_embedding_len, 
            //    0, false
            //);

            //scalar_op_3d_inplace(
            //    grads.get_ptr(), 
            //    self.n_heads as f32 + 1.0, 3, 
            //    1, self.seq_len, self.input_embedding_len
            //);
        }
        else
        {
            // the mean of embeddings was returned instead
            // (1, 1, output_embedding)

            //backpropagate through mean 
            // mean =  embeddings / y_length
            // derivative mean = 1 / y_length

            //self.extra_norm_layer.backward(grads);

            // broadcast along y axis and divide by seq_len
            //scalar_op_3d_inplace(
            //    grads.get_ptr(), 
            //    self.seq_len as f32, 3, 
            //    1, 1, self.input_embedding_len
            //);

            //broadcast_2d_to_3d(
            //    output_grads_broadcasted_ptr, 
            //    1, self.seq_len, self.input_embedding_len, 
            //    grads.get_ptr(), 1, 0, true
            //);

            // only set last embedding, all tokens share the same info
            //println!("{:?}", cuda_ptr_to_array(grads.get_ptr(), grads.get_shape()));
            //println!("{:?}", cuda_ptr_to_array(output_grads_broadcasted_ptr, &[1, self.seq_len, self.input_embedding_len]));
            let last_output_grads_ptr: *mut f32 = unsafe {output_grads_broadcasted_ptr.add((self.seq_len - 1) * self.input_embedding_len)};
            copy_cuda_to_cuda(
                last_output_grads_ptr, grads.get_ptr(), 
                &[1, 1, self.input_embedding_len]
            );
            grads.set_ptr(output_grads_broadcasted_ptr, vec![1, self.seq_len, self.input_embedding_len]);
            //println!("{:?}", cuda_ptr_to_array(grads.get_ptr(), grads.get_shape()));
            //exit(1);
        }

        //////self.aggregator_act_layer.backward(grads);
        //self.output_act_layer.backward(grads);

        //self.output_embedding_norm_layer.backward(grads);

        //println!("{:?}", cuda_ptr_to_array(grads.get_ptr(), grads.get_shape()));
        //println!("{:?}", cuda_ptr_to_array(return_grads, grads.get_shape()));
        // add gradients directly to return grads due to residual connection
        //zeroes_3d_inplace(return_grads, 1, self.seq_len, self.input_embedding_len);

        // sum due to residual connection
        sum_axis(
            return_grads, 
            grads.get_ptr(),
            1, self.seq_len, self.input_embedding_len,
            0, true
        );

        //self.aggregator_scaler_layer.backward(grads);
        self.aggregator_act_layer.backward(grads);

        // backpropagate through adaptive scaling
        //self.head_sum_scaler_layer.backward(grads);

        //scalar_op_3d_inplace(
        //    grads.get_ptr(), 
        //    self.n_heads as f32, 3, 
        //    1, self.seq_len, self.input_embedding_len
        //);

        //println!("{:?}", cuda_ptr_to_array(return_grads, &[1, self.seq_len, self.input_embedding_len]));
        //println!("{:?}", cuda_ptr_to_array(grads.get_ptr(), grads.get_shape()));
        //exit(1);

        // backpropagate through summation along first axis
        // broadcast along z axis, inverse of concatenating heads
        broadcast_2d_to_3d(
            output_grads_broadcasted_head_ptr, 
            self.n_heads, self.seq_len, self.input_embedding_len, 
            grads.get_ptr(), 0, 0, true
        );
        
        grads.set_ptr(
            output_grads_broadcasted_head_ptr, 
            vec![self.n_heads, self.seq_len, self.input_embedding_len]
        );

        //println!("{:?}", cuda_ptr_to_array(grads.get_ptr(), grads.get_shape()));

        // backpropagate through aggregator resizer layer
        //scalar_op_3d_inplace(
        //    grads.get_ptr(), 
        //    1.0 / (self.input_resized_len as f32).sqrt(), 2, 
        //    self.n_heads, self.seq_len, self.input_embedding_len
        //);
        //self.aggregator_norm_layer.backward(grads);
        //println!("{:?}", cuda_ptr_to_array(grads.get_ptr(), grads.get_shape()));
        ////self.aggregator_act_layer.backward(grads);

        zeroes_3d_inplace(
            string_to_ptr(&self.aggregator_resizer_layer.weight_gradients_ptr), 
            self.aggregator_resizer_layer.in_shape.0,
            self.aggregator_resizer_layer.in_shape.2,
            self.aggregator_resizer_layer.out_shape.2,
        );
        self.aggregator_resizer_layer.backward(grads);
        //self.aggregator_layer.backward(grads);

        // gradient from grads needs to be copied two times, due to the product
        // between the input resizer output and 1d latent vector, and both
        // need to be multiplied with separate coefficients
        /*
        copy_cuda_to_cuda(grads_input_resizer, grads.get_ptr(), &[self.n_heads, self.seq_len, self.input_resized_len]);
        copy_cuda_to_cuda(grads_latent, grads.get_ptr(), &[self.n_heads, self.seq_len, self.input_resized_len]);
        //println!("{:?}", cuda_ptr_to_array(grads_input_resizer, &[self.n_heads, self.seq_len, self.input_resized_len]));
        //exit(1);
        let mut grads_input_resizer_struct: CudaTensorPtr = CudaTensorPtr
        {
            tensor_ptr: ptr_to_string(grads_input_resizer),
            shape: vec![self.n_heads, self.seq_len, self.input_resized_len]
        };

        // for input resizer, the latent vector was multiplied, use broadcast multiply
        broadcast_2d_to_3d(
            grads_input_resizer_struct.get_ptr(), 
            self.n_heads, self.seq_len, self.input_resized_len,
            latent_resized_ptr, 
            1, 2, false
        );

        // for latent vector, the input resizer output was multiplied, use elementwise multiply
        element_op_3d_inplace(
            grads_latent, input_resizer_ptr, 
            2, 
            self.n_heads, self.seq_len, self.input_resized_len,
        );
        */

        // backpropagating through input resizer
        ////self.input_resizer_act_layer.backward(grads);
        //self.input_resizer_norm_layer.backward(grads);
        if self.input_embedding_len != self.latent_embedding_len
        {
            //self.input_resizer_act_layer.backward(grads);
            self.input_resizer_layer.backward(grads);
        }
        //self.input_resizer_pre_scaler_layer.backward(grads);
        //self.input_resizer_bottleneck_layer.backward(grads);

        //scalar_op_3d_inplace(
        //    grads.get_ptr(), self.n_heads as f32, 3, 
        //    self.n_heads, self.seq_len, self.input_embedding_len
        //);

        sum_axis(
            return_grads,
            grads.get_ptr(),
            self.n_heads, self.seq_len, self.input_embedding_len,
            0, false
        );
        
        // backpropagating through latent space branch
        // backpropagate through value layer
        let aggregator_grads_ptr: *mut f32 = self.aggregator_resizer_layer.get_weight_grads_as_ptr();
        let latent_space_shape: Vec<usize> = vec![self.n_heads, self.latent_embedding_len, self.input_embedding_len];
        
        let mut latent_space_struct: CudaTensorPtr = CudaTensorPtr
        {
            tensor_ptr: ptr_to_string(aggregator_grads_ptr),
            shape: latent_space_shape
        };

        // transpose gradients
        /*
        transpose_2d(
            value_modifier_grad_ptr, latent_space_struct.get_ptr(), 
            self.n_heads, self.latent_embedding_len, self.input_embedding_len
        );
        latent_space_struct.set_ptr(
            value_modifier_grad_ptr, 
            vec![self.n_heads, self.input_embedding_len, self.latent_embedding_len]
        );
        */

        //println!("{:?}", cuda_ptr_to_array(latent_space_struct.get_ptr(), latent_space_struct.get_shape()));

        //self.value_act_layer.backward(&mut latent_space_struct);
        //self.value_norm_layer.backward(&mut latent_space_struct);
        /*
        self.value_layer1.backward(&mut latent_space_struct);

        //println!("{:?}", cuda_ptr_to_array(latent_space_struct.get_ptr(), latent_space_struct.get_shape()));
        //exit(1);

        transpose_2d(
            latent_expanded_grad_ptr, latent_space_struct.get_ptr(), 
            self.n_heads, self.input_embedding_len, self.latent_embedding_len
        );
        latent_space_struct.set_ptr(
            latent_expanded_grad_ptr, 
            vec![self.n_heads, self.latent_embedding_len, self.input_embedding_len]
        );
        */
        //println!("{:?}", cuda_ptr_to_array(latent_space_struct.get_ptr(), latent_space_struct.get_shape()));
        //exit(1);
        //self.final_act_layer.backward(&mut latent_space_struct);
        //self.value_layer.backward(&mut latent_space_struct);
        self.latent_broadcast_layer.backward(&mut latent_space_struct);
        //println!("{:?}", cuda_ptr_to_array(latent_space_struct.get_ptr(), latent_space_struct.get_shape()));
        //exit(1);

        // transpose the latent grads to (1, resized_latent_len)
        transpose_2d(latent_grad_ptr, latent_space_struct.get_ptr(), self.n_heads, self.latent_embedding_len, 1);
        latent_space_struct.set_ptr(latent_grad_ptr, vec![self.n_heads, 1, self.latent_embedding_len]);
        //println!("{:?}", cuda_ptr_to_array(latent_space_struct.get_ptr(), latent_space_struct.get_shape()));
        self.latent_act_layer.backward(&mut latent_space_struct);
        ////self.dropout_layer.backward(&mut latent_space_struct, use_dropout);
        //self.scaler_layer.backward(&mut latent_space_struct);
        //println!("{:?}", cuda_ptr_to_array(latent_space_struct.get_ptr(), latent_space_struct.get_shape()));
        
        //self.latent_bottleneck_layer.backward(&mut latent_space_struct);
        //self.latent_norm_layer.backward(&mut latent_space_struct);

        //println!("{:?}", cuda_ptr_to_array(latent_space_struct.get_ptr(), latent_space_struct.get_shape()));
        //exit(1);

        // backpropagate through latent normalization layer

        //scalar_op_3d_inplace(
        //    latent_space_struct.get_ptr(), 
        //    1.0 / (self.seq_len as f32).sqrt(), 2, 
        //    self.n_heads, 1, self.latent_embedding_len
        //);
        //println!("{:?}", cuda_ptr_to_array(latent_space_struct.get_ptr(), latent_space_struct.get_shape()));

        // broadcast along axis 1, inverse of summing
        broadcast_2d_to_3d(
            latent_grad_broadcasted_ptr, 
            self.n_heads, self.seq_len, self.latent_embedding_len, 
            latent_space_struct.get_ptr(), 
            1, 0, true
        );

        latent_space_struct.set_ptr(
            latent_grad_broadcasted_ptr, 
            vec![self.n_heads, self.seq_len, self.latent_embedding_len]
        );

        //scalar_op_3d_inplace(
        //    latent_space_struct.get_ptr(), (self.seq_len as f32).sqrt(), 
        //    3, self.n_heads, self.seq_len, self.latent_embedding_len
        //);

        //println!("{:?}", cuda_ptr_to_array(latent_space_struct.get_ptr(), latent_space_struct.get_shape()));
        //exit(1);
        //self.navigator_output_layer.backward(&mut latent_space_struct);
        //scalar_op_3d_inplace(latent_space_struct.get_ptr(), 10.0, 0, self.n_heads, self.seq_len, self.latent_embedding_len);
        /* 
        zeroes_3d_inplace(
            string_to_ptr(&self.navigator_scaler_layer.weight_gradients_ptr), 
            self.navigator_scaler_layer.in_shape.0,
            self.navigator_scaler_layer.in_shape.1,
            self.navigator_scaler_layer.in_shape.2,
        );

        zeroes_3d_inplace(
            string_to_ptr(&self.navigator_scaler_layer.weight_gradients_temp_ptr), 
            self.navigator_scaler_layer.in_shape.0,
            self.navigator_scaler_layer.in_shape.1,
            self.navigator_scaler_layer.in_shape.2,
        );
        */
        //self.dropout_layer.backward(&mut latent_space_struct, use_dropout);
        self.navigator_scaler_layer.backward(&mut latent_space_struct, use_dropout);
        //self.navigator_pos_layer.backward(&mut latent_space_struct);
        //self.navigator_act_layer.backward(&mut latent_space_struct);
        //println!("{:?}", cuda_ptr_to_array(latent_space_struct.get_ptr(), &[self.n_heads, self.seq_len, self.latent_embedding_len]));
        //exit(1);
        //self.navigator_l2norm_layer.backward(&mut latent_space_struct);
        //scalar_op_3d_inplace(
        //    latent_space_struct.get_ptr(), 
        //    1.0 / (self.seq_len as f32).sqrt(), 2, 
        //    self.n_heads, self.seq_len, self.latent_embedding_len
        //);

        self.navigator_layer.backward(&mut latent_space_struct);
        //self.navigator_pre_scaler_layer.backward(&mut latent_space_struct);
        //self.navigator_bottleneck_layer.backward(&mut latent_space_struct);

        // backpropagate through latent modifier
        /*
        let latent_modifier_grad: *mut f32 = self.navigator_scaler_layer.get_weight_grads_as_ptr();
        let mut latent_modifier_grad_ptr: CudaTensorPtr = CudaTensorPtr
        {
            tensor_ptr: ptr_to_string(latent_modifier_grad),
            shape: vec![self.n_heads, self.seq_len, self.latent_embedding_len]
        };

        scalar_op_3d_inplace(
            latent_modifier_grad_ptr.get_ptr(), (self.seq_len as f32).sqrt(), 
            3, self.n_heads, self.seq_len, self.latent_embedding_len
        );

        //println!("{:?}", cuda_ptr_to_array(latent_space_struct.get_ptr(), latent_modifier_grad_ptr.get_shape()));
        //exit(1);

        //self.latent_modifier_scaler_layer.backward(&mut latent_modifier_grad_ptr);
        self.latent_modifier_act_layer.backward(&mut latent_modifier_grad_ptr);
        self.latent_modifier_layer.backward(&mut latent_modifier_grad_ptr);
        //println!("{:?}", cuda_ptr_to_array(latent_space_struct.get_ptr(), latent_modifier_grad_ptr.get_shape()));

        sum_axis(
            return_grads, latent_modifier_grad_ptr.get_ptr(), 
            self.n_heads, self.seq_len, self.input_embedding_len,
            0, false
        );
        */

        //println!("{:?}", cuda_ptr_to_array(latent_space_struct.get_ptr(), &[self.n_heads, self.seq_len, self.input_embedding_len]));
        //exit(1);
        //println!("{:?}", cuda_ptr_to_array(val_grads_struct.get_ptr(), val_grads_struct.get_shape()));

        //exit(1);
        // due to broadcasting along the z axis
        //scalar_op_3d_inplace(
        //    latent_space_struct.get_ptr(), self.n_heads as f32, 3, 
        //    self.n_heads, self.seq_len, self.input_embedding_len
        //);

        sum_axis(
            return_grads, 
            latent_space_struct.get_ptr(),
            self.n_heads, self.seq_len, self.input_embedding_len,
            0, false
        );

        //scalar_op_3d_inplace(
        //    return_grads, (self.n_heads * 2 + 1) as f32, 3, 
        //    1, self.seq_len, self.input_embedding_len
        //);

        //println!("{:?}", cuda_ptr_to_array(return_grads, &[1, self.seq_len, self.input_embedding_len]));
        //exit(1);
        // backpropagate through value matrix
        /*
        let val_grads_ptr: *mut f32 = self.aggregator_layer.get_weight_grads_as_ptr();
        //println!("{:?}", cuda_ptr_to_array(val_grads_ptr, &[self.n_heads, self.input_resized_len, self.input_resized_len]));
        transpose_2d(val_grads_t_ptr, val_grads_ptr, 
            self.n_heads, self.input_resized_len, self.input_resized_len
        );
        //println!("{:?}", cuda_ptr_to_array(val_grads_t_ptr, &[self.n_heads, self.input_resized_len, self.input_resized_len]));
        //exit(1);

        let mut val_grads_struct: CudaTensorPtr = CudaTensorPtr
        {
            tensor_ptr: ptr_to_string(val_grads_t_ptr),
            shape: vec![self.n_heads, self.input_resized_len, self.input_resized_len]
        };

        //println!("{:?}", cuda_ptr_to_array(val_grads_struct.get_ptr(), val_grads_struct.get_shape()));
        //println!("{:?}", cuda_ptr_to_array(val_grads_struct.get_ptr(), val_grads_struct.get_shape()));
        //exit(1);
        

        transpose_2d(
            return_grads_val,
            val_grads_struct.get_ptr(), 
            self.n_heads, self.input_resized_len, self.seq_len
        );

        val_grads_struct.set_ptr(return_grads_val, vec![self.n_heads, self.seq_len, self.input_resized_len]);
        self.value_bottleneck_layer.backward(&mut val_grads_struct);
        //println!("{:?}", cuda_ptr_to_array(val_grads_struct.get_ptr(), val_grads_struct.get_shape()));

        //println!("{:?}", cuda_ptr_to_array(return_grads, &[1, self.seq_len, self.input_embedding_len]));
        sum_axis(
            return_grads, 
            val_grads_struct.get_ptr(), 
            self.n_heads, self.seq_len, self.input_embedding_len,
            0, false
        );
        //println!("{:?}", cuda_ptr_to_array(return_grads, &[1, self.seq_len, self.input_embedding_len]));
        //exit(1);
        */

        grads.set_ptr(
            return_grads, 
            vec![1, self.seq_len, self.input_embedding_len]
        );

        //self.input_scaler_layer.backward(grads);
        //println!("{:?}", cuda_ptr_to_array(grads.get_ptr(), grads.get_shape()));
        //exit(1);
        //println!("{:?}", query_grads);

        // sum both grads for final gradients to output from head
        // shape (seq_len, input_embeddings)
        //println!("{:?}", final_grads);

        //return final_grads;
    }

    pub fn update_params(&mut self)
    {
        //self.input_scaler_layer.update_params();
        //self.navigator_bottleneck_layer.update_params(false);
        self.navigator_layer.update_params(false);
        //self.navigator_pre_scaler_layer.update_params();
        self.navigator_scaler_layer.update_params();
        //self.dropout_layer.update_params();
        //self.navigator_pos_layer.update_params();

        //self.latent_modifier_layer.update_params(false);
        //self.latent_modifier_scaler_layer.update_params();
        //self.latent_modifier_act_layer.update_params();
        //self.navigator_act_layer.update_params();
        //self.navigator_output_layer.update_params(false);

        //self.input_resizer_bottleneck_layer.update_params(false);
        if self.input_embedding_len != self.latent_embedding_len
        {
            self.input_resizer_layer.update_params(false);
            //self.input_resizer_act_layer.update_params();
        }
        //self.input_resizer_pre_scaler_layer.update_params();
        

        //self.aggregator_layer.update_params(false);

        //self.value_layer.update_params(false);
        //self.scaler_layer.update_params();
        //self.latent_act_layer.update_params();
        self.latent_broadcast_layer.update_params();
        //self.value_layer1.update_params(false);
        
        //self.head_out_dense_layer.update_params(false);
        //self.aggregator_act_layer.update_params();
        //self.value_norm_layer.update_params();
        //self.value_act_layer.update_params();
        //self.aggregator_norm_layer.update_params();
        //self.final_act_layer.update_params();
        self.aggregator_resizer_layer.update_params(true);
        //self.aggregator_scaler_layer.update_params();
        //self.value_bottleneck_layer.update_params(false);
        //self.latent_bottleneck_layer.update_params(false);

        if self.return_sequence
        {
            //self.extra_dense_layer.update_params(false);
            //self.extra_dense_layer1.update_params(false);
            //self.extra_dense_layer2.update_params(false);
            //self.extra_act_layer.update_params();
            //self.extra_act_layer1.update_params();
        }
        else
        {
            //self.extra_norm_layer.update_params();
        }
        //self.extra_norm_layer.update_params();
        //self.latent_norm_layer.update_params();
        //self.input_resizer_norm_layer.update_params();
        //self.output_embedding_norm_layer.update_params();
        //self.output_act_layer.update_params();
    }

    pub fn zero_grads(&mut self)
    {
        //self.key_layer.zero_grads();
        //self.key_norm.zero_grads();
        //self.key_layer_act.zero_grads();
        //self.query_layer.zero_grads();
        //self.query_norm.zero_grads();
        //self.query_layer_act.zero_grads();
        //self.attention_layer.zero_grads();
        //self.attention_norm.zero_grads();
        //self.attention_layer_act.zero_grads();
        //self.seq_input_layer.zero_grads();
        //self.seq_input_norm.zero_grads();
        //self.seq_input_layer_act.zero_grads();
        //self.seq_output_layer.zero_grads();
        //self.seq_output_norm.zero_grads();
        //self.seq_output_layer_act.zero_grads();
        //self.hidden_layer.zero_grads();
        //self.hidden_norm.zero_grads();
        //self.hidden_layer_act.zero_grads();
        //self.transform_layer.zero_grads();
        //self.transform_norm.zero_grads();
        //self.transform_layer_act.zero_grads();
    }

    pub fn details(&self)
    {
        //self.input_scaler_layer.details();
        self.navigator_layer.details();
        self.navigator_scaler_layer.details();
        if self.input_embedding_len != self.latent_embedding_len
        {
            self.input_resizer_layer.details();
        }
        //self.dropout_layer.details();
        //self.value_layer.details();
        self.latent_broadcast_layer.details();
        //self.value_act_layer.details();
        self.aggregator_resizer_layer.details();
        //self.aggregator_scaler_layer.details();
        //self.extra_norm_layer.details();
    }

    pub fn get_param_count(&self) -> usize
    {
        let mut count: usize = 0;
        //self.navigator_bottleneck_layer.move_ptrs_to_arrays(false);
        //count += self.input_scaler_layer.get_param_count();
        count += self.navigator_layer.get_param_count();
        count += self.navigator_scaler_layer.get_param_count();
        if self.input_embedding_len != self.latent_embedding_len
        {
            count += self.input_resizer_layer.get_param_count();
        }
        //count += ////self.navigator_pos_layer.get_param_count();

        count += self.latent_act_layer.get_param_count();
        count += self.latent_broadcast_layer.get_param_count();
        count += self.aggregator_act_layer.get_param_count();
        
        count += self.aggregator_resizer_layer.get_param_count();
        //count += //self.output_act_layer.get_param_count();
        //count += //self.aggregator_scaler_layer.get_param_count();

        return count;
    }

    pub fn move_ptrs_to_arrays(&mut self)
    {
        //self.navigator_bottleneck_layer.move_ptrs_to_arrays(false);
        //self.input_scaler_layer.move_ptrs_to_arrays();
        self.navigator_layer.move_ptrs_to_arrays(false);
        self.navigator_scaler_layer.move_ptrs_to_arrays();
        //self.dropout_layer.move_ptrs_to_arrays();
        //self.navigator_pos_layer.move_ptrs_to_arrays();
        //self.navigator_pre_scaler_layer.move_ptrs_to_arrays();
        //self.navigator_l2norm_layer.move_ptrs_to_arrays();

        //self.latent_modifier_layer.move_ptrs_to_arrays(false);
        //self.latent_modifier_scaler_layer.move_ptrs_to_arrays();
        //self.latent_modifier_act_layer.move_ptrs_to_arrays();

        //self.navigator_act_layer.move_ptrs_to_arrays();
        //self.navigator_output_layer.move_ptrs_to_arrays(false);

        //self.input_resizer_bottleneck_layer.move_ptrs_to_arrays(false);
        if self.input_embedding_len != self.latent_embedding_len
        {
            self.input_resizer_layer.move_ptrs_to_arrays(false);
            //self.input_resizer_act_layer.move_ptrs_to_arrays();
        }
        //self.input_resizer_pre_scaler_layer.move_ptrs_to_arrays();
        //self.input_resizer_norm_layer.move_ptrs_to_arrays();

        //self.aggregator_layer.move_ptrs_to_arrays(false);
        //self.value_layer.move_ptrs_to_arrays(false);
        //self.scaler_layer.move_ptrs_to_arrays();
        self.latent_act_layer.move_ptrs_to_arrays();
        self.latent_broadcast_layer.move_ptrs_to_arrays(false);
        //self.value_layer1.move_ptrs_to_arrays(false);
        //self.aggregator_norm_layer.move_ptrs_to_arrays();
        //self.head_out_dense_layer.move_ptrs_to_arrays(false);
        self.aggregator_act_layer.move_ptrs_to_arrays();
        //self.value_norm_layer.move_ptrs_to_arrays();
        //self.value_act_layer.move_ptrs_to_arrays();
        //self.final_act_layer.move_ptrs_to_arrays();
        
        self.aggregator_resizer_layer.move_ptrs_to_arrays(false);
        //self.aggregator_scaler_layer.move_ptrs_to_arrays();
        //self.value_bottleneck_layer.move_ptrs_to_arrays(false);
        //self.latent_bottleneck_layer.move_ptrs_to_arrays(false);
        //self.latent_norm_layer.move_ptrs_to_arrays();
        //self.output_embedding_norm_layer.move_ptrs_to_arrays();
        if self.return_sequence
        {
            //self.extra_dense_layer.move_ptrs_to_arrays(false);
            //self.extra_dense_layer1.move_ptrs_to_arrays(false);
            //self.extra_dense_layer2.move_ptrs_to_arrays(false);
            //self.extra_act_layer.move_ptrs_to_arrays();
            //self.extra_act_layer1.move_ptrs_to_arrays();
        }
        else
        {
            //self.extra_norm_layer.move_ptrs_to_arrays();
        }
        //self.extra_norm_layer.move_ptrs_to_arrays();
        //self.output_act_layer.move_ptrs_to_arrays();

        self.ptrs_allocated = false;
    }

    pub fn set_lr(&mut self, lr: f32)
    {
        //self.navigator_bottleneck_layer.lr = lr;
        self.navigator_layer.lr = lr;
        self.navigator_scaler_layer.lr = lr;

        self.latent_modifier_layer.lr = lr;
        //self.latent_modifier_act_layer.lr = lr;

        //self.navigator_output_layer.move_ptrs_to_arrays(false);

        //self.input_resizer_bottleneck_layer.lr = lr;
        if self.input_embedding_len != self.latent_embedding_len
        {
            self.input_resizer_layer.lr = lr;
        }
        //self.input_resizer_norm_layer.lr = lr;
        ////self.input_resizer_act_layer.lr = lr;

        //self.aggregator_layer.move_ptrs_to_arrays(false);
        //self.value_layer.lr = lr;
        self.latent_broadcast_layer.lr = lr;
        //self.value_layer1.lr = lr;
        //self.aggregator_norm_layer.lr = lr;
        //self.head_out_dense_layer.move_ptrs_to_arrays(false);
        //self.aggregator_act_layer.lr = lr;
        //self.value_norm_layer.lr = lr;
        //self.value_act_layer.lr = lr;
        //self.final_act_layer.move_ptrs_to_arrays();
        
        self.aggregator_resizer_layer.lr = lr;
        ////self.aggregator_scaler_layer.lr = lr;
        //self.value_bottleneck_layer.move_ptrs_to_arrays(false);
        //self.latent_bottleneck_layer.move_ptrs_to_arrays(false);
        //self.latent_norm_layer.lr = lr;
        //self.output_embedding_norm_layer.lr = lr;
        if self.return_sequence
        {
            //self.extra_dense_layer.lr = lr;
            //self.extra_dense_layer1.lr = lr;
            //self.extra_dense_layer2.move_ptrs_to_arrays(false);
            ////self.extra_act_layer.lr = lr;
            //self.extra_act_layer1.move_ptrs_to_arrays();
        }
        //self.extra_norm_layer.lr = lr;
        //self.output_act_layer.lr = lr;
    }

}