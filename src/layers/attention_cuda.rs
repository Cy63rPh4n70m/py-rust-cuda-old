
use serde::{Deserialize, Serialize};

use crate::{cuda_bridge::{broadcast_2d_to_3d, copy_cuda_to_cuda, scalar_op_3d_inplace, sum_axis, transpose_2d}, neuralnet::CudaTensorPtr, pointer_ops::{new_cuda_ptr_str, ptr_to_string, string_to_ptr}};

use super::{activation_cuda::ActivationCuda, dense_cuda::DenseCuda, layer_norm_cuda::LayerNormCuda};

#[derive(Serialize, Deserialize)]
pub struct SelfAttentionCuda
{
    key_layer: DenseCuda, key_norm: LayerNormCuda,
    query_layer: DenseCuda, query_norm: LayerNormCuda, query_proj_layer: DenseCuda,
    value_layer: DenseCuda, value_norm: LayerNormCuda, value_proj_layer: DenseCuda,

    attention_layer: DenseCuda, attention_layer_act: ActivationCuda, attention_norm: LayerNormCuda,
    transform_layer: DenseCuda, transform_layer_act: ActivationCuda,

    output_embedding_norm_layer: LayerNormCuda,

    
    //pub input_tensor: Array3<f32>,
    pub input_embedding_len: usize,
    pub output_embedding_len: usize,
    pub seq_len: usize,
    pub proj_len: usize,
    pub n_attention_heads: usize,
    pub mode: String, // "masked" or "unmasked"
    pub l2: f32, pub lr: f32, pub return_sequence: bool,

    //pub input_embedding_ptr: String,
    pub input_embedding_broadcasted_ptr: String,
    pub query_t_ptr: String,
    pub value_t_ptr: String,
    pub input_by_seq_ptr: String,
    pub output_embedding_ptr: String,
    pub output_embedding_vector_ptr: String,

    pub output_grads_broadcasted_ptr: String, // (1, seq_len, output)
    pub output_grads_broadcasted_head_ptr: String, // (heads, seq_len, output)
    pub seq_input_grads_ptr: String,
    pub query_proj_grad_t_ptr: String,
    pub query_grad_t_ptr: String,
    pub value_proj_grad_t_ptr: String,
    pub value_grad_t_ptr: String,
    pub return_grads: String, // (1, seq_len, input)

    pub ptrs_allocated: bool,
    pub sub_layers_allocated: bool,
}
impl SelfAttentionCuda
{
    pub fn new(
        n_attention_heads: usize, 
        input_embedding_len: usize,
        output_embedding_len: usize, 
        seq_len: usize,
        proj_len: usize,
        mode: &str, return_sequence: bool, l2: f32, lr: f32, append_scaler: f32,
        k_tiled: bool, q_tiled: bool, v_tiled: bool,
        attention_tiled: bool, transform_tiled: bool
        ) -> Self
    {
        let mut key_layer: DenseCuda = DenseCuda::new(output_embedding_len, l2, lr, k_tiled, "key");
        key_layer.use_bias = false;
        let key_norm: LayerNormCuda = LayerNormCuda::new(lr, l2, 2);
        //let key_layer_act: ActivationCuda = ActivationCuda::new("tanh", lr);

        let mut query_layer: DenseCuda = DenseCuda::new(output_embedding_len, l2, lr, q_tiled, "query");
        query_layer.use_bias = false;
        let mut query_proj_layer: DenseCuda = DenseCuda::new(proj_len, l2, lr, q_tiled, "query_proj");
        query_proj_layer.use_bias = false;
        let query_norm: LayerNormCuda = LayerNormCuda::new(lr, l2, 2);
        //let query_layer_act: ActivationCuda = ActivationCuda::new("tanh", lr);

        let mut value_layer: DenseCuda = DenseCuda::new(input_embedding_len, l2, lr, v_tiled, "value");
        value_layer.use_bias = false;
        let mut value_proj_layer: DenseCuda = DenseCuda::new(proj_len, l2, lr, q_tiled, "val_proj");
        value_proj_layer.use_bias = false;
        let value_norm: LayerNormCuda = LayerNormCuda::new(lr, l2, 2);
        //let value_layer_act: ActivationCuda = ActivationCuda::new("tanh", lr);

        //let kqv_layer: KQVCuda = KQVCuda::new(output_embedding_len, l2, lr);

        // compute attention scores
        let mut attention_layer: DenseCuda = DenseCuda::new(proj_len, l2, lr, attention_tiled, "attention");
        attention_layer.use_bias = false;
        let attention_norm: LayerNormCuda = LayerNormCuda::new(lr, l2, 2);
        let attention_layer_act: ActivationCuda = ActivationCuda::new("tanh", 1.0 / (proj_len as f32).sqrt());

        // matmul attention scores with value weights to create transform matrices
        // to append to input
        let mut transform_layer: DenseCuda = DenseCuda::new(input_embedding_len, l2, lr, transform_tiled, "transform");
        transform_layer.use_bias = false;
        //let transform_norm: LayerNormCuda = LayerNormCuda::new(lr, 2);
        let transform_layer_act: ActivationCuda = ActivationCuda::new("tanh", append_scaler);

        /*
        let seq_input_layer: DenseCuda = DenseCuda::new(input_embedding_len, l2, lr);
        let seq_input_norm: LayerNormCuda = LayerNormCuda::new(lr, 1);
        let seq_input_layer_act: ActivationCuda = ActivationCuda::new("silu", lr);

        let seq_output_layer: DenseCuda = DenseCuda::new(output_embedding_len, l2, lr);
        let seq_output_norm: LayerNormCuda = LayerNormCuda::new(lr, 1);
        let seq_output_layer_act: ActivationCuda = ActivationCuda::new("silu", lr);

        // initialise hidden layers
        let hidden_layer: DenseCuda = DenseCuda::new(input_embedding_len, l2, lr);
        let hidden_norm: LayerNormCuda = LayerNormCuda::new(lr, 2);
        let hidden_layer_act: ActivationCuda = ActivationCuda::new("silu", lr);

        // initialise transform layers
        let transform_layer: DenseCuda = DenseCuda::new(output_embedding_len, l2, lr);
        let transform_norm: LayerNormCuda = LayerNormCuda::new(lr, 2);
        let transform_layer_act: ActivationCuda = ActivationCuda::new("silu", lr);
        */

        let output_embedding_norm_layer: LayerNormCuda = LayerNormCuda::new(lr, l2, 2);
        //let extra_dense_layer: DenseCuda = DenseCuda::new(input_embedding_len, l2, lr);
        //let extra_activation_layer: ActivationCuda = ActivationCuda::new("silu", lr);

        // initialize the pointers
        //let input_embedding_ptr: String = new_cuda_ptr_str(&[1, seq_len, input_embedding_len]);
        let input_embedding_broadcasted_ptr: String = new_cuda_ptr_str(&[n_attention_heads, seq_len, input_embedding_len]);
        let input_by_seq_ptr: String = new_cuda_ptr_str(&[n_attention_heads, input_embedding_len, seq_len]);
        let output_embedding_ptr: String = new_cuda_ptr_str(&[1, seq_len, input_embedding_len]);
        let output_embedding_vector_ptr: String = new_cuda_ptr_str(&[1, 1, input_embedding_len]);

        let output_grads_broadcasted_ptr: String = new_cuda_ptr_str(&[1, seq_len, input_embedding_len]);
        let output_grads_broadcasted_head_ptr: String = new_cuda_ptr_str(&[n_attention_heads, seq_len, input_embedding_len]);
        let seq_input_grads_ptr: String = new_cuda_ptr_str(&[n_attention_heads, seq_len, input_embedding_len]);
        let query_grad_t_ptr: String = new_cuda_ptr_str(&[n_attention_heads, seq_len, output_embedding_len]);
        let return_grads: String = new_cuda_ptr_str(&[1, seq_len, input_embedding_len]);

        let self_attention: SelfAttentionCuda = Self
        {
            key_layer, key_norm,
            query_layer, query_norm, query_proj_layer,
            value_layer, value_norm, value_proj_layer,
            //kqv_layer,
            attention_layer, attention_layer_act, attention_norm,
            transform_layer, transform_layer_act,

            output_embedding_norm_layer,
            //extra_dense_layer, extra_activation_layer,

            //head_embeddings3d: Array3::zeros((n_attention_heads, seq_len, output_embedding_len)),

            //input_tensor: Array3::zeros((0, 0, 0)),
            input_embedding_len,
            output_embedding_len,
            n_attention_heads,
            proj_len,
            seq_len,
            mode: mode.to_string(),
            l2, lr, return_sequence,

            //input_embedding_ptr,
            input_embedding_broadcasted_ptr: "".to_string(),
            query_t_ptr: "".to_string(),
            value_t_ptr: "".to_string(),
            input_by_seq_ptr: "".to_string(),
            output_embedding_ptr: "".to_string(),
            output_embedding_vector_ptr: "".to_string(),

            // gradient pointers
            output_grads_broadcasted_ptr: "".to_string(),
            output_grads_broadcasted_head_ptr: "".to_string(),
            seq_input_grads_ptr: "".to_string(),
            query_proj_grad_t_ptr: "".to_string(),
            query_grad_t_ptr: "".to_string(),
            value_proj_grad_t_ptr: "".to_string(),
            value_grad_t_ptr: "".to_string(),
            return_grads: "".to_string(),

            ptrs_allocated: false,
            sub_layers_allocated: false,
        };

        return self_attention;
    }

    pub fn initialize_ptrs(&mut self)
    {
        self.input_embedding_broadcasted_ptr = new_cuda_ptr_str(&[self.n_attention_heads, self.seq_len, self.input_embedding_len]);
        self.input_by_seq_ptr = new_cuda_ptr_str(&[self.n_attention_heads, self.input_embedding_len, self.seq_len]);
        self.query_t_ptr = new_cuda_ptr_str(&[self.n_attention_heads, self.output_embedding_len, self.seq_len]);
        self.value_t_ptr = new_cuda_ptr_str(&[self.n_attention_heads, self.input_embedding_len, self.seq_len]);
        self.output_embedding_ptr = new_cuda_ptr_str(&[1, self.seq_len, self.input_embedding_len]);
        self.output_embedding_vector_ptr = new_cuda_ptr_str(&[1, 1, self.input_embedding_len]);

        self.output_grads_broadcasted_ptr = new_cuda_ptr_str(&[1, self.seq_len, self.input_embedding_len]);
        self.output_grads_broadcasted_head_ptr = new_cuda_ptr_str(&[self.n_attention_heads, self.seq_len, self.input_embedding_len]);
        self.seq_input_grads_ptr = new_cuda_ptr_str(&[self.n_attention_heads, self.seq_len, self.input_embedding_len]);
        self.query_grad_t_ptr = new_cuda_ptr_str(&[self.n_attention_heads, self.seq_len, self.output_embedding_len]);
        self.query_proj_grad_t_ptr = new_cuda_ptr_str(&[self.n_attention_heads, self.proj_len, self.output_embedding_len]);
        self.value_grad_t_ptr = new_cuda_ptr_str(&[self.n_attention_heads, self.seq_len, self.input_embedding_len]);
        self.value_proj_grad_t_ptr = new_cuda_ptr_str(&[self.n_attention_heads, self.input_embedding_len, self.proj_len]);
        self.return_grads = new_cuda_ptr_str(&[1, self.seq_len, self.input_embedding_len]);
    }

    // input_embeddings is 3D (1, seq_len, input_embedding_len)
    pub fn forward(&mut self, input_embeddings_struct: &mut CudaTensorPtr)
    {
        if !self.ptrs_allocated
        {
            self.initialize_ptrs();
            self.ptrs_allocated = true;
        }

        let input_embedding_broadcasted_ptr: *mut f32 = string_to_ptr(&self.input_embedding_broadcasted_ptr);
        let query_t_ptr: *mut f32 = string_to_ptr(&self.query_t_ptr);
        let value_t_ptr: *mut f32 = string_to_ptr(&self.value_t_ptr);
        let output_embedding_ptr: *mut f32 = string_to_ptr(&self.output_embedding_ptr);
        let output_embedding_vector_ptr: *mut f32 = string_to_ptr(&self.output_embedding_vector_ptr);

        //println!("{:?}", cuda_ptr_to_array(input_embeddings_struct.get_ptr(), input_embeddings_struct.get_shape()));
        // broadcast along z axis
        //let start = Instant::now();
        broadcast_2d_to_3d(
            input_embedding_broadcasted_ptr, 
            self.n_attention_heads, self.seq_len, self.input_embedding_len, 
            input_embeddings_struct.get_ptr(), 0, 0, true
        );
        //let end = start.elapsed();
        //println!("broadcast_2d_to_3d: {:?}", end.as_secs_f64());
        //println!("{:?}", cuda_ptr_to_array(input_embedding_broadcasted_ptr, &[self.n_attention_heads, self.seq_len, self.input_embedding_len]));

        //exit(1);
        // define pointer structs for passing through query and key layers
        // make a second copy for passing through hidden layer and transformation layer
        let mut key_ptr_struct: CudaTensorPtr = CudaTensorPtr {
            tensor_ptr: ptr_to_string(input_embedding_broadcasted_ptr),
            shape: vec![self.n_attention_heads, self.seq_len, self.input_embedding_len]
        };
        let mut query_ptr_struct: CudaTensorPtr = key_ptr_struct.clone();
        let mut value_ptr_struct: CudaTensorPtr = key_ptr_struct.clone();

        //let mut original_input_ptr_struct: CudaTensorPtr = key_ptr_struct.clone();

        // make a copy of the broadcast 
        //////////////////////////////////////////////////
        // batch matrix multiplication to create key and query embeddings

        //println!("{:?}", cuda_ptr_to_array(key_ptr_struct.get_ptr(), key_ptr_struct.get_shape()));
        //println!("{:?}", cuda_ptr_to_array(query_ptr_struct.get_ptr(), query_ptr_struct.get_shape()));
        //println!("{:?}", cuda_ptr_to_array(value_ptr_struct.get_ptr(), value_ptr_struct.get_shape()));
        //println!("---------------------------------");

        self.key_layer.forward(&mut key_ptr_struct);
        //println!("key: {:?}\n", cuda_ptr_to_array(key_ptr_struct.get_ptr(), key_ptr_struct.get_shape()));
        //self.key_norm.forward(&mut key_ptr_struct);
        //println!("key_norm: {:?}", cuda_ptr_to_array(key_ptr_struct.get_ptr(), key_ptr_struct.get_shape()));
        self.query_layer.forward(&mut query_ptr_struct);
        transpose_2d(query_t_ptr, query_ptr_struct.get_ptr(), self.n_attention_heads, self.seq_len, self.output_embedding_len);
        query_ptr_struct.set_ptr(query_t_ptr, 
            vec![self.n_attention_heads, self.output_embedding_len, self.seq_len]
        );
        self.query_proj_layer.forward(&mut query_ptr_struct);
        //exit(1);
        //self.query_norm.forward(&mut query_ptr_struct);
        //println!("query_proj: {:?}\n", cuda_ptr_to_array(query_ptr_struct.get_ptr(), query_ptr_struct.get_shape()));
        self.value_layer.forward(&mut value_ptr_struct);
        //println!("value: {:?}\n", cuda_ptr_to_array(value_ptr_struct.get_ptr(), value_ptr_struct.get_shape()));
        
        transpose_2d(value_t_ptr, value_ptr_struct.get_ptr(), self.n_attention_heads, self.seq_len, self.input_embedding_len);
        value_ptr_struct.set_ptr(value_t_ptr, 
            vec![self.n_attention_heads, self.input_embedding_len, self.seq_len]
        );
        self.value_proj_layer.forward(&mut value_ptr_struct);
        //println!("value_proj: {:?}\n", cuda_ptr_to_array(value_ptr_struct.get_ptr(), value_ptr_struct.get_shape()));
        //exit(1);
        //self.value_norm.forward(&mut value_ptr_struct);

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

        // add the transposed query embedding result as weights for 
        //let start = Instant::now();
        //println!("{:?}", key_ptr_struct.get_shape());
        //println!("{:?}", query_ptr_struct.get_shape());
        //println!("{:?}", value_ptr_struct.get_shape());
        //exit(1);
        let shape: &[usize] = query_ptr_struct.get_shape();
        self.attention_layer.set_weights_from_ptr(
            query_ptr_struct.get_ptr(), 
            (shape[0], shape[1], shape[2]), 
            false,
        );
        //println!("{:?}", cuda_ptr_to_array(string_to_ptr(&self.attention_layer.weight_ptr),  &[shape[0], shape[2], shape[1]]));

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
        self.attention_layer.forward(&mut key_ptr_struct);
        //println!("{:?}", cuda_ptr_to_array(key_ptr_struct.get_ptr(), key_ptr_struct.get_shape()));
        //exit(1);
        //self.attention_norm.forward(&mut key_ptr_struct);
        scalar_op_3d_inplace(
            key_ptr_struct.get_ptr(), 
            (self.output_embedding_len as f32).sqrt(), 3, 
            self.n_attention_heads, self.seq_len, self.proj_len
        );

        self.attention_layer_act.forward(&mut key_ptr_struct);
        //println!("{:?}", cuda_ptr_to_array(key_ptr_struct.get_ptr(), key_ptr_struct.get_shape()));
        //exit(1);

        // apply matrix multiplications to convert the (seq, seq) attention values
        // to a (input, output) transformation matrix
        // output is (seq, input)
        //println!("attention: {:?}\n", cuda_ptr_to_array(key_ptr_struct.get_ptr(), key_ptr_struct.get_shape()));
        
        // set the output of value layer as weights, attention scores as inputs
        // and calculate transform weights
        
        //let start = Instant::now();
        let val_embed_shape: &[usize] = value_ptr_struct.get_shape();
        //println!("{:?}", val_embed_shape);
        //println!("value_proj: {:?}\n", cuda_ptr_to_array(value_ptr_struct.get_ptr(), value_ptr_struct.get_shape()));
        //exit(1);
        self.transform_layer.set_weights_from_ptr(
            value_ptr_struct.get_ptr(), 
            (val_embed_shape[0], val_embed_shape[1], val_embed_shape[2]),
            true
        );
        
        //println!("value_proj_weight: {:?}\n", cuda_ptr_to_array(string_to_ptr(&self.transform_layer.weight_ptr), &[val_embed_shape[0], val_embed_shape[2], val_embed_shape[1]]));
        //exit(1);
        self.transform_layer.forward(&mut key_ptr_struct);
        //let end = start.elapsed();
        //println!("transform: {:?}", end.as_secs_f64());
        //println!("appender: {:?}\n", cuda_ptr_to_array(key_ptr_struct.get_ptr(), key_ptr_struct.get_shape()));
        //exit(1);
        //self.transform_norm.forward(&mut key_ptr_struct);
        //self.transform_layer_act.forward(&mut key_ptr_struct);
        //exit(1);

        //self.seq_input_layer.forward(&mut key_ptr_struct);
        //self.seq_input_norm.forward(&mut key_ptr_struct);
        //self.seq_input_layer_act.forward(&mut key_ptr_struct);
        //println!("{:?}", cuda_ptr_to_array(key_ptr_struct.get_ptr(), key_ptr_struct.get_shape()));
        //exit(1);
        
        // transpose to (input, seq) so it compatible with (seq, output)
        // and set new pointer (not freeing)
        //transpose_2d(
        //    input_by_seq_ptr, key_ptr_struct.get_ptr(), 
        //    self.n_attention_heads, self.seq_len, self.input_embedding_len
        //); 

        //key_ptr_struct.set_ptr(
        //    input_by_seq_ptr, 
        //    vec![self.n_attention_heads, self.input_embedding_len, self.seq_len]
        //);
        //println!("{:?}", cuda_ptr_to_array(key_ptr_struct.get_ptr(), key_ptr_struct.get_shape()));
        
        // final transformation matrix is (input, output)
        //let attention_vals_prev: ArrayD<f32> = attention_vals.clone();
        //self.seq_output_layer.forward(&mut key_ptr_struct);
        //self.seq_output_norm.forward(&mut key_ptr_struct);
        //self.seq_output_layer_act.forward(&mut key_ptr_struct);
        //println!("{:?}", cuda_ptr_to_array(key_ptr_struct.get_ptr(), key_ptr_struct.get_shape()));
        
        // set the output as the weights for the transform layer
        //let shape: &[usize] = key_ptr_struct.get_shape();
        //self.transform_layer.set_weights_from_ptr(
        //    key_ptr_struct.get_ptr(),
        //    (shape[0], shape[1], shape[2]),
        //    false
        //);
        //println!("{:?}", cuda_ptr_to_array(string_to_ptr(&self.transform_layer.weight_ptr), &[shape[0], shape[1], shape[2]]));
        //println!("{:?}", (shape[0], shape[1], shape[2]));
        //println!("{:?}, {:?}, {:?}", self.n_attention_heads, self.input_embedding_len, self.output_embedding_len);
        //exit(1);

        // using the original input embeddings
        // pass through hidden layer first
        //println!("{:?}", cuda_ptr_to_array(original_input_ptr_struct.get_ptr(), original_input_ptr_struct.get_shape()));
        //self.hidden_layer.forward(&mut original_input_ptr_struct);
        //self.hidden_norm.forward(&mut original_input_ptr_struct);
        //self.hidden_layer_act.forward(&mut original_input_ptr_struct);
        //println!("{:?}", cuda_ptr_to_array(original_input_ptr_struct.get_ptr(), original_input_ptr_struct.get_shape()));

        // apply the transformation matrix
        //self.transform_layer.forward(&mut original_input_ptr_struct);
        //self.transform_norm.forward(&mut original_input_ptr_struct);
        //self.transform_layer_act.forward(&mut original_input_ptr_struct);
        //println!("{:?}", cuda_ptr_to_array(original_input_ptr_struct.get_ptr(), original_input_ptr_struct.get_shape()));
        
        // sum output embeddings along first dimension (batch/attention heads)
        //println!("{:?}", cuda_ptr_to_array(key_ptr_struct.get_ptr(), key_ptr_struct.get_shape()));
        //let result_ptr: *mut f32 = array_to_cuda_ptr(&mut Array3::ones((1, self.seq_len, self.input_embedding_len)).into_dyn());
        //let start: Instant = Instant::now();

        //zeroes_3d_inplace(output_embedding_ptr, 1, self.seq_len, self.input_embedding_len);

        //let start = Instant::now();        
        sum_axis(
            output_embedding_ptr, key_ptr_struct.get_ptr(), 
            self.n_attention_heads, self.seq_len, self.input_embedding_len, 
            0, true
        );

        let mut head_sum_ptr_struct: CudaTensorPtr = CudaTensorPtr
        {
            tensor_ptr: ptr_to_string(output_embedding_ptr),
            shape: vec![1, self.seq_len, self.input_embedding_len]
        };

        self.transform_layer_act.forward(&mut head_sum_ptr_struct);

        //println!("{:?}", cuda_ptr_to_array(output_embedding_ptr, &[1, self.seq_len, self.input_embedding_len]));
        //exit(1);
        //println!("{:?}", cuda_ptr_to_array(input_embeddings_struct.get_ptr(), input_embeddings_struct.get_shape()));
        
        // add the original input embeddings (transforms the input embedding)
        sum_axis(
            head_sum_ptr_struct.get_ptr(), 
            input_embeddings_struct.get_ptr(), 
            1, self.seq_len, self.input_embedding_len, 
            0, false
        );
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
            head_sum_ptr_struct.get_ptr(), 
            vec![1, self.seq_len, self.input_embedding_len]
        );

        //let start = Instant::now(); 
        if self.return_sequence
        {
            //input_embeddings_struct.set_ptr(
            //    output_embedding_ptr, 
            //    vec![1, self.seq_len, self.input_embedding_len]
            //);
        //    output_embeddings = output_embeddings.mean_axis(Axis(1)).unwrap();
        //    output_embeddings = output_embeddings.broadcast((1, self.output_embedding_len)).unwrap().into_dyn().into_owned();
        }
        else
        {
            // get last token ptr
            let last_embedding_ptr: *mut f32 = unsafe { input_embeddings_struct.get_ptr().add((self.seq_len - 1) * self.input_embedding_len)};
            copy_cuda_to_cuda(
                output_embedding_vector_ptr, last_embedding_ptr, 
                &[1, 1, self.input_embedding_len]
            );
            input_embeddings_struct.set_ptr(
                output_embedding_vector_ptr, 
                vec![1, 1, self.input_embedding_len]
            );
        }

        //self.output_embedding_norm_layer.forward(input_embeddings_struct);
        //let end = start.elapsed();
        //println!("sum_and_norm: {:?}", end.as_secs_f64());

        //println!("{:?}", cuda_ptr_to_array(input_embeddings_struct.get_ptr(), input_embeddings_struct.get_shape()));
        //exit(1);
    }

    pub fn backward(&mut self, grads: &mut CudaTensorPtr)
    {
        let output_grads_broadcasted_ptr: *mut f32 = string_to_ptr(&self.output_grads_broadcasted_ptr);
        let output_grads_broadcasted_head_ptr: *mut f32 = string_to_ptr(&self.output_grads_broadcasted_head_ptr);
        let query_grad_t_ptr: *mut f32 = string_to_ptr(&self.query_grad_t_ptr);
        let query_proj_grad_t_ptr: *mut f32 = string_to_ptr(&self.query_proj_grad_t_ptr);
        let value_grad_t_ptr: *mut f32 = string_to_ptr(&self.value_grad_t_ptr);
        let value_proj_grad_t_ptr: *mut f32 = string_to_ptr(&self.value_proj_grad_t_ptr);
        let return_grads: *mut f32 = string_to_ptr(&self.return_grads);
        
        //self.output_embedding_norm_layer.backward(grads);

        //println!("{:?}", cuda_ptr_to_array(grads.get_ptr(), grads.get_shape()));
        if self.return_sequence
        {
            // already (1, seq_len, output_embedding)
            // pass
        }
        else
        {
            // the mean of embeddings was returned instead
            // (1, 1, output_embedding)

            //backpropagate through mean 
            // mean =  embeddings / y_length
            // derivative mean = 1 / y_length

            let last_ptr: *mut f32 = unsafe { output_grads_broadcasted_ptr.add((self.seq_len - 1) * self.input_embedding_len) };
            copy_cuda_to_cuda(last_ptr, grads.get_ptr(), &[1, 1, self.input_embedding_len]);
            grads.set_ptr(output_grads_broadcasted_ptr, vec![1, self.seq_len, self.input_embedding_len]);
        }

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

        self.transform_layer_act.backward(grads);

        //println!("{:?}", cuda_ptr_to_array(return_grads, &[1, self.seq_len, self.input_embedding_len]));
        //println!("{:?}", cuda_ptr_to_array(grads.get_ptr(), grads.get_shape()));
        //exit(1);

        // backpropagate through summation along first axis
        // broadcast along z axis, inverse of concatenating heads
        broadcast_2d_to_3d(
            output_grads_broadcasted_head_ptr, 
            self.n_attention_heads, self.seq_len, self.input_embedding_len, 
            grads.get_ptr(), 0, 0, true
        );
        
        grads.set_ptr(
            output_grads_broadcasted_head_ptr, 
            vec![self.n_attention_heads, self.seq_len, self.input_embedding_len]
        );

        //println!("{:?}", cuda_ptr_to_array(grads.get_ptr(), grads.get_shape()));

        // backpropagate through transform layer (attention vals * val embedding)
        self.transform_layer.backward(grads); // grads is now at attention layer

        //println!("\x1b[31mattention_matrix: {:?}\x1b[0m", cuda_ptr_to_array(grads.get_ptr(), grads.get_shape()));
        //exit(1);

        // get value grads
        let val_grads: *mut f32 = self.transform_layer.get_weight_grads_as_ptr();
        let val_grads_shape: Vec<usize> = vec![self.n_attention_heads, self.proj_len, self.input_embedding_len];
        
        transpose_2d(value_proj_grad_t_ptr, val_grads, self.n_attention_heads, self.proj_len, self.input_embedding_len);
        let mut val_grads_struct: CudaTensorPtr = CudaTensorPtr
        {
            tensor_ptr: ptr_to_string(value_proj_grad_t_ptr),
            shape: vec![self.n_attention_heads, self.input_embedding_len, self.proj_len]
        };

        self.value_proj_layer.backward(&mut val_grads_struct);
        //println!("{:?}", cuda_ptr_to_array(val_grads_struct.get_ptr(), val_grads_struct.get_shape()));
        transpose_2d(value_grad_t_ptr, val_grads_struct.get_ptr(), self.n_attention_heads, self.input_embedding_len, self.seq_len);
        val_grads_struct.set_ptr(
            value_grad_t_ptr, 
            vec![self.n_attention_heads, self.seq_len, self.input_embedding_len]
        );
        //println!("{:?}", cuda_ptr_to_array(val_grads_struct.get_ptr(), val_grads_struct.get_shape()));
        //exit(1);
        // backpropagate through value layer
        //self.value_norm.backward(&mut val_grads_struct);
        self.value_layer.backward(&mut val_grads_struct);
        //println!("{:?}", cuda_ptr_to_array(val_grads_struct.get_ptr(), val_grads_struct.get_shape()));
        //exit(1);
        //println!("\x1b[31mattention_matrix: {:?}\x1b[0m", cuda_ptr_to_array(grads.get_ptr(), grads.get_shape()));

        //exit(1);
        // due to broadcasting along the z axis
        sum_axis(
            return_grads, 
            val_grads_struct.get_ptr(),
            self.n_attention_heads, self.seq_len, self.input_embedding_len,
            0, false
        );

        //println!("{:?}", cuda_ptr_to_array(val_grads, &[self.transform_layer.out_shape.0, self.transform_layer.out_shape.1, self.transform_layer.out_shape.2]));
        //println!("{:?}", cuda_ptr_to_array(val_grads, &[self.kqv_layer.v_out_shape.0, self.kqv_layer.v_out_shape.1, self.kqv_layer.v_out_shape.2]));
        //exit(1);

        //println!("{:?}", cuda_ptr_to_array(return_grads, &[1, self.seq_len, self.input_embedding_len]));
        //exit(1);

        //println!("{:?}", cuda_ptr_to_array(grads.get_ptr(), grads.get_shape()));
        //exit(1);

        // final embeddings was created by summing each output embedding produced from each attention
        // head, derivative for dy/(dx,dy,dz) in y = x + w + z is 1, thus use the same gradients for
        // each head
        
        ////////////////////////////////////////////////
        // backpropagate through transform and hidden layers first
        //println!("{:?}", cuda_ptr_to_array(grads.get_ptr(), grads.get_shape()));

        // sum along first axis as the inputs was original input embeddings (1, seq_len, in_embeddings)
        // was broadcasted
        //println!("return_grads: {:?}", cuda_ptr_to_array(return_grads, &[1, self.seq_len, self.input_embedding_len]));
        //unsafe
        //{
        //    sum_axis(
        //        return_grads, 
        //        grads.get_ptr(),
        //        self.n_attention_heads, self.seq_len, self.input_embedding_len,
        //        0, true
        //    );
        //}
        //println!("return_grads: {:?}", cuda_ptr_to_array(return_grads, &[1, self.seq_len, self.input_embedding_len]));

        // use weight gradients from the transform layer to backpropagate through
        // seq_output, seq_input, and key/query matrices branch
        //let transform_weight_grads: *mut f32 = self.transform_layer.get_weight_grads_as_ptr();
        //grads.set_ptr(
        //    transform_weight_grads, 
        //    vec![self.n_attention_heads, self.input_embedding_len, self.output_embedding_len]
        //);
        //println!("{:?}", cuda_ptr_to_array(grads.get_ptr(), grads.get_shape()));

        // grads is (input_embedding, output_embedding)
        // backpropagate through seq_output activation and matrix
        //self.seq_output_layer_act.backward(grads);
        //self.seq_output_norm.backward(grads);
        //self.seq_output_layer.backward(grads);

        // batch transpose the grads from (input, seq) to (seq, input)
        //transpose_2d(
        //    seq_input_grads_ptr, grads.get_ptr(), 
        //    self.n_attention_heads, self.input_embedding_len, self.seq_len
        //);

        //grads.set_ptr(seq_input_grads_ptr, vec![self.n_attention_heads, self.seq_len, self.input_embedding_len]);

        //println!("{:?}", cuda_ptr_to_array(seq_input_grads_ptr, &[self.n_attention_heads, self.seq_len, self.input_embedding_len]));
        //println!("{:?}", cuda_ptr_to_array(grads.get_ptr(), grads.get_shape()));
        
        // gradients for key/query matrix
        self.attention_layer_act.backward(grads);
        //self.attention_norm.backward(grads);
        scalar_op_3d_inplace(
            grads.get_ptr(), (self.output_embedding_len as f32).sqrt(), 
            3, 
            self.n_attention_heads, self.seq_len, self.proj_len
        );
        self.attention_layer.backward(grads); // key matrix
        //println!("{:?}", cuda_ptr_to_array(grads.get_ptr(), grads.get_shape()));

        let query_grads: *mut f32 = self.attention_layer.get_weight_grads_as_ptr();
        //println!("{:?}", cuda_ptr_to_array(query_grads, &[self.n_attention_heads, self.output_embedding_len, self.proj_len]));
        //exit(1);

        // batch transpose query gradients
        //transpose_2d(
        //    query_proj_grad_t_ptr, query_grads, 
        //    self.n_attention_heads, self.output_embedding_len, self.proj_len
        //);
        let mut query_grad_struct: CudaTensorPtr = CudaTensorPtr
        {
            tensor_ptr: ptr_to_string(query_grads),
            shape: vec![self.n_attention_heads, self.output_embedding_len, self.proj_len]
        };

        self.query_proj_layer.backward(&mut query_grad_struct);
        //println!("{:?}", cuda_ptr_to_array(query_grad_struct.get_ptr(), query_grad_struct.get_shape()));
        //exit(1);
        transpose_2d(
            query_grad_t_ptr, query_grad_struct.get_ptr(), 
            self.n_attention_heads, self.output_embedding_len, self.seq_len
        );
        query_grad_struct.set_ptr(query_grad_t_ptr, vec![self.n_attention_heads, self.seq_len, self.output_embedding_len]);
        //println!("{:?}", cuda_ptr_to_array(query_grad_struct.get_ptr(), query_grad_struct.get_shape()));
        //exit(1);
        //self.key_norm.backward(grads);
        self.key_layer.backward(grads);
        //println!("{:?}", cuda_ptr_to_array(grads.get_ptr(), grads.get_shape()));
        //self.query_norm.backward(&mut query_grad_struct);
        self.query_layer.backward(&mut query_grad_struct);
        //println!("{:?}", cuda_ptr_to_array(query_grad_struct.get_ptr(), query_grad_struct.get_shape()));
        //exit(1);
        //println!("{:?}", cuda_ptr_to_array(grads.get_ptr(), &[self.n_attention_heads, self.seq_len, self.input_embedding_len]));
        //println!("{:?}", cuda_ptr_to_array(query_grad_struct.get_ptr(), &[self.n_attention_heads, self.seq_len, self.input_embedding_len]));

        sum_axis(
            return_grads, 
            grads.get_ptr(),
            self.n_attention_heads, self.seq_len, self.input_embedding_len,
            0, false
        );

        sum_axis(
            return_grads, 
            query_grad_struct.get_ptr(),
            self.n_attention_heads, self.seq_len, self.input_embedding_len,
            0, false
        );


        // create another CudaTensorPtr for query grads
        //println!("{:?}", cuda_ptr_to_array(query_grads.get_ptr(), query_grads.get_shape()));
        //exit(1);

        // backpropagate through key layer activation and weights
        //println!("{:?}", self.key_layer.weight_gradients);
        //println!("key_grad: {:?}", cuda_ptr_to_array(grads.get_ptr(), grads.get_shape()));
        //println!("{:?}", cuda_ptr_to_array(grads.get_ptr(), grads.get_shape()));

        //println!("return_grads: {:?}", cuda_ptr_to_array(return_grads, &[1, self.seq_len, self.input_embedding_len]));
        // sum along z axis and add to return grads
        //println!("return_grads: {:?}", cuda_ptr_to_array(return_grads, &[1, self.seq_len, self.input_embedding_len]));
        //println!("{:?}", self.key_layer.weight_gradients);
        //println!("{:?}", key_grads);

        // backpropagate through query layer activation and weights
        //println!("{:?}", cuda_ptr_to_array(query_grads.get_ptr(), query_grads.get_shape()));
        //println!("{:?}", cuda_ptr_to_array(query_grads.get_ptr(), query_grads.get_shape()));
        //exit(1);

        //println!("return_grads: {:?}", cuda_ptr_to_array(return_grads, &[1, self.seq_len, self.input_embedding_len]));
        //exit(1);
        // sum along z axis and add to return grads

        grads.set_ptr(
            return_grads, 
            vec![1, self.seq_len, self.input_embedding_len]
        );
        //println!("return_grads: {:?}", cuda_ptr_to_array(grads.get_ptr(), grads.get_shape()));
        //exit(1);
        //println!("{:?}", query_grads);

        // sum both grads for final gradients to output from head
        // shape (seq_len, input_embeddings)
        //println!("{:?}", final_grads);

        //return final_grads;
    }

    pub fn update_params(&mut self)
    {
        self.key_layer.update_params(false);
        //self.key_norm.update_params();
        //self.key_layer_act.update_params();

        self.query_layer.update_params(false);
        self.query_proj_layer.update_params(false);
        //self.query_norm.update_params();
        //self.query_layer_act.update_params();

        self.value_layer.update_params(false);
        self.value_proj_layer.update_params(false);
        //self.value_norm.update_params();
        //self.value_layer_act.update_params();
        //self.kqv_layer.update_params();
        
        self.attention_layer.update_params(false);
        //self.attention_norm.update_params();
        self.attention_layer_act.update_params();

        self.transform_layer.update_params(false);
        self.transform_layer_act.update_params();

        //self.output_embedding_norm_layer.update_params();
    }

    pub fn get_param_count(&self) -> usize
    {
        let mut count: usize = 0;
        count += self.key_layer.get_param_count();
        //self.key_norm.update_params();
        //self.key_layer_act.update_params();

        count += self.query_layer.get_param_count();
        count += self.query_proj_layer.get_param_count();
        //self.query_norm.update_params();
        //self.query_layer_act.update_params();

        count += self.value_layer.get_param_count();
        count += self.value_proj_layer.get_param_count();
        //self.value_norm.update_params();
        //self.value_layer_act.update_params();
        //self.kqv_layer.update_params();
        
        //count += self.attention_layer.get_param_count();
        //count += //self.attention_norm.get_param_count();
        count += self.attention_layer_act.get_param_count();

        count += self.transform_layer.get_param_count();
        count += self.transform_layer_act.get_param_count();

        //count += //self.output_embedding_norm_layer.get_param_count();

        return count;
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
        self.key_layer.details();

        self.query_layer.details();
        self.query_proj_layer.details();

        self.value_layer.details();
        self.value_proj_layer.details();

        self.attention_layer.details();
        self.attention_layer_act.details();

        self.transform_layer.details();
        self.transform_layer_act.details();
    }

    pub fn move_ptrs_to_arrays(&mut self)
    {
        self.key_layer.move_ptrs_to_arrays(false);
        //self.key_norm.move_ptrs_to_arrays();
        self.query_layer.move_ptrs_to_arrays(false);
        self.query_proj_layer.move_ptrs_to_arrays(false);
        //self.query_norm.move_ptrs_to_arrays();
        self.value_layer.move_ptrs_to_arrays(false);
        self.value_proj_layer.move_ptrs_to_arrays(false);
        //self.value_norm.move_ptrs_to_arrays();

        self.attention_layer.move_ptrs_to_arrays(true);
        //self.attention_norm.move_ptrs_to_arrays();
        self.attention_layer_act.move_ptrs_to_arrays();

        self.transform_layer.move_ptrs_to_arrays(true);
        self.transform_layer_act.move_ptrs_to_arrays();
        //self.output_embedding_norm_layer.move_ptrs_to_arrays();

        self.ptrs_allocated = false;
    }

}