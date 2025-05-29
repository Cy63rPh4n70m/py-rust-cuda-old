
use ndarray::{Array3, ArrayD, ArrayView3, Axis, Ix4, IxDyn};
use serde::{Deserialize, Serialize};

use super::{activation::Activation, dense_for_attention::DenseForAttention, layer_norm::NormLayer};

#[derive(Serialize, Deserialize)]
pub struct SelfAttention
{
    key_layer: DenseForAttention, key_layer_act: Activation, key_norm: NormLayer,
    query_layer: DenseForAttention, query_layer_act: Activation, query_norm: NormLayer,
    attention_layer: DenseForAttention, attention_layer_act: Activation, //attention_norm: NormLayer,

    key_mat: Array3<f32>, query_mat: Array3<f32>,

    // used to convert/process the (n, n) attention matrix into a (input, output) matrix
    seq_input_layer: DenseForAttention, seq_input_layer_act: Activation, seq_input_norm: NormLayer,
    seq_output_layer: DenseForAttention, seq_output_layer_act: Activation, seq_output_norm: NormLayer,

    // the layers that transform the input embeddings with different 
    transform_layer: DenseForAttention, transform_layer_act: Activation, transform_norm: NormLayer,
    hidden_layer: DenseForAttention, hidden_layer_act: Activation, hidden_norm: NormLayer,
    
    // holds the transform matrices produced by each attention head
    // all summed along first axis to produce a single transformation matrix
    pub head_embeddings3d: Array3<f32>,
    
    pub input_tensor: Array3<f32>,
    pub input_embedding_len: usize,
    pub output_embedding_len: usize,
    pub seq_len: usize,
    pub n_attention_heads: usize,
    pub mode: String, // "masked" or "unmasked"
    pub l2: f32, pub lr: f32, pub return_sequence: bool
}
impl SelfAttention
{
    pub fn new(
        n_attention_heads: usize, 
        input_embedding_len: usize,
        output_embedding_len: usize, 
        seq_len: usize,
        mode: &str, return_sequence: bool, l2: f32, lr: f32) -> Self
    {
        let key_layer: DenseForAttention = DenseForAttention::new(n_attention_heads, output_embedding_len, l2, lr);
        let key_norm: NormLayer = NormLayer::new(lr, 2);
        let key_layer_act: Activation = Activation::new("silu", lr);

        let query_layer: DenseForAttention = DenseForAttention::new(n_attention_heads, output_embedding_len, l2, lr);
        let query_norm: NormLayer = NormLayer::new(lr, 2);
        let query_layer_act: Activation = Activation::new("silu", lr);

        let attention_layer: DenseForAttention = DenseForAttention::new(n_attention_heads, seq_len, l2, lr);
        //let attention_norm: NormLayer = NormLayer::new(lr, 1);
        let attention_layer_act: Activation = Activation::new("silu", lr);

        let seq_input_layer: DenseForAttention = DenseForAttention::new(n_attention_heads, input_embedding_len, l2, lr);
        let seq_input_norm: NormLayer = NormLayer::new(lr, 2);
        let seq_input_layer_act: Activation = Activation::new("silu", lr);
        //let mut seq_input_residual: Dense = Dense::new(input_embedding, l2, lr);
        //seq_input_residual.set_weights(Array2::ones((seq_len, input_embedding)));

        let seq_output_layer: DenseForAttention = DenseForAttention::new(n_attention_heads, output_embedding_len, l2, lr);
        let seq_output_norm: NormLayer = NormLayer::new(lr, 0);
        let seq_output_layer_act: Activation = Activation::new("silu", lr);
        //let mut seq_output_residual: Dense = Dense::new(output_embedding, l2, lr);
        //seq_output_residual.set_weights(Array2::ones((seq_len, output_embedding)));

        // initialise hidden layers
        let hidden_layer: DenseForAttention = DenseForAttention::new(n_attention_heads, input_embedding_len, l2, lr);
        let hidden_norm: NormLayer = NormLayer::new(lr, 1);
        let hidden_layer_act: Activation = Activation::new("silu", lr);

        // initialise transform layers
        let transform_layer: DenseForAttention = DenseForAttention::new(n_attention_heads, output_embedding_len, l2, lr);
        let transform_norm: NormLayer = NormLayer::new(lr, 1);
        let transform_layer_act: Activation = Activation::new("silu", lr);
        //let mut transform_residual: Dense = Dense::new(output_embedding, l2, lr);
        //transform_residual.set_weights(Array2::ones((input_embedding, output_embedding)));

        return Self
        {
            key_layer, key_layer_act, key_norm,
            query_layer, query_layer_act, query_norm,
            attention_layer, attention_layer_act,
            key_mat: Array3::zeros((0, 0, 0)),
            query_mat: Array3::zeros((0, 0, 0)),
            seq_input_layer, seq_input_layer_act, seq_input_norm,
            seq_output_layer, seq_output_layer_act, seq_output_norm,
            hidden_layer, hidden_layer_act, hidden_norm,
            transform_layer, transform_layer_act, transform_norm,

            head_embeddings3d: Array3::zeros((n_attention_heads, seq_len, output_embedding_len)),

            input_tensor: Array3::zeros((0, 0, 0)),
            input_embedding_len,
            output_embedding_len,
            n_attention_heads,
            seq_len,
            mode: mode.to_string(),
            l2, lr, return_sequence
        };
    }

    // input_embeddings is 3D (1, seq_len, input_embedding_len)
    pub fn forward(&mut self, input_embeddings: ArrayD<f32>) -> ArrayD<f32>
    {
        
        // zero 3d output embedding matrices buffers
        self.head_embeddings3d.fill(0.0);

        //////////////////////////////////////////////////
        // batch matrix multiplication to create key and query embeddings
        let mut key_mat: ArrayD<f32> = self.key_layer.forward(input_embeddings.clone());
        //key_mat = self.key_norm.forward(key_mat);
        key_mat = self.key_layer_act.forward(key_mat);

        let mut query_mat: ArrayD<f32> = self.query_layer.forward(input_embeddings.clone());
        //query_mat = self.query_norm.forward(query_mat);
        query_mat = self.query_layer_act.forward(query_mat);

        // batch transpose and add axis for attention value computation
        let mut query_mat_t: ArrayD<f32> = query_mat.permuted_axes(IxDyn(&[0, 2, 1]));
        query_mat_t = query_mat_t.insert_axis(Axis(1));

        self.attention_layer.set_weights(
            query_mat_t.into_dimensionality::<Ix4>().unwrap()
        );

        // calculate attention scores and record key/query matrices
        let mut attention_vals: ArrayD<f32> = self.attention_layer.forward(key_mat.clone());
        //attention_vals = self.attention_norm.forward(attention_vals);
        attention_vals = self.attention_layer_act.forward(attention_vals);

        // apply matrix multiplications to convert the (seq, seq) attention values
        // to a (input, output) transformation matrix
        attention_vals = self.seq_input_layer.forward(attention_vals);
        //attention_vals = self.seq_input_norm.forward(attention_vals);
        attention_vals = self.seq_input_layer_act.forward(attention_vals);
        
        // batch transpose to (input, seq) so it compatible with (seq, output)
        attention_vals = attention_vals.permuted_axes(IxDyn(&[0, 2, 1]));
        
        // final transformation matrix is (input, output)
        //let attention_vals_prev: ArrayD<f32> = attention_vals.clone();
        attention_vals = self.seq_output_layer.forward(attention_vals);
        //attention_vals = self.seq_output_norm.forward(attention_vals);
        attention_vals = self.seq_output_layer_act.forward(attention_vals);
        
        // insert axis to make calculation compatible with 4D array
        attention_vals = attention_vals.insert_axis(Axis(1));

        // set transform layer weights
        self.transform_layer.set_weights(
            attention_vals.into_dimensionality::<Ix4>().unwrap()
        );

        // pass through hidden layer first
        let mut hidden: ArrayD<f32> = self.hidden_layer.forward(input_embeddings.clone());
        //hidden = self.hidden_norm.forward(hidden);
        hidden = self.hidden_layer_act.forward(hidden);
        //let mut residual: ArrayD<f32> = self.hidden_residual.forward(input_embedding);
        //hidden += &residual;

        // apply the transformation matrix
        //let hidden_prev: ArrayD<f32> = hidden.clone();
        let mut output_embeddings: ArrayD<f32> = self.transform_layer.forward(hidden);
        //output_embeddings = self.transform_norm.forward(output_embeddings);
        output_embeddings = self.transform_layer_act.forward(output_embeddings);
        self.head_embeddings3d = output_embeddings.into_dimensionality().unwrap();
        //////////////////////////////////////////////////
        
        // sum output embeddings along first dimension (batch/attention heads)
        let embeddings_sum: Array3<f32> = (self.head_embeddings3d.sum_axis(Axis(0))).insert_axis(Axis(0));

        let mut output_embeddings: ArrayD<f32> = embeddings_sum.into_dyn();
        
        let input_tensors3d: Array3<f32> = input_embeddings.into_dimensionality().unwrap();
        self.input_tensor = input_tensors3d;

        if !self.return_sequence
        {
            output_embeddings = output_embeddings.mean_axis(Axis(1)).unwrap();
            output_embeddings = output_embeddings.broadcast((1, self.output_embedding_len)).unwrap().into_dyn().into_owned();
        }

        return output_embeddings;
    }

    pub fn backward(&mut self, mut grads: ArrayD<f32>) -> ArrayD<f32>
    {
        // the mean of embeddings was returned instead
        // (1, 1, output_embedding)
        if !self.return_sequence
        {
            //backpropagate through mean 
            // mean =  embeddings / y_length
            // derivative mean = 1 / y_length
            let grads_broadcasted: ArrayView3<f32> = grads.broadcast((1, self.seq_len, self.output_embedding_len)).unwrap();
            let grads_broadcasted: Array3<f32> = &grads_broadcasted * (1.0 / self.seq_len as f32);
            grads = grads_broadcasted.into_dyn();
        }

        // backpropagate through summation along first axis
        let mut grads: ArrayD<f32> 
            = grads.broadcast(
                (self.n_attention_heads, self.seq_len, self.output_embedding_len)
                ).unwrap().into_dyn().into_owned();

        // final embeddings was created by summing each output embedding produced from each attention
        // head, derivative for dy/(dx,dy,dz) in y = x + w + z is 1, thus use the same gradients for
        // each head
        
        ////////////////////////////////////////////////
        // backpropagate through transform and hidden layers first
        grads = self.transform_layer_act.backward(grads);
        grads = self.transform_norm.backward(grads);
        grads = self.transform_layer.backward(grads);
        
        // final grads
        let mut final_grads: ArrayD<f32> = self.hidden_layer_act.backward(grads.clone());
        final_grads = self.hidden_norm.backward(final_grads);
        final_grads = self.hidden_layer.backward(final_grads);

        // sum along first axis as the inputs was original input embeddings (1, seq_len, in_embeddings)
        final_grads = final_grads.sum_axis(Axis(0)).insert_axis(Axis(0));

        // use weight gradients from the transform layer to backpropagate through
        // seq_output, seq_input, and key/query matrices branch
        grads = self.transform_layer.get_weight_grads();
        // remove axis previously needed for 
        grads = grads.remove_axis(Axis(1));
        //std::process::exit(1);

        // grads is (input_embedding, output_embedding)
        // backpropagate through seq_output activation and matrix
        grads = self.seq_output_layer_act.backward(grads);
        grads = self.seq_output_norm.backward(grads);
        grads = self.seq_output_layer.backward(grads);

        // batch transpose the grads
        grads = grads.permuted_axes(IxDyn(&[0, 2, 1]));

        // backpropagate through seq_input activation and matrix
        grads = self.seq_input_layer_act.backward(grads);
        //grads = self.seq_input_norm.backward(grads);
        grads = self.seq_input_layer.backward(grads);
        
        grads = self.attention_layer_act.backward(grads);
        //grads = self.attention_norm.backward(grads);
        // gradients for key matrix
        let mut key_grads: ArrayD<f32> = self.attention_layer.backward(grads);
        let mut query_grads: ArrayD<f32> = self.attention_layer.get_weight_grads();
        query_grads = query_grads.remove_axis(Axis(1)); // remove axis previously needed for broadcasting

        // batch transpose query gradients
        query_grads = query_grads.permuted_axes(IxDyn(&[0, 2, 1]));

        //println!("{:?}", key_grads);
        //println!("{:?}", query_grads);

        // backpropagate through key layer activation and weights
        //println!("{:?}", self.key_layer.weight_gradients);
        key_grads = self.key_layer_act.backward(key_grads);
        key_grads = self.key_norm.backward(key_grads);
        key_grads = self.key_layer.backward(key_grads);
        key_grads = key_grads.sum_axis(Axis(0)).insert_axis(Axis(0));
        //println!("{:?}", self.key_layer.weight_gradients);
        //println!("{:?}", key_grads);

        // backpropagate through query layer activation and weights
        query_grads = self.query_layer_act.backward(query_grads);
        query_grads = self.query_norm.backward(query_grads);
        query_grads = self.query_layer.backward(query_grads);
        query_grads = query_grads.sum_axis(Axis(0)).insert_axis(Axis(0));
        //println!("{:?}", query_grads);

        // sum both grads for final gradients to output from head
        // shape (seq_len, input_embeddings)
        final_grads += &(&key_grads + &query_grads);
        //println!("{:?}", final_grads);

        return final_grads;
    }

    pub fn update_params(&mut self)
    {
        self.key_layer.update_params();
        self.key_norm.update_params();
        self.key_layer_act.update_params();
        self.query_layer.update_params();
        self.query_norm.update_params();
        self.query_layer_act.update_params();
        self.attention_layer.update_params();
        //self.attention_norm.update_params();
        self.attention_layer_act.update_params();
        self.seq_input_layer.update_params();
        self.seq_input_norm.update_params();
        self.seq_input_layer_act.update_params();
        self.seq_output_layer.update_params();
        self.seq_output_norm.update_params();
        self.seq_output_layer_act.update_params();
        self.hidden_layer.update_params();
        self.hidden_norm.update_params();
        self.hidden_layer_act.update_params();
        self.transform_layer.update_params();
        self.transform_norm.update_params();
        self.transform_layer_act.update_params();
    }

    pub fn zero_grads(&mut self)
    {
        self.key_layer.zero_grads();
        self.key_norm.zero_grads();
        self.key_layer_act.zero_grads();
        self.query_layer.zero_grads();
        self.query_norm.zero_grads();
        self.query_layer_act.zero_grads();
        self.attention_layer.zero_grads();
        //self.attention_norm.zero_grads();
        self.attention_layer_act.zero_grads();
        self.seq_input_layer.zero_grads();
        self.seq_input_norm.zero_grads();
        self.seq_input_layer_act.zero_grads();
        self.seq_output_layer.zero_grads();
        self.seq_output_norm.zero_grads();
        self.seq_output_layer_act.zero_grads();
        self.hidden_layer.zero_grads();
        self.hidden_norm.zero_grads();
        self.hidden_layer_act.zero_grads();
        self.transform_layer.zero_grads();
        self.transform_norm.zero_grads();
        self.transform_layer_act.zero_grads();
    }

    pub fn details(&self)
    {
        /*
        println!("KEY_MATRICES:");
        for i in 0..self.n_attention_heads
        {
            self.attention_heads[i].key_layer.details();
        }
        println!("======================================");
        println!("QUERY_MATRICES:");
        for i in 0..self.n_attention_heads
        {
            self.attention_heads[i].query_layer.details();
        }
        println!("=====================================");
        */
    }

}

/*
// each attention head produces a different transformation matrix
// of size (input_embedding, output_embedding)
#[derive(Serialize, Deserialize)]
pub struct AttentionHead
{
    // key and matrix layers to produce the attention matrix
    key_layer: Dense, key_layer_act: Activation, key_norm: NormLayer,
    query_layer: Dense, query_layer_act: Activation, query_norm: NormLayer,
    attention_layer: Dense, attention_layer_act: Activation, attention_norm: NormLayer,

    key_mat: Array2<f32>, query_mat: Array2<f32>,

    // used to convert/process the (n, n) attention matrix into a (input, output) matrix
    seq_input_layer: Dense, seq_input_layer_act: Activation, seq_input_norm: NormLayer,
    seq_output_layer: Dense, seq_output_layer_act: Activation, seq_output_norm: NormLayer,
    transform_layer: Dense, transform_layer_act: Activation, transform_norm: NormLayer,
    hidden_layer: Dense, hidden_layer_act: Activation, hidden_norm: NormLayer,
    lr: f32, l2: f32
}
impl AttentionHead
{
    pub fn new(seq_len: usize, input_embedding: usize, output_embedding: usize, lr: f32, l2: f32) -> Self
    {
        let key_layer: Dense = Dense::new(output_embedding, l2, lr);
        let key_norm: NormLayer = NormLayer::new(lr, 1);
        let key_layer_act: Activation = Activation::new("silu", lr);

        let query_layer: Dense = Dense::new(output_embedding, l2, lr);
        let query_norm: NormLayer = NormLayer::new(lr, 1);
        let query_layer_act: Activation = Activation::new("silu", lr);

        let attention_layer: Dense = Dense::new(seq_len, l2, lr);
        let attention_norm: NormLayer = NormLayer::new(lr, 1);
        let attention_layer_act: Activation = Activation::new("silu", lr);

        let seq_input_layer: Dense = Dense::new(input_embedding, l2, lr);
        let seq_input_norm: NormLayer = NormLayer::new(lr, 1);
        let seq_input_layer_act: Activation = Activation::new("silu", lr);
        //let mut seq_input_residual: Dense = Dense::new(input_embedding, l2, lr);
        //seq_input_residual.set_weights(Array2::ones((seq_len, input_embedding)));

        let seq_output_layer: Dense = Dense::new(output_embedding, l2, lr);
        let seq_output_norm: NormLayer = NormLayer::new(lr, 0);
        let seq_output_layer_act: Activation = Activation::new("silu", lr);
        //let mut seq_output_residual: Dense = Dense::new(output_embedding, l2, lr);
        //seq_output_residual.set_weights(Array2::ones((seq_len, output_embedding)));

        // initialise transform layers
        let transform_layer: Dense = Dense::new(output_embedding, l2, lr);
        let transform_norm: NormLayer = NormLayer::new(lr, 1);
        let transform_layer_act: Activation = Activation::new("silu", lr);
        //let mut transform_residual: Dense = Dense::new(output_embedding, l2, lr);
        //transform_residual.set_weights(Array2::ones((input_embedding, output_embedding)));

        // initialise hidden layers
        let hidden_layer: Dense = Dense::new(input_embedding, l2, lr);
        let hidden_norm: NormLayer = NormLayer::new(lr, 1);
        let hidden_layer_act: Activation = Activation::new("silu", lr);
        //let mut hidden_residual: Dense = Dense::new(input_embedding, l2, lr);
        //hidden_residual.set_weights(Array2::ones((input_embedding, input_embedding)));

        return Self
        {
            key_layer, key_layer_act, key_norm,
            query_layer, query_layer_act, query_norm,
            attention_layer, attention_layer_act, attention_norm,
            key_mat: Array2::zeros((0, 0)),
            query_mat: Array2::zeros((0, 0)),
            seq_input_layer, seq_input_layer_act, seq_input_norm,
            seq_output_layer, seq_output_layer_act, seq_output_norm,
            hidden_layer, hidden_layer_act, hidden_norm,
            transform_layer, transform_layer_act, transform_norm,
            lr, l2
        }
    }

    pub fn forward(&mut self, input_embedding: ArrayD<f32>) -> ArrayD<f32>
    {
        let mut key_mat: ArrayD<f32> = self.key_layer.forward(input_embedding.clone());
        key_mat = self.key_norm.forward(key_mat);
        key_mat = self.key_layer_act.forward(key_mat);

        let mut query_mat: ArrayD<f32> = self.query_layer.forward(input_embedding.clone());
        query_mat = self.query_norm.forward(query_mat);
        query_mat = self.query_layer_act.forward(query_mat);

        self.attention_layer.set_weights(
            query_mat.t().into_owned().into_dimensionality::<Ix2>().unwrap()
        );
        // calculate attention scores and record key/query matrices
        let mut attention_vals: ArrayD<f32> = self.attention_layer.forward(key_mat.clone());
        attention_vals = self.attention_norm.forward(attention_vals);
        attention_vals = self.attention_layer_act.forward(attention_vals);
        // apply matrix multiplications to convert the (seq, seq) attention values
        // to a (input, output) transformation matrix
        //println!("{:?}", attention_vals);

        // transform (seq, seq) to (seq, input)
        //let attention_vals_prev: ArrayD<f32> = attention_vals.clone();
        attention_vals = self.seq_input_layer.forward(attention_vals);
        attention_vals = self.seq_input_norm.forward(attention_vals);
        attention_vals = self.seq_input_layer_act.forward(attention_vals);
        //let attention_vals_residual: ArrayD<f32> = self.seq_input_residual.forward(attention_vals_prev);
        //attention_vals += &attention_vals_residual;
        //println!("{:?}", attention_vals);
        
        // transpose to (input, seq) so it compatible with (seq, output)
        attention_vals = attention_vals.t().into_owned();
        
        // final transformation matrix is (input, output)
        //let attention_vals_prev: ArrayD<f32> = attention_vals.clone();
        attention_vals = self.seq_output_layer.forward(attention_vals);
        attention_vals = self.seq_output_norm.forward(attention_vals);
        attention_vals = self.seq_output_layer_act.forward(attention_vals);
        //let attention_vals_residual: ArrayD<f32> = self.seq_output_residual.forward(attention_vals_prev);
        //attention_vals += &attention_vals_residual;
        //println!("{:?}", attention_vals);
        // set transform layer weights
        self.transform_layer.set_weights(
            attention_vals.into_dimensionality::<Ix2>().unwrap()
        );

        // pass through hidden layer first
        let mut hidden: ArrayD<f32> = self.hidden_layer.forward(input_embedding.clone());
        hidden = self.hidden_norm.forward(hidden);
        hidden = self.hidden_layer_act.forward(hidden);
        //let mut residual: ArrayD<f32> = self.hidden_residual.forward(input_embedding);
        //hidden += &residual;

        // apply the transformation matrix
        //let hidden_prev: ArrayD<f32> = hidden.clone();
        let mut output_embeddings: ArrayD<f32> = self.transform_layer.forward(hidden);
        output_embeddings = self.transform_norm.forward(output_embeddings);
        output_embeddings = self.transform_layer_act.forward(output_embeddings);
        //residual = self.transform_residual.forward(hidden_prev);
        //output_embeddings += &residual;
        return output_embeddings;
    }

    pub fn backward(&mut self, mut grads: ArrayD<f32>) -> ArrayD<f32>
    {
        //println!("{:?}", grads);
        // backpropagate through transform and hidden layers first
        grads = self.transform_layer_act.backward(grads);
        grads = self.transform_norm.backward(grads);
        grads = self.transform_layer.backward(grads);
        //println!("{:?}", self.transform_layer.weights);
        //println!("{:?}", grads);
        // final grads
        let mut final_grads: ArrayD<f32> = self.hidden_layer_act.backward(grads.clone());
        final_grads = self.hidden_norm.backward(final_grads);
        final_grads = self.hidden_layer.backward(final_grads);

        // use weight gradients from the transform layer to backpropagate through
        // seq_output, seq_input, and key/query matrices
        grads = self.transform_layer.get_weight_grads();
        //println!("{:?}", grads);
        //std::process::exit(1);

        // grads is (input_embedding, output_embedding)
        // backpropagate through seq_output activation and matrix
        grads = self.seq_output_layer_act.backward(grads);
        grads = self.seq_output_norm.backward(grads);
        grads = self.seq_output_layer.backward(grads);

        // transpose the grads
        grads = grads.t().into_owned();

        // backpropagate through seq_input activation and matrix
        grads = self.seq_input_layer_act.backward(grads);
        //grads = self.seq_input_norm.backward(grads);
        grads = self.seq_input_layer.backward(grads);

        grads = self.attention_layer_act.backward(grads);
        //grads = self.attention_norm.backward(grads);
        
        // gradients for key matrix
        let mut key_grads: ArrayD<f32> = self.attention_layer.backward(grads);
        let mut query_grads: ArrayD<f32> = self.attention_layer.get_weight_grads();
        // transpose query gradients
        query_grads = query_grads.t().into_owned();

        //println!("{:?}", key_grads);
        //println!("{:?}", query_grads);

        // backpropagate through key layer activation and weights
        //println!("{:?}", self.key_layer.weight_gradients);
        key_grads = self.key_layer_act.backward(key_grads);
        key_grads = self.key_norm.backward(key_grads);
        key_grads = self.key_layer.backward(key_grads);
        //println!("{:?}", self.key_layer.weight_gradients);
        //println!("{:?}", key_grads);

        // backpropagate through query layer activation and weights
        query_grads = self.query_layer_act.backward(query_grads);
        query_grads = self.query_norm.backward(query_grads);
        query_grads = self.query_layer.backward(query_grads);
        //println!("{:?}", query_grads);

        // sum both grads for final gradients to output from head
        // shape (seq_len, input_embeddings)
        final_grads += &(&key_grads + &query_grads);
        //println!("{:?}", final_grads);
        
        //std::process::exit(1);
        return final_grads;
    }

    pub fn update_params(&mut self)
    {
        self.key_layer.update_params();
        self.key_norm.update_params();
        self.key_layer_act.update_params();
        self.query_layer.update_params();
        self.query_norm.update_params();
        self.query_layer_act.update_params();
        self.attention_layer.update_params();
        self.attention_norm.update_params();
        self.attention_layer_act.update_params();
        self.seq_input_layer.update_params();
        self.seq_input_norm.update_params();
        self.seq_input_layer_act.update_params();
        self.seq_output_layer.update_params();
        self.seq_output_norm.update_params();
        self.seq_output_layer_act.update_params();
        self.hidden_layer.update_params();
        self.hidden_norm.update_params();
        self.hidden_layer_act.update_params();
        self.transform_layer.update_params();
        self.transform_norm.update_params();
        self.transform_layer_act.update_params();
    }

    pub fn zero_grads(&mut self)
    {
        self.key_layer.zero_grads();
        self.key_norm.zero_grads();
        self.key_layer_act.zero_grads();
        self.query_layer.zero_grads();
        self.query_norm.zero_grads();
        self.query_layer_act.zero_grads();
        self.attention_layer.zero_grads();
        self.attention_norm.zero_grads();
        self.attention_layer_act.zero_grads();
        self.seq_input_layer.zero_grads();
        self.seq_input_norm.zero_grads();
        self.seq_input_layer_act.zero_grads();
        self.seq_output_layer.zero_grads();
        self.seq_output_norm.zero_grads();
        self.seq_output_layer_act.zero_grads();
        self.hidden_layer.zero_grads();
        self.hidden_norm.zero_grads();
        self.hidden_layer_act.zero_grads();
        self.transform_layer.zero_grads();
        self.transform_norm.zero_grads();
        self.transform_layer_act.zero_grads();
    }
}
*/