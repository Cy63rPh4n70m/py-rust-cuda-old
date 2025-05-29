use ndarray::{Array3, Array4, ArrayD, ArrayViewD, Axis};
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct DenseForAttention
{
    pub in_shape: Vec<usize>,
    pub out_shape: Vec<usize>,
    pub n_heads: usize, 
    pub n_out: usize,

    pub input: Array4<f32>,

    pub weights: Array4<f32>,
    pub weight_gradients: Array4<f32>,

    pub biases: Array3<f32>,
    pub bias_gradients: Array3<f32>,
    
    pub summed: Array3<f32>,
    pub l2: f32,
    pub lr: f32,

    pub backward_passes_count: u128,
    pub count: u128
}
impl DenseForAttention
{
    // weight matrix initialize during first ever run
    pub fn new(
        n_heads: usize, n_out: usize,
        l2: f32, lr: f32
    ) -> Self
    {
        let weights: Array4<f32> = Array4::zeros((0, 0, 0, 0));
        let weight_gradients: Array4<f32> = Array4::zeros((0, 0, 0, 0));

        let biases: Array3<f32> = Array3::zeros((0, 0, 0));
        let bias_gradients: Array3<f32> = Array3::zeros((0, 0, 0));

        //biases1d.par_map_inplace(|val: &mut f32| *val = rand::thread_rng().gen_range(-bias_range..bias_range));

        return Self
        {
            input: Array4::zeros((0, 0, 0, 0)),
            weights,
            weight_gradients,
            biases,
            bias_gradients,
            summed: Array3::zeros((0, 0, 0)),
            in_shape: Vec::new(),
            out_shape: Vec::new(),
            n_heads,
            n_out,
            l2,
            backward_passes_count: 0,
            count: 0,
            lr
        }
    }

    pub fn set_weights(&mut self, weights: Array4<f32>)
    {
        self.weights = weights;
        let shape: &[usize] = self.weights.shape();
        self.weight_gradients = Array4::zeros((shape[0], shape[1], shape[2], shape[3]));
    }

    // accepts a 3D input and outputs a 3D tensor
    pub fn forward(&mut self, input: ArrayD<f32>) -> ArrayD<f32>
    {
        if self.biases.shape() == &[0, 0, 0]
        {
            let input_shape: &[usize] = input.shape();
            self.biases = Array3::zeros((self.n_heads, input_shape[1], self.n_out));
            self.bias_gradients = self.biases.clone();

            if self.weights.shape() == &[0, 0, 0, 0]
            {
                let range: f32 
                    = (6.0 / (input_shape[1] * input_shape[2] + input_shape[1] * self.n_out) as f32).sqrt();
                self.weights 
                    = Array4::from_shape_fn((self.n_heads, 1, input_shape[2], self.n_out), 
                    |_| rand::thread_rng().gen_range(-range..range));
                self.weight_gradients = self.weights.clone();
            }

            self.in_shape = input_shape.to_vec();
            self.out_shape = self.biases.shape().to_vec();
        }

        // assumes input is always 3D
        let input_reshaped: Array4<f32> = input.insert_axis(Axis(3)).into_dimensionality().unwrap();
        //println!("{:?}", input_reshaped);
        //println!("{:?}", self.weights);

        // equivalent to batch matrix multiplication
        let result: Array3<f32>
            = (&input_reshaped * &self.weights).sum_axis(Axis(2))
            + &self.biases;
        //println!("{:?}", self.biases);
        //exit(1);
        
        self.input = input_reshaped;

        return result.into_dyn();
    }

    pub fn backward(&mut self, mut loss_r_summed: ArrayD<f32>) -> ArrayD<f32>
    {
        // calculate summed respect to bias
        // calculate bias gradients
        self.bias_gradients += &(1.0 * &loss_r_summed); // bias derivative is 1.0

        // loss_r_summed is 3D, (n_heads, input_rows, output_cols)
        // broadcast to (n_heads, input_rows, input_cols, output_cols)
        let mut shape: Vec<usize> = loss_r_summed.shape().to_vec();
        shape.insert(shape.len() - 1, self.in_shape[2]);
        loss_r_summed = loss_r_summed.insert_axis(Axis(2));
        let grads: ArrayViewD<f32> = loss_r_summed.broadcast(shape).unwrap();
        
        // calculate update gradients for weights (multiply with the reshaped inputs)
        let weight_grads: ArrayD<f32> = (&grads * &self.input).sum_axis(Axis(1)).insert_axis(Axis(1));
        self.weight_gradients += &weight_grads;

        // calculate gradients to return (input shape) (multiply with weights)
        let return_grads: ArrayD<f32> = (&grads * &self.weights).sum_axis(Axis(3));
        return return_grads;
    }

    pub fn get_weight_grads(&mut self) -> ArrayD<f32>
    {
        return self.weight_gradients.clone().into_dyn();
    }

    pub fn update_params(&mut self)
    {   
        self.weights -= &(self.lr * (&self.weight_gradients + self.l2 * &self.weights));
        self.biases -= &(self.lr * &self.bias_gradients);
    }

    pub fn zero_grads(&mut self)
    {
        self.weight_gradients.fill(0.0);
        self.bias_gradients.fill(0.0);
    }

    pub fn details(&self)
    {
        println!("Layer type: DENSE");
        println!("Input shape: {:?}", self.in_shape);
        println!("Output shape: {:?}", self.out_shape);
        println!("L2: {}", self.l2);
        println!("Weights: \n{:?}", self.weights);
        println!("--------------------------------------");
        println!("Biases: \n{:?}", self.biases);
    }

}

 /*   
        self.weight_gradients.map_mut(
            |v: &mut f32| 
            if *v > 0.0
            {
                *v = (*v).min(1.0);
            }
            else if *v < 0.0
            {
                *v = (*v).max(-1.0);
            }
            );
        */

        /*
        // add squared weight_grads
        let weights_powered: Array2<f32> = self.weight_gradients.mapv(|v: f32| v.powf(2.0));
        let bias_powered: Array1<f32> = self.bias_gradients.mapv(|v: f32| v.powf(2.0));
        //self.weight_grads_mov_ave += &weights_powered;
        //self.bias_grads_mov_ave += &bias_powered;

        //self.weight_grads_sqr_sum.push_back(weights_powered);
        //self.bias_grads_sqr_sum.push_back(bias_powered);

        //self.count += 1;

        //if self.weight_grads_sqr_sum.len() > 3000
        //{
        //    let oldest_weights: Array2<f32> = self.weight_grads_sqr_sum.pop_front().unwrap();
        //    let oldest_biases: Array1<f32> = self.bias_grads_sqr_sum.pop_front().unwrap();
        //    self.weight_grads_mov_ave -= &oldest_weights;
        //    self.bias_grads_mov_ave -= &oldest_biases;
        //}
        if self.backward_passes_count == 0
        {
            self.weight_grads_mov_ave += &weights_powered;
            self.bias_grads_mov_ave += &bias_powered;
            self.backward_passes_count += 1;
        }
        else
        {
            //self.weight_m = self.b1 * &self.weight_m + (1.0 - self.b1) * &self.weight_gradients;
            self.weight_grads_mov_ave = self.b * &self.weight_grads_mov_ave + (1.0 - self.b) * &weights_powered;
            //self.bias_m = self.b1 * &self.bias_m + (1.0 - self.b1) * &self.bias_gradients;
            self.bias_grads_mov_ave = self.b * &self.bias_grads_mov_ave + (1.0 - self.b) * &bias_powered;
        }
        
        //let weight_m1: Array2<f32> = &self.weight_m / (1.0 - self.b1.powf(self.backward_passes_count));
        //let weight_v1: Array2<f32> = &self.weight_v / (1.0 - self.b2.powf(self.backward_passes_count));
        //let bias_m1: Array1<f32> = &self.bias_m / (1.0 - self.b1.powf(self.backward_passes_count));
        //let bias_v1: Array1<f32> = &self.bias_v / (1.0 - self.b2.powf(self.backward_passes_count));
        
        //let denominator: Array2<f32> = self.weight_grads_mov_ave.mapv(|v: f32| (v + 1e-8).sqrt());
        let weight_adaptive_lrs: Array2<f32> = self.weight_grads_mov_ave.mapv(
            |v: f32| 
            if lr / (v + 1e-8).sqrt() > max_lr { max_lr } 
            else if lr / (v + 1e-8).sqrt() < min_lr { min_lr } 
            else { lr / (v + 1e-8).sqrt() }
        );

        //let denominator: Array1<f32> = self.bias_grads_mov_ave.mapv(|v: f32| (v + 1e-8).sqrt());
        let bias_adaptive_lrs: Array1<f32> = self.bias_grads_mov_ave.mapv(
            |v: f32| 
            if lr / (v + 1e-8).sqrt() > max_lr { max_lr } 
            else if lr / (v + 1e-8).sqrt() < min_lr { min_lr } 
            else { lr / (v + 1e-8).sqrt() }
        );
        */

        //self.weights2d -= &(lr * &self.weight_gradients);
        //self.biases1d -= &(lr * &self.bias_gradients);

        /**/
        /*
        // calculate summed respect to bias
        // calculate bias gradients
        self.bias_gradients += &(1.0 * &loss_r_summed); // bias derivative is 1.0

        // from each output calculate summed respect to weights and
        // summed respect to inputs (activated from next layer in reverse)
        
        //// backpropagating for weights
        // transpose input vector

        let input2d = self.input.clone().into_dimensionality::<Ix2>().unwrap();
        let grads2d = loss_r_summed.into_dimensionality::<Ix2>().unwrap();
        let weights2d = self.weights.clone().into_dimensionality::<Ix2>().unwrap();
        let transposed_input: ArrayView2<f32> = input2d.t();

        // derivative of y = w * inputs respect to w is inputs 
        self.weight_gradients += &transposed_input.dot(&grads2d);

        //// backpropagating for inputs
        // derivative of y = w * inputs respect to inputs is w
        let loss_r_next_layer: Array2<f32> = grads2d.dot(&weights2d.t());
        
        //println!("{:?}", loss_r_summed);
        //println!("{:?}", self.weights2d);
        //println!("{:?}", loss_r_next_layer);
        //std::process::exit(1);
        */