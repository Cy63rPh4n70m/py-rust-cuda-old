use ndarray::{Array2, ArrayD, ArrayView2, Ix2};
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Dense
{
    pub in_shape: (usize, usize),
    pub out_shape: (usize, usize),
    pub n_out: usize,

    pub weights: Array2<f32>,
    pub weight_gradients: Array2<f32>,

    pub biases: Array2<f32>,
    pub bias_gradients: Array2<f32>,
    
    pub input: Array2<f32>,
    pub l2: f32,
    pub lr: f32,
}
impl Dense
{
    // weight matrix initialize during first ever run
    pub fn new(
        n_out: usize,
        l2: f32, lr: f32
    ) -> Self
    {
        /*
        unsafe 
        {
        let mut vector: Array3<f32> = Array3::from_shape_fn(
            (1, 5, 2000), |(i, j, k)| (i * 5 * 2000 + j * 2000 + k) as f32 / 10000.0);
        let pointer: *mut f32 = vector.as_mut_ptr();
        let mut vector1: Array3<f32> = Array3::from_shape_fn(
            (1, 2000, 5), |(i, j, k)| (i * 2000 * 5 + j * 5 + k) as f32 / 10000.0);
        let pointer1: *mut f32 = vector1.as_mut_ptr();
        let ptr_cuda: *mut f32 = to_cuda(pointer, 1 * 5 * 2000);
        let ptr1_cuda: *mut f32 = to_cuda(pointer1, 1 * 2000 * 5);
        let result: *mut f32 = new_cuda_array( 1 * 5 * 5);
        let broadcast_buf: *mut f32 = new_cuda_array(1 * 5 * 2000 * 5);

        println!("{:?}", vector);
        println!("{:?}", vector1);

        mat_mul(
            ptr_cuda, 1, 5, 2000, 
            ptr1_cuda, 1, 2000, 5, 
            result, broadcast_buf
        );
        
        let result1: *mut f32 = new_cuda_array(1 * 5 * 5);

        activation_fn(result1, result, 1, 5, 5, "sigmoid");

        let result: *mut f32 = to_cpu(result, 1 * 5 * 5);
        let result_slice: &mut [f32] = std::slice::from_raw_parts_mut(result, 25);
        let result_array: Array3<f32> = Array3::from_shape_vec((1, 5, 5), result_slice.to_vec()).unwrap();
        println!("{:?}", result_array);

        let result1: *mut f32 = to_cpu(result1, 1 * 5 * 5);
        let result_slice: &mut [f32] = std::slice::from_raw_parts_mut(result1, 25);
        println!("{:?}", result_slice);
        exit(1);

        }
        */

        let weights: Array2<f32> = Array2::zeros((0, 0));
        let weight_gradients: Array2<f32> = Array2::zeros((0, 0));

        let biases: Array2<f32> = Array2::zeros((0, 0));
        let bias_gradients: Array2<f32> = Array2::zeros((0, 0));

        //biases1d.par_map_inplace(|val: &mut f32| *val = rand::thread_rng().gen_range(-bias_range..bias_range));

        return Self
        {
            input: Array2::zeros((0, 0)),
            weights,
            weight_gradients,
            biases,
            bias_gradients,
            in_shape: (0, 0),
            out_shape: (0, 0),
            n_out,
            l2,
            lr
        }
    }

    pub fn set_weights(&mut self, weights2d: Array2<f32>)
    {
        let rows: usize = weights2d.shape()[0];
        let cols: usize = weights2d.shape()[1];
        self.weights = weights2d;
        self.weight_gradients = Array2::zeros((0, 0));
        self.in_shape = (rows, cols);
        self.out_shape = (rows, self.n_out);
    }

    pub fn forward(&mut self, input_vec: ArrayD<f32>) -> ArrayD<f32>
    {
        if self.weights.shape() == &[0, 0]
        {
            let input_shape: &[usize] = input_vec.shape();

            let rows: usize = input_shape[0];
            let cols: usize = input_shape[1];

            let range: f32 = (6.0 / (cols + self.n_out) as f32).sqrt();
            self.weights = Array2::from_shape_fn((cols, self.n_out), |_| rand::thread_rng().gen_range(-range..range));
            self.weight_gradients = Array2::zeros((cols, self.n_out));
            self.biases = Array2::zeros((rows, self.n_out));
            self.bias_gradients = Array2::zeros((rows, self.n_out));

            self.in_shape = (rows, cols);
            self.out_shape = (rows, self.n_out);
        }

        if self.biases.shape() == &[0, 0]
        {
            let input_shape: &[usize] = input_vec.shape();

            // needed to get the sequence length
            let rows: usize = input_shape[0];
            let cols: usize = input_shape[1];

            let range: f32 = (6.0 / (cols + self.n_out) as f32).sqrt();
            
            self.biases = Array2::zeros((rows, self.n_out));
            self.bias_gradients = Array2::zeros((rows, self.n_out));
        }

        //let input_reshaped: ArrayD<f32> = input_vec.insert_axis(Axis(2));
        //self.summed
        //    = (&input_reshaped * &self.weights).sum_axis(Axis(1))
        //    + &self.biases;
        
        //self.input = input_reshaped;
        let input: Array2<f32> = input_vec.into_dimensionality().unwrap();
        let result: Array2<f32> = input.dot(&self.weights) + &self.biases;
        self.input = input;
        
        return result.into_dyn();
    }

    pub fn backward(&mut self, loss_r_summed: ArrayD<f32>) -> ArrayD<f32>
    {

        let grads2d: Array2<f32> = loss_r_summed.into_dimensionality::<Ix2>().unwrap();
        //// backpropagating for weights
        // transpose input vector
        let transposed_input: ArrayView2<f32> = self.input.t();
        // derivative of y = w * inputs respect to w is inputs 
        self.weight_gradients = transposed_input.dot(&grads2d);

        // from each output calculate summed respect to weights and
        // summed respect to inputs (activated from next layer in reverse)

        //// backpropagating for inputs
        // derivative of y = w * inputs respect to inputs is w
        let loss_r_next_layer: Array2<f32> = grads2d.dot(&self.weights.t());

        // calculate summed respect to bias
        // calculate bias gradients
        self.bias_gradients = 1.0 * grads2d; // bias derivative is 1.0
        //println!("{:?}", loss_r_next_layer);
        
        //println!("{:?}", loss_r_summed);
        //println!("{:?}", self.weights2d);
        //println!("{:?}", loss_r_next_layer);
        //std::process::exit(1);

        /*
        // calculate summed respect to bias
        // calculate bias gradients
        self.bias_gradients += &(1.0 * &loss_r_summed); // bias derivative is 1.0

        // reshape
        let mut shape: Vec<usize> = loss_r_summed.shape().to_vec();
        shape.insert(shape.len() - 1, self.in_shape.1);
        let grads: ArrayViewD<f32> = loss_r_summed.broadcast(shape).unwrap();
        
        // calculate update gradients for weights (multiply with the reshaped inputs)

        let weight_grads: ArrayD<f32> = (&grads * &self.input).sum_axis(Axis(0));
        self.weight_gradients += &weight_grads;

        // calculate update gradients for input (multiply with weights)
        let return_grads: ArrayD<f32> = (&grads * &self.weights).sum_axis(Axis(2));
        //println!("{:?}", return_grads);
        //println!("{:?}", weight_grads);
        //exit(1);

        return return_grads;
        */
        return loss_r_next_layer.into_dyn();
        
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
        self.weight_gradients *= 0.0;
        self.bias_gradients *= 0.0;
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