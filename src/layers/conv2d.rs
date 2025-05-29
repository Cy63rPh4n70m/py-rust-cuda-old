use std::ops::Range;

use ndarray::{s, Array2, Array3, ArrayD, ArrayView2, ArrayViewMut2, ArrayViewMut3};
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Conv2d
{
    pub strides_len: (usize, usize), // stride length/interval
    pub strides_count: (usize, usize), // actual number of times to move by stride

    pub filter_sets: Array2<f32>, // each filter is 1d
    pub filter_set_grads: Array2<f32>,
    pub biases: Array3<f32>, // one bias per filter set/output channel
    pub bias_grads: Array3<f32>,

    pub filter_dim: usize,

    pub input_tensors: Array3<f32>, // spatial representation of image
    pub strided_input_2d: Array2<f32>,
    pub summed_tensors: Array3<f32>, // reshaped output

    pub in_channels: usize,
    pub out_channels: usize,
    pub l2: f32,
    //pub weight_range: f32,
    //pub bias_range: f32
}
impl Conv2d
{
    pub fn new(
        in_channels: usize,
        out_channels: usize, 
        filter_dim: usize,
        stride_len: usize, 
        l2: f32,
    ) -> Self
    {
        //let a: Array3<f32> = Array3::ones((3, 3, 3));
        //let b: Array4<f32> = Array4::ones((10, 3, 3, 3));
        //println!("{:?}", b.sum_axis(Axis(0)));
        //std::process::exit(1);
        //let start_y: usize = 0;
        //let end_y: usize = 1;
        //let start_x: usize = 1;
        //let end_x: usize = 2;
        //a.slice_mut(s![start_y..end_y + 1, start_x..end_x + 1]).zip_mut_with(&c, |av, cv| *av -= cv);

        // number of filter sets -> n output channels
        // (out_channels, flattened_filter)
        let mut filter_sets: Array2<f32> = 
            Array2::zeros((out_channels, in_channels * filter_dim * filter_dim));
        let filter_set_grads: Array2<f32> = 
            Array2::zeros((out_channels, in_channels * filter_dim * filter_dim));

        // initialize filters with random values
        let range: f32 = (6.0 / (in_channels * filter_dim * filter_dim + out_channels) as f32).sqrt();
        filter_sets.par_map_inplace(
            |val: &mut f32| 
            *val = rand::thread_rng().gen_range(-range..range)
        );

        // stores the sections of the input tensor as values that are
        // elementwise multiplied with the filters
        // used to speed up backpropagation

        return Self
        {
            filter_sets,
            filter_set_grads,
            strides_len: (stride_len, stride_len),
            filter_dim,

            // proper shapes initialized during the first ever forward pass called
            biases: Array3::zeros((0, 0, 0)),
            bias_grads: Array3::zeros((0, 0, 0)),
            input_tensors: Array3::zeros((0, 0, 0)),
            strided_input_2d: Array2::zeros((0, 0)),
            summed_tensors: Array3::zeros((0, 0, 0)),
            in_channels,
            out_channels,
            l2,
            //weight_range,
            //bias_range,
            strides_count: (0, 0)
        };

    }

    pub fn forward(&mut self, input_tensors: ArrayD<f32>) -> ArrayD<f32>
    {
        // if first time ever calling the forward method
        if self.input_tensors.shape() == &[0, 0, 0]
        {
            let input_shape: &[usize] = input_tensors.shape();
            self.input_tensors = Array3::zeros(
                (
                    input_shape[0],
                    input_shape[1],
                    input_shape[2]
                )
            );

            // calculate output shape
            let strides_count: (usize, usize) = 
                calculate_stride_counts(
                    (input_shape[1], input_shape[2]), 
                    self.filter_dim, self.strides_len
                );
            
            // initialize the tensors with the output shape
            self.summed_tensors = Array3::zeros(
                (self.out_channels, strides_count.0, strides_count.1)
                );

            self.biases = self.summed_tensors.clone();
            //self.biases.par_map_inplace(
            //    |val: &mut f32| 
            //    *val = rand::thread_rng().gen_range(-self.bias_range..self.bias_range)
            //);

            self.bias_grads = self.summed_tensors.clone();

            self.strides_count = strides_count;

            // obtain the strides into a 2d matrix
            // matrix multiplied with filter matrix
            // (stride_flattened_len, total_strides_across_whole_img)
            self.strided_input_2d = 
                Array2::zeros(
                    (self.in_channels * self.filter_dim * self.filter_dim, 
                        self.strides_count.0 * self.strides_count.1)
                );
            
            println!("Input shape: {:?}", input_shape);
            println!("Output shape: {:?}", (self.out_channels, strides_count.0, strides_count.1));
        }
        
        // values in summed and activated are initialized to zero during
        // backpropagation when they are no longer needed
        self.input_tensors.fill(0.0);
        self.summed_tensors.fill(0.0);
        self.strided_input_2d.fill(0.0);
        
        self.input_tensors += &input_tensors;

        // used for selection of sub 3D tensor ranges
        let in_chan_range: Range<usize> = 0..self.in_channels;
        let mut start_y: usize = 0;
        let mut end_y: usize = self.filter_dim;
        let mut start_x: usize = 0;
        let mut end_x: usize = self.filter_dim;
        // calculate n times to move left and right depending
        // on filter and stride length

        // add weight sums
        //*weight_sum = self.filter_sets.map(|val| val.abs()).sum();
        //*squared_weight_sum = self.filter_sets.map(|val| val * val).sum();
        
        //println!("{:?}", self.summed_tensors.shape());
        //println!("{:?}", self.filter_sets.shape());

        let mut n_strides_passed: usize = 0;

        for _stride_y in 0..self.strides_count.0
        {
            // will not require second loop for activated_vector
            for _stride_x in 0..self.strides_count.1
            {
                // get 3d slice
                let windowed_input: Array3<f32> = 
                    self.input_tensors.slice(
                        s![
                            in_chan_range.clone(), 
                            start_y..end_y,
                            start_x..end_x
                        ]
                    ).into_owned();
                //println!("{:?}", windowed_input);

                
                // vertical flattened
                //println!("{:?}, {:?}", self.in_channels, self.filter_dim);
                let windowed_input_flattened: Array2<f32> = 
                    windowed_input.into_shape((self.in_channels * self.filter_dim * self.filter_dim, 1))
                    .unwrap();

                //println!("{:?}", windowed_input_flattened);
                
                // get vertical 1d slice of input stride matrix
                let mut stride_matrix_slice: ArrayViewMut2<f32> = 
                    self.strided_input_2d.slice_mut(
                        s![
                        0..self.in_channels * self.filter_dim * self.filter_dim,
                        n_strides_passed..n_strides_passed + 1
                        ]
                    );
                
                // recording flattened stride
                stride_matrix_slice += &windowed_input_flattened;
                n_strides_passed += 1;
                //println!("{:?}", self.strided_input_2d);
                
                // move window right
                start_x += self.strides_len.1;
                end_x += self.strides_len.1;
            }
            
            // move window down
            start_y += self.strides_len.0;
            end_y += self.strides_len.0;

            // move window back to left
            start_x = 0;
            end_x = self.filter_dim;
        }

        // perform matrix multiplication
        // between stride_matrix and filter_matrix
        // equivalent to performing strided elementwise dot prods faster

        // (out_channels, flattened_img_dim)
        //let mut output_2d: Array2<f32> = Array2::zeros((self.out_channels, self.strides_count.0 * self.strides_count.1));
        //general_mat_mul(1.0, &self.filter_sets, &self.strided_input_2d, 1.0, &mut output_2d);
        let output_2d: Array2<f32> = self.filter_sets.dot(&self.strided_input_2d);
        //println!("{:?}", output_2d);

        // convert to (out_channels, new_height, new_width)
        let mut output_3d: Array3<f32> = 
            output_2d.into_shape((self.out_channels, self.strides_count.0, self.strides_count.1))
            .unwrap();

        //println!("{:?}", output_3d);

        output_3d += &self.biases;
        self.summed_tensors += &output_3d;

        //println!("{:?}", self.summed_tensors);
        return output_3d.into_dyn();

    }

    pub fn backward(&mut self, loss_r_summed: ArrayD<f32>) -> ArrayD<f32>
    {
        // num output gradients matrices should equal number of filters/filter sets
        //println!("{:?}", self.input_shape);
        //println!("{:?}", self.output_shape);
        //println!("{:?}", obtain_shape_of_3d_mat(&gradients_3d));
        //println!("{:?}", obtain_shape_of_3d_mat(&self.filter_sets));
        //println!("{:?}", obtain_shape_of_3d_mat(&self.activated_tensors));
        //std::process::exit(1);
        //println!("{:?}", loss_r_summed);

        // bias gradient is constant 1.0
        self.bias_grads += &(1.0 * &loss_r_summed);

        // calculate gradients for filters
        let loss_r_summed_2d: Array2<f32> = 
            loss_r_summed.into_shape(
                (self.out_channels, self.strides_count.0 * self.strides_count.1)).unwrap();
                
        self.filter_set_grads += &loss_r_summed_2d.dot(&self.strided_input_2d.t());

        // create gradient storage for input strides (currently in 2D)
        let mut loss_r_input_strides: Array2<f32> = Array2::zeros(
            (self.in_channels * self.filter_dim * self.filter_dim,
            self.strides_count.0 * self.strides_count.1)
        );

        //println!("{:?}", loss_r_summed_2d);
        //println!("{:?}", loss_r_input_strides);
        //println!("{:?}", transposed_filters);

        // calculate gradients for input strides 
        // (requires broadcasting on each filter iteratively)
        let transposed_filters: ArrayView2<f32> = self.filter_sets.t();

        for i in 0..self.out_channels
        {
            let vertical_filter: ArrayView2<f32> = transposed_filters.slice(
                s![0..self.in_channels * self.filter_dim * self.filter_dim, 
                i..i + 1]
            );

            //println!("{:?}", vertical_filter);

            let vertical_filter_broadcasted: ArrayView2<f32> = 
                vertical_filter.broadcast(
                    (self.in_channels * self.filter_dim * self.filter_dim, 
                    self.strides_count.0 * self.strides_count.1)
                ).unwrap();
            
            //println!("{:?}", vertical_filter_broadcasted);
            
            let loss_r_sum_slice: ArrayView2<f32> = 
                loss_r_summed_2d.slice(
                    s![i..i + 1, 0..self.strides_count.0 * self.strides_count.1]
                );
            //println!("{:?}", loss_r_summed_2d);
            //println!("{:?}", loss_r_sum_slice);
            
            loss_r_input_strides += &(&loss_r_sum_slice * &vertical_filter_broadcasted);
        }

        // initialize sub tensor ranges
        let in_chan_range: Range<usize> = 0..self.in_channels;
        let mut start_y: usize = 0;
        let mut end_y: usize = self.filter_dim;
        let mut start_x: usize = 0;
        let mut end_x: usize = self.filter_dim;
        
        let input_shape: &[usize] = self.input_tensors.shape();
        let mut loss_r_next_layer: Array3<f32> = Array3::zeros(
            (input_shape[0], input_shape[1], input_shape[2])
        );

        let mut n_strides_passed: usize = 0;
        // move filter through input matrices
        /**/

        // transfer gradients in form of 2d strided input
        // back to original 3d input shape
        for _stride_y in 0..self.strides_count.0
        {
            for _stride_x in 0..self.strides_count.1
            {
                // from 3d tensor
                let mut loss_r_next_layer_slice: ArrayViewMut3<f32> = 
                    loss_r_next_layer.slice_mut(
                        s![
                            in_chan_range.clone(),
                            start_y..end_y,
                            start_x..end_x
                        ]
                    );
                //println!("{:?}", loss_r_next_layer_slice);
                
                // from 2d matrix
                let vertical_input_stride_slice: Array2<f32> = 
                    loss_r_input_strides.slice(
                        s![0..self.in_channels * self.filter_dim * self.filter_dim,
                        n_strides_passed..n_strides_passed + 1
                        ]
                    ).into_owned();
                                
                //let horizontal_input_stride: Array1<f32> = 
                //    vertical_input_stride_slice.into_shape(
                //        self.in_channels * self.filter_dim * self.filter_dim
                //    ).unwrap();
                
                //println!("{:?}", horizontal_input_stride);
                
                // final 3d version of gradients
                let input_stride_3d: Array3<f32> = 
                    vertical_input_stride_slice.into_shape(
                        (self.in_channels, self.filter_dim, self.filter_dim)
                    ).unwrap();
                
                //println!("{:?}", input_stride_3d);
                
                loss_r_next_layer_slice += &input_stride_3d;

                n_strides_passed += 1;

                // move window right
                start_x += self.strides_len.1;
                end_x += self.strides_len.1;
            }
            // move window down
            start_y += self.strides_len.0;
            end_y += self.strides_len.0;

            // move window back to left
            start_x = 0;
            end_x = self.filter_dim;
        }
        //println!("{:?}", loss_r_next_layer);
        //std::process::exit(1);
        /**/

        return loss_r_next_layer.into_dyn();
    }

    pub fn update_params(&mut self, lr: f32)
    {
        self.filter_sets -= &(lr * (&self.filter_set_grads + self.l2 * &self.filter_sets));
        self.biases -= &(lr * &self.bias_grads);
    }

    pub fn zero_grads(&mut self)
    {
        self.filter_set_grads *= 0.0;
        self.bias_grads *= 0.0;
    }

    pub fn details(&self)
    {
        println!("Num input channels: {}", self.in_channels);
        println!("Num output channels: {}", self.out_channels);
        println!("Input shape: {:?}", self.input_tensors.shape());
        println!("Output shape: {:?}", self.summed_tensors.shape());
        println!("Filters: \n{:?}", self.filter_sets);
        println!("--------------------------------------");
        println!("Biases: \n{:?}", self.biases);
        println!("=======================================");
    }

}

pub fn calculate_stride_counts(
    matrix2d_shape: (usize, usize), 
    filter_dim: usize, strides_len: (usize, usize)
) -> (usize, usize)
{
    let n_strides_vertical: f32 = 
        ((matrix2d_shape.0 as f32 - filter_dim as f32) / 
            strides_len.0 as f32 + 1.0).floor();

    let n_strides_side: f32 = 
        ((matrix2d_shape.1 as f32 - filter_dim as f32) / 
            strides_len.1 as f32 + 1.0).floor();
    
    return (n_strides_vertical as usize, n_strides_side as usize);
}


/*

fn move_window_down(window: &mut Window2D, stride: usize)
{
    let smallest_x: usize = window[0].1; 
    for coord in window
    {
        coord.0 += stride; // move down by y stride
        coord.1 -= smallest_x; // translate x coordinate back to 0, 1, 2, ...
    }
}

fn reset_window_coords(window: &mut Window2D)
{
    let smallest_y: usize = window[0].0; 
    let smallest_x: usize = window[0].1; 

    // translate x and y coordinates back to 0, 1, 2, ...
    for coord in window
    {
        coord.0 -= smallest_y; // translate y coordinate back to 0, 1, 2, ...
        coord.1 -= smallest_x; // translate x coordinate back to 0, 1, 2, ...
    }
}

fn flip_filter_weights(filter: &mut Vec<f32>)
{
    filter.reverse();
}

#[derive(Serialize, Deserialize)]
pub struct Filter
{
    pub idx_records: Vec<Vec<usize>>, // contains 2d indexes
    pub dim: usize,
    pub weights: Vec<f32>,
    pub bias: f32,
    pub weights_gradients: Vec<f32>
}
impl Filter
{
    pub fn new(dim: usize, weight_range: f32, bias_range: f32) -> Self
    {
        let mut weights: Vec<f32> = Vec::new();
        let mut weights_gradients: Vec<f32> = Vec::new();
        for _ in 0..dim
        {
            weights.push(rand::thread_rng().gen_range(-weight_range..=weight_range));
            weights_gradients.push(0.0);
        }

        return Self
        {
            idx_records: Vec::new(),
            dim,
            bias: rand::thread_rng().gen_range(-bias_range..=bias_range),
            weights,
            weights_gradients
        }
    }
}
*/