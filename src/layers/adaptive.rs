use ndarray::{ArrayD, IxDyn, Zip};
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct AdaptiveSwish
{
    pub input_tensor: ArrayD<f32>,
    pub a: ArrayD<f32>,
    pub b: ArrayD<f32>,
    pub c: ArrayD<f32>,
    pub a_grads: ArrayD<f32>,
    pub b_grads: ArrayD<f32>,
    pub c_grads: ArrayD<f32>,

}
impl AdaptiveSwish
{
    pub fn new() -> Self
    {
        return Self
        {
            input_tensor: ArrayD::zeros(IxDyn(&[0])),
            a: ArrayD::zeros(IxDyn(&[0])),
            b: ArrayD::zeros(IxDyn(&[0])),
            c: ArrayD::zeros(IxDyn(&[0])),
            a_grads: ArrayD::zeros(IxDyn(&[0])),
            b_grads: ArrayD::zeros(IxDyn(&[0])),
            c_grads: ArrayD::zeros(IxDyn(&[0])),
        };
    }

    pub fn forward(&mut self, input_tensor: ArrayD<f32>) -> ArrayD<f32>
    {
        if self.input_tensor.shape() == &[0]
        {
            let input_shape: &[usize] = input_tensor.shape();
            self.input_tensor = ArrayD::zeros(input_shape);

            self.a = ArrayD::ones(input_shape);
            self.b = ArrayD::ones(input_shape);
            self.c = ArrayD::zeros(input_shape);

            self.a_grads = ArrayD::zeros(input_shape);
            self.b_grads = ArrayD::zeros(input_shape);
            self.c_grads = ArrayD::zeros(input_shape);

            self.a.par_map_inplace(
                |val: &mut f32|
                *val = rand::thread_rng().gen_range(-1.0..1.0)
            );
            self.b.par_map_inplace(
                |val: &mut f32|
                *val = rand::thread_rng().gen_range(-1.0..1.0)
            );
            self.c.par_map_inplace(
                |val: &mut f32|
                *val = rand::thread_rng().gen_range(-1.0..1.0)
            );
        }

        // output = b * (1 / (1 + e^-ax)) + c
        self.input_tensor.fill(0.0);
        self.input_tensor += &input_tensor;

        let a_input_tensor: ArrayD<f32> = &self.input_tensor * &self.a;
        let sigmoid_result: ArrayD<f32> = 
            1.0 / (1.0 + a_input_tensor.mapv(|v: f32| (-v).exp()));
        
        let b_x_sigmoid_result: ArrayD<f32> = 
            &self.b * &self.input_tensor * sigmoid_result;

        let output: ArrayD<f32> = b_x_sigmoid_result + &self.c;
        return output;
    
    }

    pub fn backward(&mut self, loss_r_output: ArrayD<f32>) -> ArrayD<f32>
    {
        // calculate gradient of c
        self.c_grads += &(1.0 * &loss_r_output);

        // calculate gradient of b
        let a_input_tensor: ArrayD<f32> = &self.input_tensor * &self.a;
        let a_input_tensor_e: ArrayD<f32> = a_input_tensor.mapv(|v: f32| (-v).exp());
        let output_r_b: ArrayD<f32> = &self.input_tensor / (1.0 + &a_input_tensor_e);
        self.b_grads += &(output_r_b * &loss_r_output);
        
        // calculate gradient of a
        let numerator: ArrayD<f32> = 
            &self.b * (&self.input_tensor * &self.input_tensor) * &a_input_tensor_e;
        
        let denominator: ArrayD<f32> = 
            (1.0 + &a_input_tensor_e) * (1.0 + &a_input_tensor_e);

        let output_r_a: ArrayD<f32> = numerator / &denominator;
        self.a_grads += &(output_r_a * &loss_r_output);

        // calculate gradient of x (input)
        let numerator: ArrayD<f32> = 
            &self.b * (1.0 + &a_input_tensor_e + &self.a * &self.input_tensor * a_input_tensor_e);
        
        let output_r_inputs: ArrayD<f32> = numerator / denominator;

        return output_r_inputs * loss_r_output;

    }

    pub fn update_params(&mut self, lr: f32)
    {
        self.a -= &(lr * &self.a_grads);
        self.b -= &(lr * &self.b_grads);
        self.c -= &(lr * &self.c_grads);
    }

    pub fn zero_grads(&mut self)
    {
        self.a_grads.fill(0.0);
        self.b_grads.fill(0.0);
        self.c_grads.fill(0.0);
    }

    pub fn details(&self)
    {
        println!("Layer type: ADAPTIVE_SWISH");
        println!("a: {:?}", self.a);
        println!("------------------");
        println!("b: {:?}", self.b);
        println!("------------------");
        println!("c: {:?}", self.c);
        println!("=======================================");
    }
}




////////////////////////////////////////////////////////////////////////////

#[derive(Serialize, Deserialize)]
pub struct AdaptiveReLU
{
    pub input_tensor: ArrayD<f32>,
    pub output_tensor: ArrayD<f32>,
    pub a: ArrayD<f32>,
    pub b: ArrayD<f32>,
    pub a_grads: ArrayD<f32>,
    pub b_grads: ArrayD<f32>,
}
impl AdaptiveReLU
{
    pub fn new() -> Self
    {
        return Self
        {
            input_tensor: ArrayD::zeros(IxDyn(&[0])),
            output_tensor: ArrayD::zeros(IxDyn(&[0])),
            a: ArrayD::zeros(IxDyn(&[0])),
            b: ArrayD::zeros(IxDyn(&[0])),
            a_grads: ArrayD::zeros(IxDyn(&[0])),
            b_grads: ArrayD::zeros(IxDyn(&[0])),
        }
    }

    pub fn forward(&mut self, input_tensor: ArrayD<f32>) -> ArrayD<f32>
    {
        if self.input_tensor.shape() == &[0]
        {
            let input_shape: &[usize] = input_tensor.shape();
            self.input_tensor = ArrayD::zeros(input_shape);

            self.a = ArrayD::ones(input_shape);
            self.b = ArrayD::zeros(input_shape);
            self.a_grads = ArrayD::zeros(input_shape);
            self.b_grads = ArrayD::zeros(input_shape);

            self.output_tensor = ArrayD::zeros(input_shape);

            self.a.par_map_inplace(
                |val: &mut f32|
                *val = rand::thread_rng().gen_range(-1.0..1.0)
            );
            self.b.par_map_inplace(
                |val: &mut f32|
                *val = rand::thread_rng().gen_range(-0.01..0.01)
            );
        }

        self.input_tensor.fill(0.0);
        self.output_tensor.fill(0.0);
        self.input_tensor += &input_tensor;

        Zip::from(&mut self.output_tensor)
            .and(&self.input_tensor)
            .and(&self.a)
            .and(&self.b)
            .for_each(
                |o: &mut f32, i: &f32, a: &f32, b: &f32|
                if *i >= 0.0
                {
                    *o = i * a;
                }
                else if *i < 0.0
                {
                    *o = i * b;
                }
            );
        
        return self.output_tensor.clone();

    }

    pub fn backward(&mut self, loss_r_outputs: ArrayD<f32>) -> ArrayD<f32>
    {
        // calculate gradient respect to a and b

        // y = a * x
        // dy/da = x
        Zip::from(&loss_r_outputs)
            .and(&self.input_tensor)
            .and(&mut self.a_grads)
            .and(&mut self.b_grads)
            .for_each(
                |o: &f32, i: &f32, a: &mut f32, b: &mut f32|
                if *i >= 0.0
                {
                    *a += i * o;
                }
                else if *i < 0.0
                {
                    *b += i * o;
                }
            );
        
        // calculate gradient respect to input
        let input_shape: &[usize] = self.input_tensor.shape();
        let mut loss_r_inputs: ArrayD<f32> = ArrayD::zeros(input_shape);
        
        // y = a * x
        // dy/dx = a
        Zip::from(&loss_r_outputs)
            .and(&mut loss_r_inputs)
            .and(&self.input_tensor)
            .and(&self.a)
            .and(&self.b)
            .for_each(
                |o: &f32, l_r_i: &mut f32, i: &f32, a: &f32, b: &f32|
                if *i >= 0.0
                {
                    *l_r_i += a * o;
                }
                else if *i < 0.0
                {
                    *l_r_i += b * o;
                }
            );
        
        return loss_r_inputs;
    }

    pub fn update_params(&mut self, lr: f32)
    {
        self.a -= &(lr * &self.a_grads);
        self.b -= &(lr * &self.b_grads);
    }

    pub fn zero_grads(&mut self)
    {
        self.a_grads.fill(0.0);
        self.b_grads.fill(0.0);
    }

    pub fn details(&self)
    {
        println!("Layer type: ADAPTIVE_RELU");
        println!("a: {:?}", self.a);
        println!("------------------");
        println!("b: {:?}", self.b);
        println!("=======================================");
    }
}

#[derive(Serialize, Deserialize)]
pub struct AdaptiveTanh
{
    pub input_tensor: ArrayD<f32>,
    pub processed_input_tensor: ArrayD<f32>,
    pub a: ArrayD<f32>, // steepness of function
    pub b: ArrayD<f32>, // translation along x axis
    pub a_grads: ArrayD<f32>,
    pub b_grads: ArrayD<f32>,
    pub amplification: f32
}

impl AdaptiveTanh
{
    pub fn new(amplification: f32) -> Self
    {
        return Self
        {
            input_tensor: ArrayD::zeros(IxDyn(&[0])),
            processed_input_tensor: ArrayD::zeros(IxDyn(&[0])),
            a: ArrayD::zeros(IxDyn(&[0])),
            b: ArrayD::zeros(IxDyn(&[0])),
            a_grads: ArrayD::zeros(IxDyn(&[0])),
            b_grads: ArrayD::zeros(IxDyn(&[0])),
            amplification
        }
    }

    pub fn forward(&mut self, input_tensor: ArrayD<f32>) -> ArrayD<f32>
    {
        if self.input_tensor.shape() == &[0]
        {
            let input_shape: &[usize] = input_tensor.shape();
            self.input_tensor = ArrayD::zeros(IxDyn(input_shape));
            self.processed_input_tensor = ArrayD::zeros(IxDyn(input_shape));
            self.a = ArrayD::ones(IxDyn(input_shape));
            self.b = ArrayD::zeros(IxDyn(input_shape));
            self.a_grads = ArrayD::zeros(IxDyn(input_shape));
            self.b_grads = ArrayD::zeros(IxDyn(input_shape));

            self.a.map_mut(
                |v: &mut f32|
                *v = rand::thread_rng().gen_range(-1.0..1.0)
            );
        }

        self.input_tensor.fill(0.0);
        self.processed_input_tensor.fill(0.0);

        // calculate tanh(a*x - b)
        self.input_tensor += &input_tensor;
        self.processed_input_tensor += &input_tensor;
        self.processed_input_tensor *= &self.a;
        self.processed_input_tensor -= &self.b;

        let mut activated_tensor: ArrayD<f32> = self.processed_input_tensor.mapv(
            |v: f32|
            v.tanh()
        );

        activated_tensor *= self.amplification;

        return activated_tensor;

    }

    pub fn backward(&mut self, loss_r_output: ArrayD<f32>) -> ArrayD<f32>
    {
        // output = amplification * tanh_result
        let output_r_tanh: f32 = self.amplification;

        // tanh_result = tanh(a*x - b)

        // respect to a
        // x * (1 - tanh(a*x - b)^2)
        let tanh_r_a: ArrayD<f32> = 
            (1.0 - self.processed_input_tensor.mapv(
                |v: f32|
                (v.tanh()).powf(2.0)
            )) * &self.input_tensor;
        
        self.a_grads += &(&loss_r_output * output_r_tanh * tanh_r_a);

        // respect to b
        // -1 * (1 - tanh(a*x - b)^2)
        let tanh_r_b: ArrayD<f32> = 
            (1.0 - self.processed_input_tensor.mapv(
                |v: f32|
                (v.tanh()).powf(2.0)
            )) * -1.0;
        
        self.b_grads += &(&loss_r_output * output_r_tanh * tanh_r_b);

        // respect to x
        // a * (1 - tanh(a*x - b)^2)
        let tanh_r_inputs: ArrayD<f32> = 
            (1.0 - self.processed_input_tensor.mapv(
                |v: f32|
                (v.tanh()).powf(2.0)
            )) * &self.a;

        return loss_r_output * output_r_tanh * tanh_r_inputs;

    }

    pub fn update_params(&mut self, lr: f32)
    {
        self.a -= &(lr * &self.a_grads);
        self.b -= &(lr * &self.b_grads);
    }

    pub fn zero_grads(&mut self)
    {
        self.a_grads.fill(0.0);
        self.b_grads.fill(0.0);
    }

    pub fn details(&self)
    {
        println!("Layer type: ADAPTIVE_TANH");
        println!("a: {:?}", self.a);
        println!("------------------");
        println!("b: {:?}", self.b);
        println!("=======================================");
    }
}