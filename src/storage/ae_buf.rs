use ndarray::ArrayD;

use crate::neuralnet::NeuralNet;

pub struct AutoencoderBuf
{
    pub array_max_len: usize,
    pub array_buf: Vec<ArrayD<f32>>,
    pub encoder_ptr: *mut NeuralNet,
    pub decoder_ptr: *mut NeuralNet,
}

impl AutoencoderBuf 
{
    pub fn new(array_max_len: usize, encoder: *mut NeuralNet, decoder: *mut NeuralNet) -> Self
    {
        return Self
        {
            array_max_len,
            array_buf: Vec::new(),
            encoder_ptr: encoder,
            decoder_ptr: decoder
        }
    }

    pub fn add_new_array(&mut self, array: ArrayD<f32>)
    {
        self.array_buf.push(array);
    }

    pub unsafe fn train_ae(&mut self)
    {
        /*
        let array_idxs: Vec<f32> = Array1::range(0.0, self.array_buf.len() as f32, 1.0).into_raw_vec();
        
        let mut total_loss: f32 = 0.0;
        let mut count: f32 = 0.0;

        for i in 0..self.array_buf.len()
        {
            //let choice_idx: usize = rand::thread_rng().gen_range(0..array_idxs.len());
            //let array_idx: usize = array_idxs[choice_idx] as usize;
            let latent_vec: ArrayD<f32> = (*self.encoder_ptr).forward(self.array_buf[i].clone());
            let reconstructed: ArrayD<f32> = (*self.decoder_ptr).forward(latent_vec);

            let loss_func: LossFn = get_loss_from_str("log_cosh_loss").unwrap();
            let loss_func_deriv: LossFnDeriv = get_loss_deriv_from_str("log_cosh_loss").unwrap();
            let loss_val: f32 = loss_func(reconstructed.clone(), self.array_buf[i].clone());
            let mut grads: ArrayD<f32> = loss_func_deriv(reconstructed.clone(), self.array_buf[i].clone());

            //array_idxs.remove(choice_idx);

            (*self.encoder_ptr).zero_grads();
            (*self.decoder_ptr).zero_grads();

            grads = (*self.decoder_ptr).backward(grads);
            (*self.encoder_ptr).backward(grads);

            //(*self.encoder_ptr).update_params(0.001);
            //(*self.decoder_ptr).update_params(0.001);
            total_loss += loss_val;
            count += 1.0
            */

        //return total_loss / count;


    }
}