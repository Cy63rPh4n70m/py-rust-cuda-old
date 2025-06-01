use crate::{

use super::layer_cuda::{AllocationStatus, IOPtrs, LayerCuda, ParameterPtrs, WeightTensors};

pub struct Embedding2DCuda
{
    pub io_ptrs: IOPtrs,
    pub parameter_ptrs: ParameterPtrs,
    pub allocation_status: AllocationStatus,
    pub weight_tensors: WeightTensors,

    pub vocab_size: usize,
    pub embedding_len: usize,
    pub seq_len: usize,

    pub embedding_lookup_grad_count_ptr: *mut f32,
    pub embedding_lookup_grad_temp_ptr: *mut f32,

    pub batch_size: f32,
}
impl Embedding2DCuda
{
    // weight matrix initialize during first ever run
    pub fn new(vocab_size: usize, embedding_len: usize, seq_len: usize) -> Self
    {
        return Self
        {
            embedding_len,
            vocab_size,
            seq_len,

            io_ptrs: IOPtrs::new((1, 1, seq_len), (1, seq_len, embedding_len)),
            parameter_ptrs: ParameterPtrs::new(),
            weight_tensors: WeightTensors::new(),
            allocation_status: AllocationStatus::new(),
            
            embedding_lookup_grad_count_ptr: std::ptr::null_mut(),
            embedding_lookup_grad_temp_ptr: std::ptr::null_mut(),
            batch_size: 0.0,
        }
    }
}

impl LayerCuda for Embedding2DCuda
{
    // supports batch matrix multiplication unlike cpu
    fn forward(&mut self, trav_ptr_in: *mut TraversePtrs, _trav_ptr_weight: *mut TraversePtrs, _use_dropout: bool) -> *mut TraversePtrs
    {
        if !self.allocation_status.ptrs_allocated
        {   

            let embedding_lookup_len: u32 = (1 * self.vocab_size * self.embedding_len) as u32;
            if !self.allocation_status.arrays_allocated
            {   
                let range: f32 = (6.0 / ((self.embedding_len + self.embedding_len) as f32)).sqrt();
                self.weight_tensors.weight = random_float_vec(
                    1 * self.vocab_size * self.embedding_len, 
                    -range, range
                );
            }

            self.parameter_ptrs.weight_ptr = vec_to_cuda_ptr(&mut self.weight_tensors.weight);
            self.parameter_ptrs.weight_grad_ptr = new_cuda_array(embedding_lookup_len);
            self.parameter_ptrs.weight_vel_ptr = new_cuda_array(embedding_lookup_len);
            self.parameter_ptrs.weight_moment_ptr = new_cuda_array(embedding_lookup_len);
            self.embedding_lookup_grad_count_ptr = new_cuda_array(embedding_lookup_len);
            self.embedding_lookup_grad_temp_ptr = new_cuda_array(embedding_lookup_len);

            init_trav_in_ptrs(
                &trav_ptr_in, &mut self.io_ptrs.backward_count,
                &mut self.io_ptrs.backward_count_in_prev, 
                &mut self.io_ptrs.input_ptr, &mut self.io_ptrs.input_grad_ptr, 
                &mut self.io_ptrs.output_ptr, &mut self.io_ptrs.output_grad_ptr, 
                &mut self.io_ptrs.output_traverse_ptr, 
                (self.io_ptrs.out_shape.0 * self.io_ptrs.out_shape.1 * self.io_ptrs.out_shape.2) as usize
            );

            self.allocation_status.ptrs_allocated = true;
            self.allocation_status.arrays_allocated = true;
        }

        ////println!("{:?}", cuda_ptr_to_array(input_ptr, &[self.in_shape.0, self.in_shape.1, self.in_shape.2]));
        ////println!("{:?}", cuda_ptr_to_array(embedding_lookup_ptr, &[1, self.vocab_size, self.embedding_len]));
        ////println!("{:?}", self.embedding_lookup_mat);
        ////println!("{:?}", cuda_ptr_to_array(result_ptr, &[self.out_shape.0, self.out_shape.1, self.out_shape.2]));

        embedding_forward(
            self.io_ptrs.input_ptr, self.seq_len, 
            self.parameter_ptrs.weight_ptr, self.vocab_size, self.embedding_len, 
            self.io_ptrs.output_ptr
        );

        set_zero_counter(self.io_ptrs.backward_count);

        //println!("{:?}", cuda_ptr_to_array(input_ptr, &[self.in_shape.0, self.in_shape.1, self.in_shape.2]));
        //println!("{:?}", cuda_ptr_to_array(embedding_lookup_ptr, &[1, self.vocab_size, self.embedding_len]));
        //println!("{:?}", cuda_ptr_to_array(result_ptr, &[1, self.in_shape.2, self.embedding_len]));

        return self.io_ptrs.output_traverse_ptr;
    }

    fn backward(&mut self, _use_dropout: bool)
    {
        ////println!("{:?}", cuda_ptr_to_array(ptr.get_ptr(), &[self.out_shape.0, self.out_shape.1, self.out_shape.2]));
        ////println!("{:?}", cuda_ptr_to_array(embedding_lookup_grad_ptr, &[1, self.vocab_size, self.embedding_len]));

        embedding_backward(
            self.io_ptrs.input_ptr, self.seq_len, 
            self.parameter_ptrs.weight_ptr, self.vocab_size, self.embedding_len, 
            self.parameter_ptrs.weight_grad_ptr,
            self.embedding_lookup_grad_temp_ptr,
            self.embedding_lookup_grad_count_ptr,
            self.io_ptrs.output_grad_ptr, 
            self.io_ptrs.input_grad_ptr
        );

        self.batch_size += 1.0;

        //println!("{:?}", cuda_ptr_to_array(input_ptr, &[1, 1, self.seq_len]));
        //println!("{:?}", cuda_ptr_to_array(embedding_lookup_grad_ptr, &[1, self.vocab_size, self.embedding_len]));
        //exit(1);
    }

    fn update_params(&mut self, optimizer_type: i32, lr: f32, l2: f32, alpha: f32, beta: f32)
    {   
        gradient_desc_3d(
            lr, l2,
            self.parameter_ptrs.weight_ptr, self.parameter_ptrs.weight_grad_ptr,
            self.parameter_ptrs.weight_vel_ptr, self.parameter_ptrs.weight_moment_ptr,
            1, self.vocab_size, self.embedding_len,

            self.parameter_ptrs.weight_ptr, self.parameter_ptrs.weight_grad_ptr,
            self.parameter_ptrs.weight_vel_ptr, self.parameter_ptrs.weight_moment_ptr,
            1, self.vocab_size, self.embedding_len,

            true, false, self.batch_size, optimizer_type, alpha, beta
        );

        self.batch_size = 0.0;
    }

    fn details(&self)
    {
        println!("Layer type: Embedding2D");
        println!("Vocab shape: {:?}", (1, self.vocab_size, self.embedding_len));
        println!("Input ptr: {:?} | Input grad ptr: {:?}", self.io_ptrs.input_ptr, self.io_ptrs.input_grad_ptr);
        println!("Output ptr: {:?} | Output grad ptr: {:?}", self.io_ptrs.output_ptr, self.io_ptrs.output_grad_ptr);
        println!("Embedding matrix:");
        println!("{:?}", self.weight_tensors.weight);
    }

    fn get_param_count(&self) -> usize
    {
        return self.vocab_size * self.embedding_len;
    }

    fn move_ptrs_to_arrays(&mut self)
    {
        //free_cuda_array(string_to_ptr(&self.embedding_lookup_ptr));
        //free_cuda_array(string_to_ptr(&self.embedding_lookup_grad_ptr));
        self.weight_tensors.weight = cuda_ptr_to_vec(
            self.parameter_ptrs.weight_ptr, 
            1 * self.vocab_size * self.embedding_len
        );
    }
}