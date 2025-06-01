use serde::{Deserialize, Serialize};

use crate::{cuda_bridge::copy_cuda_to_cuda, pointer_ops::{increment_counter, init_layer_connections, set_zero_counter, string_to_ptr}};

#[derive(Serialize, Deserialize)]
pub struct CLSCuda
{
    pub io_ptrs: IOPtrs,
    pub allocation_status: AllocationStatus,

    pub token_idx: usize,
    pub batch_size: f32,
    pub count: u128,
}
impl CLSCuda
{
    // weight matrix initialize during first ever run
    pub fn new(
        in_batch: usize, in_rows: usize, in_cols: usize, 
        token_idx: usize
    ) -> Self
    {
        return Self
        {
            io_ptrs: IOPtrs::new(
                (in_batch, in_rows, in_cols), 
                (in_batch, 1, in_cols)
            ),
            allocation_status: AllocationStatus::new(),
            token_idx,
            batch_size: 0.0,
            count: 0,
        }
    }
}

impl LayerCuda for CLSCuda
{
    fn forward(&mut self, trav_ptr_in: *mut TraversePtrs, _trav_ptr_weight: *mut TraversePtrs, _use_dropout: bool) -> *mut TraversePtrs
    {
        if !self.allocation_status.ptrs_allocated
        {
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

        // pointer arithmetic to select token embedding at index
        let chosen_embedding_ptr: *mut f32 = unsafe { self.io_ptrs.input_ptr.add(self.token_idx * self.io_ptrs.in_shape.2) };
        copy_cuda_to_cuda(self.io_ptrs.output_ptr, chosen_embedding_ptr, &[self.io_ptrs.in_shape.2]);

        set_zero_counter(self.io_ptrs.backward_count);

        //println!("input: {:?}\n", cuda_ptr_to_array(input_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
        //println!("results: {:?}\n", cuda_ptr_to_array(result_ptr, &[1, 1, self.shape.2]));

        return self.io_ptrs.output_traverse_ptr;

    }

    fn backward(&mut self, _use_dropout: bool)
    {
        let chosen_dst_ptr: *mut f32 = unsafe { self.io_ptrs.input_grad_ptr.add(self.token_idx * self.io_ptrs.in_shape.2) };
        copy_cuda_to_cuda(chosen_dst_ptr, self.io_ptrs.output_grad_ptr, &[self.io_ptrs.in_shape.2]);

        increment_counter(self.io_ptrs.backward_count);

        self.batch_size += 1.0;
        
        //let end = start.elapsed();
        //println!("backward: {:.6}", end.as_secs_f64());

        
        //println!("original_grads: {:?}\n", cuda_ptr_to_array(&[self.shape.0, self.shape.1, self.shape.2]));
        //ptr.set_ptr(input_grad_ptr, vec![self.shape.0, self.shape.1, self.shape.2]);
        //println!("input_gradients: {:?}\n", cuda_ptr_to_array(ptr.get_ptr(), ptr.get_shape()));
        //println!("weight_gradients: {:?}\n", cuda_ptr_to_array(input_grad_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
        //println!("mask: {:?}\n", cuda_ptr_to_array(mask_ptr, &[self.shape.0, self.shape.1, self.shape.2]));
        //println!("=========================================================");
        //exit(1);      
    }

    fn update_params(&mut self, _optimizer_type: i32, _lr: f32, _l2: f32, _alpha: f32, _beta: f32)
    {   
        self.batch_size = 0.0;
    }

    pub fn zero_io(&mut self, io_ptr_name: &String)
    {
        //let io_ptr: *mut f32 = string_to_ptr(self.io_ptrs.get(io_ptr_name).unwrap());

        if io_ptr_name.contains("input")
        {
            //zeroes_3d_inplace(io_ptr, self.in_shape.0, self.in_shape.1, self.in_shape.2);
            self.zero_input_grad = true;
        }
        
        if io_ptr_name.contains("output")
        {
            //zeroes_3d_inplace(io_ptr, self.shape.0, self.shape.1, self.shape.2);
            self.zero_output = true;
        }
    }

    pub fn details(&self)
    {
        println!("Layer type: CLS");
        println!("Input ptr: {:?} | Input grad ptr: {:?}", self.input_ptr, self.input_grad_ptr);
        println!("Output ptr: {:?} | Output grad ptr: {:?}", self.output_ptr, self.output_grad_ptr);
        println!("Input shape: {:?}", self.shape);
        println!("CLS token index: {:?}", self.token_idx);
    }

    pub fn get_param_count(&self) -> usize
    {
        return 0_usize;
    }

    pub fn move_ptrs_to_arrays(&mut self)
    {
        //self.weights = cuda_ptr_to_array(
        //    string_to_ptr(self.io_ptrs.get("weight").unwrap()), 
        //    &[self.shape.0, self.shape.1, self.shape.2]
        //);
        //self.biases = cuda_ptr_to_array(string_to_ptr(&self.biases_ptr), &[self.shape.0, self.shape.1, self.shape.2]);
        //free_cuda_array(string_to_ptr(&self.weight_ptr));
        //free_cuda_array(string_to_ptr(&self.biases_ptr));
        //free_cuda_array(string_to_ptr(&self.input_ptr));
        //free_cuda_array(string_to_ptr(&self.result_ptr));
        //free_cuda_array(string_to_ptr(&self.input_grads_ptr));
        //free_cuda_array(string_to_ptr(&self.output_grads_ptr));
        //free_cuda_array(string_to_ptr(&self.weight_gradients_ptr));
        //free_cuda_array(string_to_ptr(&self.bias_gradients_ptr));
        self.in_out_ptrs_allocated = false;
    }

    pub fn set_ptrs_allocated(&mut self)
    {
        self.in_out_ptrs_allocated = true;
    }
}