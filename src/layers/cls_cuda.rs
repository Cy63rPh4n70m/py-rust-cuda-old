use crate::{
    cuda_bridge::copy_cuda_to_cuda, neuralnet::TraversePtrs, 
    pointer_ops::{increment_counter, init_trav_in_ptrs, set_zero_counter}};

use super::layer_cuda::{AllocationStatus, IOPtrs, LayerCuda};

/// obtains only a 1D slice of a 2D array
/// specifically for language/sequence processing
/// where only one contextually rich token is selected
/// for classification purposes
pub struct CLSCuda
{
    pub io_ptrs: IOPtrs,
    pub allocation_status: AllocationStatus,

    pub token_idx: usize,
    pub batch_size: f32,
}

// implement constructor
impl CLSCuda
{
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
        }
    }
}

// trait implementation
impl LayerCuda for CLSCuda
{
    fn forward(&mut self, trav_ptr_in: *mut TraversePtrs, _trav_ptr_weight: *mut TraversePtrs, _use_dropout: bool) -> *mut TraversePtrs
    {
        if !self.allocation_status.ptrs_allocated
        {
            // connect the output pointers of the previous layer with the 
            // current layer's input pointers
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

        // - pointer arithmetic to shift by number of rows in a 2D matrix 
        // - token index n must be multiplied by the number of column to get the 
        //   correct pointer of the nth row of the matrix
        let chosen_embedding_ptr: *mut f32 = unsafe { self.io_ptrs.input_ptr.add(self.token_idx * self.io_ptrs.in_shape.2) };
        copy_cuda_to_cuda(self.io_ptrs.output_ptr, chosen_embedding_ptr, &[self.io_ptrs.in_shape.2]);

        // important for zeroing gradients during backward pass
        set_zero_counter(self.io_ptrs.backward_count);

        return self.io_ptrs.output_traverse_ptr;

    }

    fn backward(&mut self, _use_dropout: bool)
    {
        // - pointer arithmetic to shift by number of rows in a 2D matrix 
        // - token index n must be multiplied by the number of column to get the 
        //   correct pointer of the nth row of the matrix
        let chosen_dst_ptr: *mut f32 = unsafe { self.io_ptrs.input_grad_ptr.add(self.token_idx * self.io_ptrs.in_shape.2) };
        copy_cuda_to_cuda(chosen_dst_ptr, self.io_ptrs.output_grad_ptr, &[self.io_ptrs.in_shape.2]);

        // increment counter to tell other layers connected to the same previous layer
        // to accumulate the gradient instead of zeroing it first
        increment_counter(self.io_ptrs.backward_count_in_prev);

        self.batch_size += 1.0;
    }

    fn update_params(&mut self, _optimizer_type: i32, _lr: f32, _l2: f32, _alpha: f32, _beta: f32)
    {   
        self.batch_size = 0.0;
    }

    fn details(&self)
    {
        // print all details of layer (e.g. IO shape, weights, etc)
        println!("Layer type: CLS");
        println!("Input ptr: {:?} | Input grad ptr: {:?}", self.io_ptrs.input_ptr, self.io_ptrs.input_grad_ptr);
        println!("Output ptr: {:?} | Output grad ptr: {:?}", self.io_ptrs.output_ptr, self.io_ptrs.output_grad_ptr);
        println!("Input shape: {:?}", self.io_ptrs.in_shape);
        println!("CLS token index: {:?}", self.token_idx);
    }
}