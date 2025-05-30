use std::os::raw::c_void;

use crate::neuralnet::TraversePtrs;

trait LayerCuda
{
    fn forward(
        &mut self, inputs: *mut c_void, weights: *mut c_void, use_dropout: bool
    ) -> *mut c_void;

    fn backward(&mut self);
    fn update_params(&mut self);
    fn details(&mut self);
    fn get_param_count(&self) -> f32;
    fn move_ptrs_to_arrays(&mut self);
}

// composition structs to reduce code repetition
pub struct IOPtrs
{
    pub in_shape: (usize, usize, usize),
    pub out_shape: (usize, usize, usize),
    pub output_traverse_ptr: *mut TraversePtrs,
}

pub struct ParameterPtrs
{

}

pub struct AllocationStatusPtrs
{

}

pub struct MiscPtrs
{

}