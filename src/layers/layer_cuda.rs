use std::os::raw::c_void;

trait LayerCuda
{
    fn forward(
        &mut self, input: *mut c_void, weight: *mut c_void, use_dropout: bool
    ) -> *mut c_void;

    fn backward(&mut self);
    fn update_params(&mut self);
    fn details(&mut self);
    fn get_param_count(&self) -> f32;
    fn move_ptrs_to_arrays(&mut self);
}

// contains the usual pointers each layer will typically require
pub struct LayerBaseStruct
{
    
}