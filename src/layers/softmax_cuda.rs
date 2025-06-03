
use crate::{
    cuda_bridge::{copy_cuda_to_cuda, element_op_3d_inplace, new_cuda_array, scalar_op_3d_inplace, softmax_forward}, 
    neuralnet::TraversePtrs, 
    pointer_ops::{counter_is_zero, increment_counter, init_trav_in_ptrs, set_zero_counter}};

use super::layer_cuda::{AllocationStatus, IOPtrs, LayerCuda};

pub struct SoftmaxCuda
{
    pub name: String,
    pub io_ptrs: IOPtrs,
    pub allocation_status: AllocationStatus,

    pub input_exp_ptr: *mut f32,
    pub exp_sum_ptr: *mut f32,
    pub broadcast_temp_ptr: *mut f32,
    
    pub temperature: f32,
    pub input_grad_temp: *mut f32,

    pub backward_passes_count: u128,
    pub count: u128,
    pub batch_size: f32,
}
impl SoftmaxCuda
{
    // weight matrix initialize during first ever run
    pub fn new(batch: usize, rows: usize, cols: usize, temperature: f32, name: String) -> Self
    {
        return Self
        {   
            name,
            io_ptrs: IOPtrs::new((batch, rows, cols), (batch, rows, cols)),
            allocation_status: AllocationStatus::new(),

            input_exp_ptr: std::ptr::null_mut(),
            exp_sum_ptr: std::ptr::null_mut(),
            input_grad_temp: std::ptr::null_mut(),
            broadcast_temp_ptr: std::ptr::null_mut(),

            temperature,
            backward_passes_count: 0,
            count: 0,
            batch_size: 0.0
        }
    }
}

impl LayerCuda for SoftmaxCuda
{
    // supports batch matrix multiplication unlike cpu
    fn forward(&mut self, trav_ptr_in: *mut TraversePtrs, _trav_ptr_weight: *mut TraversePtrs, _use_dropout: bool) -> *mut TraversePtrs
    {
        let batch: usize = self.io_ptrs.in_shape.0;
        let rows: usize = self.io_ptrs.in_shape.1;
        let cols: usize = self.io_ptrs.in_shape.2;

        let flattened_shape: u32 = (batch * rows * cols) as u32;

        if !self.allocation_status.ptrs_allocated
        {   
            self.input_exp_ptr = new_cuda_array(flattened_shape);

            self.exp_sum_ptr = new_cuda_array((batch * rows * 1) as u32);
            self.broadcast_temp_ptr = new_cuda_array(flattened_shape);
            self.input_grad_temp = new_cuda_array(flattened_shape);

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

        softmax_forward(
            self.io_ptrs.input_ptr, self.input_exp_ptr, self.exp_sum_ptr,
            self.io_ptrs.output_ptr, self.broadcast_temp_ptr, self.temperature,
            batch, rows, cols, self.allocation_status.zero_output
        );

        set_zero_counter(self.io_ptrs.backward_count);
        return self.io_ptrs.output_traverse_ptr;
    }

    fn get_param_count(&self) -> usize
    {
        return 0_usize;
    }

    fn backward(&mut self, _use_dropout: bool)
    {
        copy_cuda_to_cuda(self.input_grad_temp, self.io_ptrs.output_grad_ptr, &[self.io_ptrs.in_shape.0, self.io_ptrs.in_shape.1, self.io_ptrs.in_shape.2]);
        // for temperature
        scalar_op_3d_inplace(
            self.input_grad_temp, self.temperature, 3, 
            self.io_ptrs.in_shape.0, self.io_ptrs.in_shape.1, self.io_ptrs.in_shape.2
        );

        if counter_is_zero(self.io_ptrs.backward_count_in_prev)
        {
            copy_cuda_to_cuda(
                self.io_ptrs.input_grad_ptr, self.input_grad_temp, 
                &[self.io_ptrs.in_shape.0, self.io_ptrs.in_shape.1, self.io_ptrs.in_shape.2]
            );
        }
        else
        {
            element_op_3d_inplace(
                self.io_ptrs.input_grad_ptr, self.input_grad_temp, 0, 
                self.io_ptrs.in_shape.0, self.io_ptrs.in_shape.1, self.io_ptrs.in_shape.2
            );
        }
        increment_counter(self.io_ptrs.backward_count);
        self.batch_size += 1.0;

        ////println!("=========================================================");
        ////println!("original_grad: {:?}", cuda_ptr_to_array(original_grads, &[self.shape.0, self.shape.1, self.shape.2]));
        ////println!("input_grad: {:?}", cuda_ptr_to_array(input_grad_ptr, &[self.shape.0, self.shape.1, self.shape.2]));

        //let start: Instant = Instant::now();
        //softmax_backward(
        //    original_grads, input_exp_ptr, input_grad_ptr,
        //    exp_sum_ptr, 
        //    self.in_shape.0, self.in_shape.1, self.in_shape.2
        //);
        //let end = start.elapsed();
        ////println!("backward: {:.6}", end.as_secs_f64());

        ////println!("\noriginal_grads: {:?}", cuda_ptr_to_array(ptr.get_ptr(), ptr.get_shape()));
        //exit(1);
        //ptr.set_ptr(input_grad_ptr, vec![self.in_shape.0, self.in_shape.1, self.in_shape.2]);
        ////println!("\nchained_gradients: {:?}", cuda_ptr_to_array(ptr.get_ptr(), ptr.get_shape()));
        //exit(1);
        ////println!("\nweight_gradients: {:?}", cuda_ptr_to_array(weight_grad_ptr, &[self.in_shape.0, self.in_shape.2, self.out_shape.2]));
        ////println!("\nbias_gradients: {:?}", cuda_ptr_to_array(bias_grad_ptr, &[self.out_shape.0, self.out_shape.1, self.out_shape.2]));
        ////println!("=========================================================");
        //exit(1);      
    }

    fn update_params(&mut self, _optimizer_type: i32, _lr: f32, _l2: f32, _alpha: f32, _beta: f32)
    {    
        self.batch_size = 0.0;
    }

    fn details(&self)
    {
        println!("Layer type: Softmax");
        println!("Input ptr: {:?} | Input grad ptr: {:?}", self.io_ptrs.input_ptr, self.io_ptrs.input_grad_ptr);
        println!("Output ptr: {:?} | Output grad ptr: {:?}", self.io_ptrs.output_ptr, self.io_ptrs.output_grad_ptr);
        println!("Shape: {:?}", self.io_ptrs.in_shape);
    }

    fn get_layer_id(&self) -> String {
        return self.name.clone();
    }

    fn move_ptrs_to_arrays(&mut self)
    {
    }
}