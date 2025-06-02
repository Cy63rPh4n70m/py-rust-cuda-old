use crate::{
    cuda_bridge::{cuda_ptr_from_pinned, free_cpu_array, new_cpu_pinned_array, 
        new_cuda_array, to_cpu, to_cuda}, neuralnet::TraversePtrs};

pub fn vec_to_cuda_ptr(array: &mut Vec<f32>) -> *mut f32
{
    let ptr: *mut f32 = array.as_mut_ptr();
    let cuda_ptr: *mut f32 = to_cuda(ptr, array.len() as u32);
    return cuda_ptr;
}

pub fn cuda_ptr_to_vec(ptr: *mut f32, length: usize) -> Vec<f32>
{
    unsafe 
    {
        let result: *mut f32 = to_cpu(ptr, length as u32);
        let result_slice: &mut [f32] = std::slice::from_raw_parts_mut(result, length);
        let result_vec: Vec<f32> = result_slice.to_vec();
        
        free_cpu_array(result);

        return result_vec;
    }
}

pub fn create_host_and_cuda_ptr(flattened_shape: usize) -> (*mut f32, *mut f32)
{
    let pinned_ptr: *mut f32 = new_cpu_pinned_array(flattened_shape as u32);
    let cuda_ptr: *mut f32 = cuda_ptr_from_pinned(pinned_ptr);

    return (pinned_ptr, cuda_ptr)
}

pub fn increment_counter(backward_pass_count: *mut usize)
{
    unsafe { *backward_pass_count += 1 }
}

pub fn set_zero_counter(backward_pass_count: *mut usize)
{
    unsafe { *backward_pass_count = 0 }
}

pub fn counter_is_zero(backward_pass_count: *mut usize) -> bool
{
    if !backward_pass_count.is_null()
    {
        unsafe 
        { 
            if *backward_pass_count == 0
            {
                return true;
            }
            else
            {
                return false;
            }
        }
    }
    else
    {
        return false;
    }
}

pub fn init_trav_in_ptrs(
    trav_in_ptrs: &*mut TraversePtrs, backward_count: &mut *mut usize,
    backward_count_in_prev: &mut *mut usize,
    input_ptr: &mut *mut f32, input_grad_ptr: &mut *mut f32, 
    output_ptr: &mut *mut f32, output_grad_ptr: &mut *mut f32,
    output_traverse_ptr: &mut *mut TraversePtrs,
    output_len: usize
)
{
    unsafe
    {
        *backward_count = Box::into_raw(Box::new(0_usize));

        *backward_count_in_prev = (**trav_in_ptrs).backward_pass_count;
        *input_ptr = (**trav_in_ptrs).ptr;
        *input_grad_ptr = (**trav_in_ptrs).grad_ptr;

        // make output ptrs and the traverse pointer for next layer/s
        *output_ptr = new_cuda_array(output_len as u32);
        *output_grad_ptr = new_cuda_array(output_len as u32);

        *output_traverse_ptr = Box::into_raw(Box::new({
            TraversePtrs {
                ptr: *output_ptr,
                grad_ptr: *output_grad_ptr,
                backward_pass_count: *backward_count
            }
        }));
    }
}