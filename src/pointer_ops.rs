

use std::os::raw::{c_void, c_char};

use ndarray::{ArrayD, IxDyn};

use crate::{cuda_bridge::{cuda_ptr_from_pinned, free_cpu_array, new_cpu_array, new_cpu_pinned_array, new_cuda_array, to_cpu, to_cpu_f16, to_cuda}, neuralnet::TraversePtrs};

pub fn ptr_to_string(ptr: *mut f32) -> String
{
    return format!("{:p}", ptr);
}

pub fn ptr_to_string_f16(ptr: *mut half::f16) -> String
{
    return format!("{:p}", ptr);
}

pub fn ptr_to_string_int(ptr: *mut i32) -> String
{
    return format!("{:p}", ptr);
}

pub fn string_to_ptr(string: &String) -> *mut f32
{
    let ptr_string: usize = usize::from_str_radix(&string[2..], 16).unwrap();
    return ptr_string as *mut f32;
}

pub fn string_to_ptr_f16(string: &String) -> *mut half::f16
{
    let ptr_string: usize = usize::from_str_radix(&string[2..], 16).unwrap();
    return ptr_string as *mut half::f16;
}

pub fn string_to_ptr_int(string: &String) -> *mut i32
{
    let ptr_string: usize = usize::from_str_radix(&string[2..], 16).unwrap();
    return ptr_string as *mut i32;
}

pub fn string_to_ptr_void(string: &String) -> *mut c_void
{
    let ptr_string: usize = usize::from_str_radix(&string[2..], 16).unwrap();
    return ptr_string as *mut c_void;
}

pub fn ptr_to_string_void(ptr: *mut c_void) -> String
{
    return format!("{:p}", ptr);
}

pub fn string_to_char_ptr(string: &String) -> *mut c_char
{
    let ptr_string: usize = usize::from_str_radix(&string[2..], 16).unwrap();
    return ptr_string as *mut c_char;
}

pub fn char_ptr_to_string(ptr: *mut c_char) -> String
{
    return format!("{:p}", ptr);
}

pub fn array_to_cuda_ptr_str(array: &mut ArrayD<f32>) -> String
{
    return ptr_to_string(array_to_cuda_ptr(array));
}

pub fn array_to_cuda_ptr(array: &mut ArrayD<f32>) -> *mut f32
{
    let ptr: *mut f32 = array.as_mut_ptr();
    let shape: &[usize] = array.shape();
    let mut length: u32 = 1;
    for i in 0..shape.len()
    {
        length *= shape[i] as u32;
    }
    //println!("{:?}", length);
    let cuda_ptr: *mut f32 = to_cuda(ptr, length);
    return cuda_ptr;
}

pub fn vec_to_cuda_ptr(array: &mut Vec<f32>) -> *mut f32
{
    let ptr: *mut f32 = array.as_mut_ptr();
    let cuda_ptr: *mut f32 = to_cuda(ptr, array.len() as u32);
    return cuda_ptr;
}

pub fn new_cpu_ptr_str(shape: &[usize]) -> String
{
    let mut length: usize = 1;
    for i in 0..shape.len()
    {
        length *= shape[i];
    }
    let cpu_ptr: *mut f32 = new_cpu_array(length as u32);
    return ptr_to_string(cpu_ptr);
}

pub fn new_cuda_ptr_str(shape: &[usize]) -> String
{
    let mut length: usize = 1;
    for i in 0..shape.len()
    {
        length *= shape[i];
    }
    let cuda_ptr: *mut f32 = new_cuda_array(length as u32);
    return ptr_to_string(cuda_ptr);
}

pub fn cuda_ptr_to_array(ptr: *mut f32, shape: &[usize]) -> ArrayD<f32>
{
    unsafe {
        let mut length: u32 = 1;
        for i in 0..shape.len()
        {
            length *= shape[i] as u32;
        }

        let result: *mut f32 = to_cpu(ptr, length);
        let result_slice: &mut [f32] = std::slice::from_raw_parts_mut(result, length as usize);
        let result_array: ArrayD<f32> 
            = ArrayD::from_shape_vec(
                IxDyn(shape), result_slice.to_vec()
            ).unwrap();
        
        free_cpu_array(result);

        return result_array;
    }
}

pub fn cuda_ptr_to_vec(ptr: *mut f32, shape: &[usize]) -> Vec<f32>
{
    unsafe {
        let mut length: u32 = 1;
        for i in 0..shape.len()
        {
            length *= shape[i] as u32;
        }

        let result: *mut f32 = to_cpu(ptr, length);
        let result_slice: &mut [f32] = std::slice::from_raw_parts_mut(result, length as usize);
        let result_vec: Vec<f32> = result_slice.to_vec();
        
        free_cpu_array(result);

        return result_vec;
    }
}

pub fn cuda_ptr_to_array_f16(ptr: *mut half::f16, shape: &[usize]) -> ArrayD<half::f16>
{
    unsafe {
        let mut length: u32 = 1;
        for i in 0..shape.len()
        {
            length *= shape[i] as u32;
        }

        let result: *mut half::f16 = to_cpu_f16(ptr, length);
        let result_slice: &mut [half::f16] = std::slice::from_raw_parts_mut(result, length as usize);
        let result_array: ArrayD<half::f16> 
            = ArrayD::from_shape_vec(
                IxDyn(shape), result_slice.to_vec()
            ).unwrap();
        
        //free_cpu_array(result);

        return result_array;
    }
}

pub fn array_from_pinned(host_ptr: *mut f32, flattened_shape: usize) -> ArrayD<f32>
{
    unsafe {
        //let result: *mut f32 = to_cpu(ptr, length);
        //copy_cuda_to_host(host_ptr, ptr, shape);
        let result_slice: &mut [f32] = std::slice::from_raw_parts_mut(host_ptr, flattened_shape);
        let result_array: ArrayD<f32> 
            = ArrayD::from_shape_vec(
                IxDyn(&[flattened_shape]), result_slice.to_vec()
            ).unwrap();
        
        return result_array;
    }
}


pub fn tensor_ptr_to_string(ptr: *mut ArrayD<f32>) -> String
{
    return format!("{:p}", ptr);
}

pub fn string_to_tensor_ptr(string: &String) -> *mut ArrayD<f32>
{
    let ptr_string: usize = usize::from_str_radix(&string[2..], 16).unwrap();
    return ptr_string as *mut ArrayD<f32>;
}

pub fn create_host_and_cuda_ptr(flattened_shape: usize) -> (String, String)
{
    let pinned_ptr: *mut f32 = new_cpu_pinned_array(flattened_shape as u32);
    let cuda_ptr: *mut f32 = cuda_ptr_from_pinned(pinned_ptr);

    let host_ptr_str: String = ptr_to_string(pinned_ptr);
    let cuda_ptr_str: String = ptr_to_string(cuda_ptr);

    return (host_ptr_str, cuda_ptr_str)
}

pub fn traverse_ptr_to_string(ptr: *mut TraversePtrs) -> String
{
    return format!("{:p}", ptr);
}

pub fn string_to_traverse_ptr(string: &String) -> *mut TraversePtrs
{
    let ptr_string: usize = usize::from_str_radix(&string[2..], 16).unwrap();
    return ptr_string as *mut TraversePtrs;
}

pub fn get_traverse_str_ptr(string: &String) -> (*mut f32, *mut f32, String)
{
    let traverse_ptr: *mut TraversePtrs = string_to_traverse_ptr(string);
    let ptr: *mut f32 = unsafe { (*traverse_ptr).ptr };
    let grad_ptr: *mut f32 = unsafe { (*traverse_ptr).grad_ptr };
    let backward_count_ptr: String = unsafe { (*traverse_ptr).backward_pass_count.clone() };

    return (ptr, grad_ptr, backward_count_ptr);
}

pub fn new_traverse_str_ptr(ptr: *mut f32, grad_ptr: *mut f32, backward_pass_count: String) -> String
{
    let traverse_ptrs_struct: Box<TraversePtrs> = Box::new(TraversePtrs {ptr, grad_ptr, backward_pass_count});

    let traverse_raw_ptr: *mut TraversePtrs = Box::into_raw(traverse_ptrs_struct);
    let str_ptr: String = traverse_ptr_to_string(traverse_raw_ptr);
    return str_ptr;
}

pub fn create_counting_ptr_str() -> String
{
    let raw_box_ptr: *mut f32 = Box::into_raw(Box::new(0.0));
    return ptr_to_string(raw_box_ptr);
}

pub fn increment_counter(backward_pass_count: &String)
{
    let backward_pass_count: *mut f32 = string_to_ptr(backward_pass_count);
    unsafe { *backward_pass_count += 1.0 }
}

pub fn set_zero_counter(backward_pass_count: &String)
{
    let backward_pass_count: *mut f32 = string_to_ptr(backward_pass_count);
    unsafe { *backward_pass_count = 0.0 }
}

pub fn counter_is_zero(backward_pass_count: &String) -> bool
{
    if !backward_pass_count.contains("none")
    {
        let backward_pass_count: *mut f32 = string_to_ptr(backward_pass_count);
        unsafe 
        { 
            if *backward_pass_count == 0.0 
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