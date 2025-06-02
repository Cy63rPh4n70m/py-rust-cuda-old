use core::f32;
use std::process::exit;
use std::collections::HashMap;
use std::os::raw::c_char;

use crate::cuda_bridge::{copy_host_to_cuda, new_cuda_array, softmax_ce_loss};
use crate::layers::layer_cuda::LayerCuda;
use crate::pointer_ops::{char_ptr_to_string, new_cuda_ptr_str, set_zero_counter};
use crate::{cuda_bridge::{copy_cuda_to_cuda, copy_host_to_host}, 
    pointer_ops::{create_counting_ptr_str, create_host_and_cuda_ptr, 
        new_traverse_str_ptr, ptr_to_string, string_to_ptr, string_to_traverse_ptr}};

// used in replacement of transferring arrays by value, more efficient
pub struct TraversePtrs
{
    pub ptr: *mut f32,
    pub grad_ptr: *mut f32,
    pub backward_pass_count: *mut usize // a pointer to the previous layer's counter
}

pub struct NeuralNet
{
    pub all_cuda_layers: HashMap<String, Box<dyn LayerCuda>>, // name -> layer
    pub apply_dropout: bool,

    pub input_ptrs: HashMap<String, (*mut f32, *mut f32, u32)>,
    pub output_ptrs: HashMap<String, (*mut f32, *mut f32, u32)>,
    pub input_grad_ptrs: HashMap<String, (*mut f32, *mut f32, u32)>,
    pub output_grad_ptrs: HashMap<String, (*mut f32, *mut f32, u32)>,
    pub input_traverse_ptrs: HashMap<String, *mut TraversePtrs>,
    pub output_traverse_ptrs: HashMap<String, *mut TraversePtrs>,
    pub output_softmax_ce_data: HashMap<String, (*mut f32, *mut f32, *mut f32, *mut f32)>, // (loss_vals_host, loss_vals_cuda, int_classes, output_grad)

    // order of cuda layer names to call for forward or backward
    // booleans determine whether to initialize output/input grad to zero
    // required when inputs/gradients are accumulated due to architectural design
    pub backward_path: Vec<String>,
    pub backward_path_container: HashMap<String, bool>,

    pub backward_pass_count: *mut usize,
}
impl NeuralNet
{
    pub fn new() -> Self
    {
        
        return Self
        {
            all_cuda_layers: HashMap::new(),
            apply_dropout: true,

            input_ptrs: HashMap::new(),
            input_grad_ptrs: HashMap::new(),
            output_ptrs: HashMap::new(),
            output_grad_ptrs: HashMap::new(),
            input_traverse_ptrs: HashMap::new(),
            output_traverse_ptrs: HashMap::new(),
            output_softmax_ce_data: HashMap::new(),

            backward_path: Vec::new(),
            backward_path_container: HashMap::new(),
            
            backward_pass_count: Box::into_raw(Box::new(0_usize)),
        };
    }

    pub fn add_cuda_layer(&mut self, name: String, layer: Box<dyn LayerCuda>)
    {
        self.all_cuda_layers.insert(name, layer);
    }

    pub fn pass_to_input(&mut self, name: String, array_ptr: *mut f32, array_len: usize) -> *mut TraversePtrs
    {
        if !self.input_ptrs.contains_key(&name)
        {
            self.input_ptrs.insert(
                name.clone(), 
                (std::ptr::null_mut(), std::ptr::null_mut(), array_len as u32)
            );
            self.input_grad_ptrs.insert(
                name.clone(), 
                (std::ptr::null_mut(), std::ptr::null_mut(), array_len as u32)
            );
        }

        // host pointer is pinned, gpu pointer can access directly
        let ptrs: &mut (*mut f32, *mut f32, u32) = self.input_ptrs.get_mut(&name).unwrap();
        let grad_ptrs: &mut (*mut f32, *mut f32, u32) = self.input_grad_ptrs.get_mut(&name).unwrap();

        if ptrs.0.is_null()
        {
            let (host_ptr_str, cuda_ptr_str) = create_host_and_cuda_ptr(array_len);
            let (host_ptr_str_grad, cuda_ptr_str_grad) = create_host_and_cuda_ptr(array_len);
            
            ptrs.0 = host_ptr_str; 
            ptrs.1 = cuda_ptr_str;
            grad_ptrs.0 = host_ptr_str_grad; 
            grad_ptrs.1 = cuda_ptr_str_grad;

            let traverse_ptr: *mut TraversePtrs = Box::into_raw(
                Box::new(
                    TraversePtrs 
                    {
                        ptr: ptrs.1, 
                        grad_ptr: grad_ptrs.1, 
                        backward_pass_count: self.backward_pass_count
                    }
                )
            );

            self.input_traverse_ptrs.insert(name.clone(), traverse_ptr);
        }

        copy_host_to_host(ptrs.0, array_ptr, &[array_len]);
        let traverse_ptr_str: &*mut TraversePtrs = self.input_traverse_ptrs.get(&name).unwrap();
        return traverse_ptr_str.clone();
    }

    pub fn pass_to_output(&mut self, name: String, traverse_ptr: *mut TraversePtrs, array_len: usize) -> *mut f32
    {

        if !self.output_ptrs.contains_key(&name)
        {
            self.output_ptrs.insert(
                name.clone(), 
                (std::ptr::null_mut(), std::ptr::null_mut(), array_len as u32)
            );
            self.output_grad_ptrs.insert(
                name.clone(), 
                (std::ptr::null_mut(), std::ptr::null_mut(), array_len as u32)
            );
        }

        // host pointer is pinned, gpu pointer can access directly
        let ptrs: &mut (*mut f32, *mut f32, u32) = self.output_ptrs.get_mut(&name).unwrap();
        let grad_ptrs: &mut (*mut f32, *mut f32, u32) = self.output_grad_ptrs.get_mut(&name).unwrap();

        //let traverse_ptr_str: String = char_ptr_to_string(traverse_ptr);
        //let traverse_struct: *mut TraversePtrs = string_to_traverse_ptr(&traverse_ptr_str);

        // set the output/output_grad pointers
        if ptrs.0.is_null()
        {
            let (host_ptr_str, cuda_ptr_str) = create_host_and_cuda_ptr(array_len);
            let (host_ptr_str_grad, cuda_ptr_str_grad) = create_host_and_cuda_ptr(array_len);
            ptrs.0 = host_ptr_str; 
            ptrs.1 = cuda_ptr_str;
            grad_ptrs.0 = host_ptr_str_grad;
            grad_ptrs.1 = cuda_ptr_str_grad;

            self.output_traverse_ptrs.insert(name.clone(), traverse_ptr);
        }

        // copy to the output pointer
        copy_cuda_to_cuda(
            ptrs.1, 
            unsafe { (*traverse_ptr).ptr }, 
            &[array_len]
        );
        
        return ptrs.0;
    }

    pub fn ce_loss_fn(&mut self, output_name: String, target_classes: *mut f32, batch: usize, rows: usize, cols: usize) -> *mut f32
    {
        let traverse_ptr: &*mut TraversePtrs = self.output_traverse_ptrs.get(&output_name).unwrap();

        if !self.output_softmax_ce_data.contains_key(&output_name)
        {
            let (loss_vals_host, loss_vals_cuda) = 
                create_host_and_cuda_ptr(batch * rows);
            let classes:*mut f32  = new_cuda_array((batch * rows * 1) as u32);
            unsafe
            {
                self.output_softmax_ce_data.insert(
                    output_name.clone(), 
                    (loss_vals_host, loss_vals_cuda, classes, (**traverse_ptr).grad_ptr)
                );
            }
        }

        let softmax_ce_data: &(*mut f32, *mut f32, *mut f32, *mut f32) = 
            self.output_softmax_ce_data.get(&output_name).unwrap();
        let pred: *mut f32 = unsafe {(**traverse_ptr).ptr};
        let loss_vals_host: *mut f32 = softmax_ce_data.0;
        let loss_vals_cuda: *mut f32 = softmax_ce_data.1;
        let classes: *mut f32 = softmax_ce_data.2;
        let grad_ptr: *mut f32 = softmax_ce_data.3;

        copy_host_to_cuda(classes, target_classes, (batch * rows) as u32);
        
        softmax_ce_loss(
            pred, classes, loss_vals_cuda, grad_ptr, batch as i32, rows as i32, cols as i32
        );

        //println!("{:?}", cuda_ptr_to_array(loss_vals_host, &[batch, rows, 1]));
        //unsafe 
        //{
        //    println!("{:?}", cuda_ptr_to_array((*traverse_struct).ptr, &[batch, rows, cols]));
        //    println!("{:?}", cuda_ptr_to_array((*traverse_struct).grad_ptr, &[batch, rows, cols]));
        //}

        return loss_vals_host;
    }

    pub fn pass_to_output_grad(&mut self, name: String, array_ptr: *mut f32, array_len: usize)
    {
        // host pointer is pinned, gpu pointer can access directly
        let grad_ptrs: &(*mut f32, *mut f32, u32) = self.output_grad_ptrs.get(&name).unwrap();
        copy_host_to_host(grad_ptrs.0, array_ptr, &[array_len]);

        // copy cuda data to traverse pointer stored
        let traverse_ptr: &*mut TraversePtrs = self.output_traverse_ptrs.get(&name).unwrap();
        copy_cuda_to_cuda(
            unsafe { (**traverse_ptr).grad_ptr }, 
            grad_ptrs.1, 
            &[array_len]
        );
        //unsafe {
        //    println!("{:?}", cuda_ptr_to_array((*traverse_struct).ptr, &[array_len]));
        //    println!("{:?}", cuda_ptr_to_array((*traverse_struct).grad_ptr, &[array_len]));
        //}
    }

    pub fn forward(&mut self, layer_id: String, trav_in_ptr: *mut TraversePtrs, trav_weight_ptr: *mut TraversePtrs) -> *mut TraversePtrs
    {
        if !self.all_cuda_layers.contains_key(&layer_id)
        {
            println!("Error: Layer {:?} doesn't exist.", layer_id);
            exit(1);
        }
        let layer: &mut Box<dyn LayerCuda> = self.all_cuda_layers.get_mut(&layer_id).unwrap();
        let new_traverse_ptr: *mut TraversePtrs = layer.forward(trav_in_ptr, trav_weight_ptr, self.apply_dropout);

        if !self.backward_path_container.contains_key(&layer_id)
        {
            self.backward_path.push(layer_id.clone());
            self.backward_path_container.insert(layer_id.clone(), true);
        }

        return new_traverse_ptr;
    }

    pub fn backward(&mut self)
    {
        // backpropagate through layers, reverse of the layer path
        for layer_name in self.backward_path.iter().rev()
        {
            let layer: &mut Box<dyn LayerCuda> = self.all_cuda_layers.get_mut(layer_name).unwrap();
            //println!("started {:?}", layer_name);
            layer.backward(self.apply_dropout);
            //println!("completed {:?}", layer_name);
        }
    }

    pub fn update_params(&mut self, layer_id: &str, optimizer_type: i32, lr: f32, l2: f32, alpha: f32, beta: f32)
    {
        let layer: &mut Box<dyn LayerCuda> = self.all_cuda_layers.get_mut(layer_id).unwrap();
        layer.update_params(optimizer_type, lr, l2, alpha, beta);

        // zero the main backward pass count, 
        set_zero_counter(self.backward_pass_count);
    }

    pub fn details(&self)
    {
        let mut count: usize = 0;
        for (i, layer_name) in self.backward_path.iter().enumerate()
        {
            let layer: &Box<dyn LayerCuda> = self.all_cuda_layers.get(layer_name).unwrap();
            println!("LAYER_ID: {:?} | LAYER_N: {}", layer_name, i);
            layer.details();
            println!("TOTAL_LAYER_PARAM_COUNT: {:?}", layer.get_param_count());
            println!("\x1b[48;5;8m==================================\x1b[0m");

            count += layer.get_param_count();
        }

        println!("TOTAL_MODEL_PARAM_COUNT: {:?}", count);
        println!("MODEL_BYTE_SIZE: {:?}", count * 4);
    }

    pub fn set_dropout(&mut self, use_dropout: bool)
    {
        self.apply_dropout = use_dropout;
    }

    // store the parameters from the cuda pointers into ndarrays
    // only works on the cuda arrays, which use pointers to store inputs/weights
    pub fn store_cuda_params(&mut self)
    {
        for (cuda_layer_name, _) in &mut self.backward_path_container
        {
            let cuda_layer: &mut Box<dyn LayerCuda> = self.all_cuda_layers.get_mut(cuda_layer_name).unwrap();
            cuda_layer.move_ptrs_to_arrays();
        }
    }

    pub fn config_for_save(&mut self, checkpoint: bool)
    {   
        if checkpoint
        {
            self.input_ptrs_copy = self.input_ptrs.clone();
            self.input_grad_ptrs_copy = self.input_grad_ptrs.clone();
            self.output_ptrs_copy = self.output_ptrs.clone();
            self.output_grad_ptrs_copy = self.output_grad_ptrs.clone();
            self.input_traverse_ptrs_copy = self.input_traverse_ptrs.clone();
            self.output_traverse_ptrs_copy = self.output_traverse_ptrs.clone();
            self.output_softmax_ce_data_copy = self.output_softmax_ce_data.clone();
        }

        self.backward_pass_count_set = false;

        self.input_ptrs.clear();
        self.input_grad_ptrs.clear();
        self.output_ptrs.clear();
        self.output_grad_ptrs.clear();
        self.input_traverse_ptrs.clear();
        self.output_traverse_ptrs.clear();
        self.output_softmax_ce_data.clear();

        //self.input_ptrs_allocated = false;
        //self.grad_ptrs_allocated = false;
    }

    pub fn restore_for_checkpoint(&mut self)
    {
        for (cuda_layer_name, _) in &mut self.backward_path_container
        {
            let cuda_layer: &mut CudaLayer = self.all_cuda_layers.get_mut(cuda_layer_name).unwrap();
            cuda_layer.set_ptrs_allocated();
        }

        self.backward_pass_count_set = true;

        self.input_ptrs = self.input_ptrs_copy.clone();
        self.input_grad_ptrs = self.input_grad_ptrs_copy.clone();
        self.output_ptrs = self.output_ptrs_copy.clone();
        self.output_grad_ptrs = self.output_grad_ptrs_copy.clone();
        self.input_traverse_ptrs = self.input_traverse_ptrs_copy.clone();
        self.output_traverse_ptrs = self.output_traverse_ptrs_copy.clone();
        self.output_softmax_ce_data = self.output_softmax_ce_data_copy.clone();
    }
}