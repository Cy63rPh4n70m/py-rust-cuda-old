use core::f32;
use std::process::exit;
use std::collections::HashMap;
use std::os::raw::c_char;

use crate::cuda_bridge::{copy_host_to_cuda, softmax_ce_loss};
use crate::layers::layer_cuda::LayerCuda;
use crate::pointer_ops::{char_ptr_to_string, new_cuda_ptr_str};
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
        let traverse_ptr_str: &String = self.output_traverse_ptrs.get(&output_name).unwrap();
        let traverse_struct: *mut TraversePtrs = string_to_traverse_ptr(&traverse_ptr_str);

        if !self.output_softmax_ce_data.contains_key(&output_name)
        {
            let (loss_vals_host, loss_vals_cuda) = 
                create_host_and_cuda_ptr(batch * rows);
            let classes: String = new_cuda_ptr_str(&[batch, rows, 1]);
            unsafe
            {
                self.output_softmax_ce_data.insert(
                    output_name.clone(), 
                    (loss_vals_host, loss_vals_cuda, classes, ptr_to_string((*traverse_struct).grad_ptr)
                ));
            }
        }

        let softmax_ce_data: &(String, String, String, String) = 
            self.output_softmax_ce_data.get(&output_name).unwrap();
        let pred: *mut f32 = unsafe {(*traverse_struct).ptr};
        let loss_vals_host: *mut f32 = string_to_ptr(&softmax_ce_data.0);
        let loss_vals_cuda: *mut f32 = string_to_ptr(&softmax_ce_data.1);
        let classes: *mut f32 = string_to_ptr(&softmax_ce_data.2);
        let grad_ptr: *mut f32 = string_to_ptr(&softmax_ce_data.3);

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
        let grad_ptrs: &(String, String, u32) = self.output_grad_ptrs.get(&name).unwrap();
        copy_host_to_host(string_to_ptr(&grad_ptrs.0), array_ptr, &[array_len]);

        // copy cuda data to traverse pointer stored
        let traverse_ptr_str: &String = self.output_traverse_ptrs.get(&name).unwrap();
        let traverse_struct: *mut TraversePtrs = string_to_traverse_ptr(&traverse_ptr_str);
        copy_cuda_to_cuda(
            unsafe { (*traverse_struct).grad_ptr }, 
            string_to_ptr(&grad_ptrs.1), 
            &[array_len]
        );
        //unsafe {
        //    println!("{:?}", cuda_ptr_to_array((*traverse_struct).ptr, &[array_len]));
        //    println!("{:?}", cuda_ptr_to_array((*traverse_struct).grad_ptr, &[array_len]));
        //}
    }

    // assumes all layers use the same input pointer names
    // connects input pointer/s with the first input layer/s
    /*
    pub fn join_input(&mut self, input_ptr_name: String, layer_name: String)
    {
        let ptrs: &(String, String, u32) = self.input_ptrs.get(&input_ptr_name).unwrap();
        let ptrs_grad: &(String, String, u32) = self.input_grad_ptrs.get(&input_ptr_name).unwrap();
        let layer: &mut CudaLayer= self.all_cuda_layers.get_mut(&layer_name).unwrap();

        layer.set_ptr("input".to_string(), (*ptrs.1).to_string());
        layer.set_ptr("input_grad".to_string(), (*ptrs_grad.1).to_string());
    }
    */

    // assumes all layers use the same output pointer names
    // connects output/return pointer/s with the final output layer/s
    /*
    pub fn join_output(&mut self, output_ptr_name: String, layer_name: String)
    {
        let ptrs: &(String, String, u32) = self.output_ptrs.get(&output_ptr_name).unwrap();
        let ptrs_grad: &(String, String, u32) = self.output_grad_ptrs.get(&output_ptr_name).unwrap();
        let layer: &mut CudaLayer= self.all_cuda_layers.get_mut(&layer_name).unwrap();

        layer.set_ptr("output".to_string(), (*ptrs.1).to_string());
        layer.set_ptr("output_grad".to_string(), (*ptrs_grad.1).to_string());
    }
    */

    /*
    pub fn join_layers(
        &mut self, src_layer_name: String, dst_layer_name: String, 
        ptr_name_in_src: String, ptr_name_in_dst: String,
        grad_ptr_name_in_src: String, grad_ptr_name_in_dst: String,
        flattened_shape: u32
    )
    {
        // get the pointer from source and destination to check if either already exists
        let src_layer: &CudaLayer = self.all_cuda_layers.get(&src_layer_name).unwrap();
        let src_ptr_str: String = src_layer.get_ptr(ptr_name_in_src.clone());
        let src_ptr_str_grad: String = src_layer.get_ptr(grad_ptr_name_in_src.clone());

        let dst_layer: &CudaLayer = self.all_cuda_layers.get(&dst_layer_name).unwrap();
        let dst_ptr_str: String = dst_layer.get_ptr(ptr_name_in_dst.clone());
        let dst_ptr_str_grad: String = dst_layer.get_ptr(grad_ptr_name_in_dst.clone());
        
        let mut ptr_str: String = String::new();
        let mut ptr_str_grad: String = String::new();

        // create new pointers if not existing
        // else use the pointer from layer that already exists
        // for data branching capabilities
        if src_ptr_str == "none" && dst_ptr_str == "none"
        {
            ptr_str = ptr_to_string(new_cuda_array(flattened_shape));
            ptr_str_grad = ptr_to_string(new_cuda_array(flattened_shape));
        }
        else if src_ptr_str == "none" && dst_ptr_str != "none"
        {
            ptr_str = dst_ptr_str;
            ptr_str_grad = dst_ptr_str_grad;
        }
        else if src_ptr_str != "none" && dst_ptr_str == "none"
        {
            ptr_str = src_ptr_str;
            ptr_str_grad = src_ptr_str_grad;
        }

        let src_layer: &mut CudaLayer = self.all_cuda_layers.get_mut(&src_layer_name).unwrap();
        src_layer.set_ptr(ptr_name_in_src, ptr_str.clone());
        src_layer.set_ptr(grad_ptr_name_in_src, ptr_str_grad.clone());

        let dst_layer: &mut CudaLayer = self.all_cuda_layers.get_mut(&dst_layer_name).unwrap();
        dst_layer.set_ptr(ptr_name_in_dst, ptr_str.clone());
        dst_layer.set_ptr(grad_ptr_name_in_dst, ptr_str_grad.clone());
    }
    */

    pub fn forward(&mut self, layer_id: String, str_ptr_in: String, str_ptr_weight: String) -> String
    {
        if !self.all_cuda_layers.contains_key(&layer_id)
        {
            println!("Error: Layer {:?} doesn't exist.", layer_id);
            exit(1);
        }
        let layer: &mut CudaLayer = self.all_cuda_layers.get_mut(&layer_id).unwrap();
        let new_traverse_ptr: String = layer.forward(str_ptr_in, str_ptr_weight, self.apply_dropout);

        if !self.backward_path_container.contains_key(&layer_id)
        {
            self.backward_path.push(layer_id.clone());
            self.backward_path_container.insert(layer_id.clone(), true);
        }

        return new_traverse_ptr;
    }

    //pub fn get_output(&mut self, output_name: &str) -> *mut f32
    //{
    //    // host pointer is pinned, gpu pointer can access directly
    //    let ptrs: &(String, String, u32) = self.output_ptrs.get(output_name).unwrap();
    //    //let output_as_array: ArrayD<f32> = array_from_pinned(string_to_ptr(&ptrs.0), ptrs.2 as usize);
    //    return string_to_ptr(&ptrs.0);
    //}

    pub fn backward(&mut self)
    {
        // backpropagate through layers, reverse of the layer path
        for layer_name in self.backward_path.iter().rev()
        {
            let layer: &mut CudaLayer = self.all_cuda_layers.get_mut(layer_name).unwrap();
            //println!("started {:?}", layer_name);
            layer.backward(self.apply_dropout);
            //println!("completed {:?}", layer_name);
        }
            //increment_counter(&self.backward_pass_count);
    }

    //pub fn obtain_flattened_output(&self) -> usize
    //{
    //    return self.last_conv_shape.0 * self.last_conv_shape.1 * self.last_conv_shape.2;
    //}

    pub fn update_params(&mut self, layer_id: &str, optimizer_type: i32, lr: f32, l2: f32, alpha: f32, beta: f32)
    {
        let layer: &mut CudaLayer = self.all_cuda_layers.get_mut(layer_id).unwrap();
        layer.update_params(optimizer_type, lr, l2, alpha, beta);

        // zero the main backward pass count, 
        //set_zero_counter(&self.backward_pass_count);
    }
    
    pub fn update_loss_queue(&mut self, loss: f32, _maxlen: usize)
    {
        if self.prev_loss > 0.0
        {
            if loss > self.prev_loss
            {
                self.global_lr *= 0.25;
                println!("--- Reduced global loss to {:?} ---", self.global_lr);
                for (_, layer) in &mut self.all_cuda_layers
                {
                    //layer.set_lr(self.global_lr);
                }
            }
        }

        self.prev_loss = loss;
    }

    pub fn details(&self)
    {
        let mut count: usize = 0;
        for (i, layer_name) in self.backward_path.iter().enumerate()
        {
            let layer: &CudaLayer = self.all_cuda_layers.get(layer_name).unwrap();
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
            let cuda_layer: &mut CudaLayer = self.all_cuda_layers.get_mut(cuda_layer_name).unwrap();
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