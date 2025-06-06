use core::f32;
use std::ffi::c_void;
use std::fs::{read_to_string, File};
use std::io::{Read, Write};
use std::process::exit;
use std::collections::HashMap;

use crate::cuda_bridge::{copy_host_to_cuda, free_cuda_array, new_cuda_array, softmax_ce_loss};
use crate::layers::layer_cuda::LayerCuda;
use crate::pointer_ops::{create_host_and_cuda_ptr, set_zero_counter};
use crate::cuda_bridge::{copy_cuda_to_cuda, copy_host_to_host};
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
    pub io_ptrs_record: HashMap<String, *mut TraversePtrs>,

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
            io_ptrs_record: HashMap::new(),
            
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

        // store the output pointers
        if !self.io_ptrs_record.contains_key(&layer_id)
        {
            self.io_ptrs_record.insert(layer_id.clone(), new_traverse_ptr);
        }

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
        let mut cuda_layer_vec: Vec<(usize, &String, &Box<dyn LayerCuda>)> = Vec::new();
        for (layer_id, layer) in self.all_cuda_layers.iter()
        {
            let layer_id_splitted: Vec<&str> = layer_id.split("_").collect();
            let id_int: usize = layer_id_splitted[0].parse::<usize>().unwrap();
            cuda_layer_vec.push((id_int, layer_id, layer));
        }

        cuda_layer_vec.sort_by_key(|(id_int, _, _)| *id_int);
        
        for (i, (_, layer_id, layer)) in cuda_layer_vec.iter().enumerate()
        {
            println!("LAYER_ID: {:?} | LAYER_N: {}", layer_id, i);
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

    pub fn save(&mut self, filepath: &str)
    {
        let mut json_hashmap: HashMap<&str, HashMap<&str, Vec<f32>>> = HashMap::new();
        for (layer_id, layer) in &mut self.all_cuda_layers
        {
            let params: Option<HashMap<&str, Vec<f32>>> = layer.get_weights_hashmap();
            if params.is_some()
            {
                json_hashmap.insert(layer_id, params.unwrap());
            }
        }

        let json_string: String = serde_json::to_string(&json_hashmap).unwrap();

        let file: Result<File, std::io::Error> = File::create(filepath);

        if file.is_err()
        {
            println!("Error: {} ", file.unwrap_err());
            exit(1);
        }

        let mut file: File = file.unwrap();
        if let Err(e) = file.write_all(json_string.as_bytes())
        {
            print!("Error: {}", e);
            exit(1);
        }
        
    }

    pub fn load(&mut self, filepath: &str)
    {
        let file: Result<File, std::io::Error> = File::open(filepath);
        if file.is_err()
        {
            println!("Error: {} ", file.unwrap_err());
            exit(1);
        }

        let mut file: File = file.unwrap();
        let mut json_string: String = String::new();
        file.read_to_string(&mut json_string);

        let json_hashmap: HashMap<&str, HashMap<&str, Vec<f32>>> = 
            serde_json::from_str(&json_string).unwrap();

        for (layer_id, layer) in &mut self.all_cuda_layers
        {
            let param_hashmap: Option<&HashMap<&str, Vec<f32>>> = json_hashmap.get(layer_id.as_str());
            if param_hashmap.is_some()
            {
                layer.load_weights_from_hashmap(param_hashmap.unwrap());
            }
        }
    }

    pub fn delete(&mut self)
    {
        // each layer frees their "detached" pointers 
        //(pointers that aren't shared between layers)
        for (_, layer) in self.all_cuda_layers.iter()
        {
            layer.free_detached_ptrs();
        }

        // free the pointers in the ip_ptrs_record hash map
        unsafe 
        {
            for (_, traverse_ptr) in self.io_ptrs_record.iter()
            {
                free_cuda_array((**traverse_ptr).ptr as *mut c_void);
                free_cuda_array((**traverse_ptr).grad_ptr as *mut c_void);
                let _ = Box::from_raw((**traverse_ptr).backward_pass_count);

                // free the traverse ptr itself
                let _ = Box::from_raw(*traverse_ptr);
            }
        }
        
    }
}