use core::f32;
use std::ffi::c_void;
use std::fs::File;
use std::io::{Read, Write};
use std::process::exit;
use std::collections::HashMap;

use crate::cuda_bridge::{copy_host_to_cuda, free_cuda_array, free_pinned_array, new_cuda_array, softmax_ce_loss};
use crate::layers::layer_cuda::LayerCuda;
use crate::pointer_ops::{create_host_and_cuda_ptr, set_zero_counter};
use crate::cuda_bridge::{copy_cuda_to_cuda, copy_host_to_host};

// allows layers to pass their output data to subsequent layers during
// forward call
pub struct TraversePtrs
{
    pub ptr: *mut f32,
    pub grad_ptr: *mut f32,
    pub backward_pass_count: *mut usize // a pointer to the previous layer's counter
}

// the struct of the neural net itself
pub struct NeuralNet
{
    // contains all layers created and their names/ID
    pub all_cuda_layers: HashMap<String, Box<dyn LayerCuda>>, // name -> layer
    pub apply_dropout: bool,

    // stores the host and cuda pointers for input/output and
    // gradient arrays received from Python frontend, prevent
    // reallocation
    pub input_ptrs: HashMap<String, (*mut f32, *mut f32, u32)>,
    pub output_ptrs: HashMap<String, (*mut f32, *mut f32, u32)>,
    pub input_grad_ptrs: HashMap<String, (*mut f32, *mut f32, u32)>,
    pub output_grad_ptrs: HashMap<String, (*mut f32, *mut f32, u32)>,

    // input traversal pointers consist of cuda pointers from input_ptrs hash map
    pub input_traverse_ptrs: HashMap<String, *mut TraversePtrs>,

    // stores traversal pointers received from the last/output layer/s
    pub output_traverse_ptrs: HashMap<String, *mut TraversePtrs>,

    // stores arrays separate for outputs for calculating cross entropy loss
    // with softmax, prevent reallocation
    // (loss_vals_host, loss_vals_cuda, int_classes, output_grad)
    pub output_softmax_ce_data: HashMap<String, (*mut f32, *mut f32, *mut f32, *mut f32)>,

    // - keeps track of the order of layers during forward calls, contains the layer names
    // - reversed during backward call for correct gradient calculation/gradient accumulation
    pub backward_path: Vec<String>,
    pub backward_path_container: HashMap<String, bool>,

    // keeps track of "shared" pointers between layers, easier
    // to keep track of memory to free and prevent double frees
    pub io_ptrs_record: HashMap<String, *mut TraversePtrs>,

    // required for the very first input traversal pointers before
    // being passed to the input layer/s
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

    // add new layer and name
    pub fn add_cuda_layer(&mut self, name: String, layer: Box<dyn LayerCuda>)
    {
        self.all_cuda_layers.insert(name, layer);
    }

    // pass numpy array from Python frontend and memory copy to raw pointers
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

        // create new traveral pointer if not initialized
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

    // receive traversal pointer and return output array back to Python frontend
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

    // - uses CUDA to calculate the cross entropy loss given the target classes as integers
    //   for utilization of parallelization
    // - training data no longer need to store huge one hotted output arrays, saves
    //   memory usage
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

        return loss_vals_host;
    }

    // pass numpy array from Python frontend and memory copy to specified output traversal
    // pointer, which is also connected to the output pointer of output layer
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
    }

    // main method for forward propagation, also builds the pointer connections between layers
    pub fn forward(&mut self, layer_id: String, trav_in_ptr: *mut TraversePtrs, trav_weight_ptr: *mut TraversePtrs) -> *mut TraversePtrs
    {
        if !self.all_cuda_layers.contains_key(&layer_id)
        {
            println!("Error: Layer {:?} doesn't exist.", layer_id);
            exit(1);
        }
        let layer: &mut Box<dyn LayerCuda> = self.all_cuda_layers.get_mut(&layer_id).unwrap();
        let new_traverse_ptr: *mut TraversePtrs = layer.forward(trav_in_ptr, trav_weight_ptr, self.apply_dropout);

        // store the output pointers, needed for easier memory frees
        if !self.io_ptrs_record.contains_key(&layer_id)
        {
            self.io_ptrs_record.insert(layer_id.clone(), new_traverse_ptr);
        }

        // record forward call order
        if !self.backward_path_container.contains_key(&layer_id)
        {
            self.backward_path.push(layer_id.clone());
            self.backward_path_container.insert(layer_id.clone(), true);
        }

        return new_traverse_ptr;
    }

    // - reverse iterate the backward_path vector, perform backpropagation
    // - layer pointers have already been connected during the forward pass
    pub fn backward(&mut self)
    {
        for layer_name in self.backward_path.iter().rev()
        {
            let layer: &mut Box<dyn LayerCuda> = self.all_cuda_layers.get_mut(layer_name).unwrap();
            layer.backward(self.apply_dropout);
        }
    }

    // perform gradient descent on specified layer given ID
    pub fn update_params(&mut self, layer_id: &str, optimizer_type: i32, lr: f32, l2: f32, alpha: f32, beta: f32)
    {
        let layer: &mut Box<dyn LayerCuda> = self.all_cuda_layers.get_mut(layer_id).unwrap();
        layer.update_params(optimizer_type, lr, l2, alpha, beta);

        // zero the main backward pass count
        set_zero_counter(self.backward_pass_count);
    }

    // get details of all layers in neural net, provides the total number of parameters
    pub fn details(&self)
    {
        let mut count: usize = 0;
        let mut cuda_layer_vec: Vec<(usize, &String, &Box<dyn LayerCuda>)> = Vec::new();
        
        // - record all layer references in a vector that can be sorted
        // - ensures order is consistent when printing details of each layer
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

    // sets whether dropout should be enabled during forward and backward passes
    pub fn set_dropout(&mut self, use_dropout: bool)
    {
        self.apply_dropout = use_dropout;
    }

    // - write only the weight tensors for each layer if available to a hash map
    //   that is converted to JSON
    // - saves model state in JSON file, easy for model checkpointing
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

    // load specified JSON file and obtain data as a hash map
    // each layer updates their weight tensors if layer ID is available
    // in hash map
    pub fn load(&mut self, filepath: &str)
    {
        let file: Result<File, std::io::Error> = File::open(filepath);
        if file.is_err()
        {
            println!("Error: {} (Unable to located {})", file.unwrap_err(), filepath);
            exit(1);
        }

        let mut file: File = file.unwrap();
        let mut json_string: String = String::new();
        if let Err(e) = file.read_to_string(&mut json_string)
        {
            print!("Error: {}", e);
            exit(1);
        }

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

    // frees all the pointers allocated in all layers and hash map attributes
    // to prevent memory leaks, important since neural nets can be created and
    // destroyed multiple times in the Python frontend
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

        // free the memory in input/output hashmaps and corresponding memory for gradients
        for ((_, (host_ptr, _, _)), (_, (host_ptr_grad, _, _))) in 
            self.input_ptrs.iter().zip(self.input_grad_ptrs.iter())
        {
            free_pinned_array(*host_ptr as *mut c_void);
            free_pinned_array(*host_ptr_grad as *mut c_void);
        }

        for ((_, (host_ptr, _, _)), (_, (host_ptr_grad, _, _))) in 
            self.output_ptrs.iter().zip(self.output_grad_ptrs.iter())
        {
            free_pinned_array(*host_ptr as *mut c_void);
            free_pinned_array(*host_ptr_grad as *mut c_void);
        }

        // clear memory in cross entropy data hash map
        for (_, (loss_val_host, _, target_classes, _)) in self.output_softmax_ce_data.iter()
        {
            free_pinned_array(*loss_val_host as *mut c_void);
            free_cuda_array(*target_classes as *mut c_void);
        }

        
    }
}