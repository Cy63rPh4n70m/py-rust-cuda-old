
use std::collections::VecDeque;

use ndarray::ArrayD;
use rand::Rng;

use crate::heap_dict::{heap_dict::HeapDict, heap_node::HeapNode}; 

/*
#[derive(Debug)]
pub struct MemoryMap
{
    mmap: Option<MmapMut>,
    byte_len: usize,
    mmap_file_path: String
}
impl MemoryMap
{
    pub fn new(initial_bytes: Vec<u8>) -> Self
    {
        let random_path: String = "memory".to_string() + "/" + gen_random_name(15).as_str();
        let file: File = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(random_path.clone()).unwrap();
        
        file.set_len(100000 as u64).unwrap();
        let mut mmap: MmapMut = unsafe { 
            MmapMut::map_mut(&file).unwrap() 
        };

        let byte_len: usize = initial_bytes.len();
        mmap[0..byte_len].copy_from_slice(&initial_bytes);
        mmap.flush().unwrap();

        return Self
        {
            mmap: Some(mmap),
            byte_len,
            mmap_file_path: random_path
        };
    }

    pub fn write_bytes(&mut self, bytes: Vec<u8>)
    {
        let new_byte_len: usize = bytes.len();
        if new_byte_len > self.byte_len
        {
            self.clear_mmap();
            // remap file with more bytes
            let file: File = OpenOptions::new()
                .create(true)
                .write(true)
                .read(true)
                .open(self.mmap_file_path.clone()).unwrap();
        
            file.set_len(new_byte_len as u64).unwrap();

            // reassign mmap
            self.mmap = unsafe {
                Some(MmapMut::map_mut(&file).unwrap())
            };
        }

        self.byte_len = new_byte_len;
        let mmap: &mut MmapMut = self.mmap.as_mut().unwrap();
        mmap[0..self.byte_len].copy_from_slice(&bytes);
        mmap.flush().unwrap();

    }

    pub fn clear_mmap(&mut self)
    {
        self.mmap = None;
    }

    pub fn read_bytes(&self) -> &[u8]
    {
        return &self.mmap.as_ref().unwrap()[0..self.byte_len];
    }
}
*/

pub struct ReplayBuf
{
    // constantly appending/popping inputs/outputs
    online_count: u128,
    online_state_queue: VecDeque<(ArrayD<f32>, ArrayD<f32>)>,
    online_reward_queue: VecDeque<f32>,
    online_choice_queue: VecDeque<f32>, // 0 = from pred, 1 = random action
    //online_ae_loss_seq: VecDeque<f32>,
    online_reward_sum: f32,
    online_choice_sum: f32, // plus one if action was not desired action

    //seq_maps: Vec<MmapMut>,
    //seqs: Vec<Vec<u8>>,
    //ratings: Vec<f32>,
    seq_heap_dict: HeapDict<f32, (ArrayD<f32>, ArrayD<f32>)>,
    //seq_in_use: DataSeq,
    //seq_byte_lens: Vec<usize>,
    // to be used for random selection on heap array
    // weights based on heap level
    //cumul_level_weights: Vec<f32>,
    //idx_level_cat: Vec<Vec<usize>>,

    max_buffer_len: usize,
    max_online_queue_len: usize,
    //current_idx: i32,
    //ascend: bool,
}

impl ReplayBuf
{
    pub fn new(max_buffer_len: usize, max_online_queue_len: usize) -> Self
    {
        let seq_storage: ReplayBuf = Self
        {
            online_count: 0,
            online_state_queue: VecDeque::new(),
            online_reward_queue: VecDeque::new(),
            online_choice_queue: VecDeque::new(),
            //online_ae_loss_seq: VecDeque::new(),
            online_reward_sum: 0.0,
            online_choice_sum: 0.0,
            
            //seq_maps: Vec::new(),
            //seqs: Vec::new(),
            //ratings: Vec::new(),
            //seq_in_use: DataSeq::new(VecDeque::new(), VecDeque::new()),
            seq_heap_dict: HeapDict::new(),
            //seq_byte_lens: Vec::new(),

            max_buffer_len, max_online_queue_len,
            //ascend: true,
            //current_idx: 0,
        };

        return seq_storage;
    }

    pub fn add_state(&mut self, input: ArrayD<f32>, output: ArrayD<f32>, reward: f32, choice_type: f32)
    {   
        // add to online sequence
        self.online_state_queue.push_back((input, output));
        self.online_reward_queue.push_back(reward);
        self.online_choice_queue.push_back(choice_type);
        self.online_reward_sum += reward;
        self.online_choice_sum += choice_type;

        // remove if reach over limit
        if self.online_state_queue.len() > self.max_online_queue_len
        {
            self.online_state_queue.pop_front();

            self.online_reward_sum -= self.online_reward_queue.pop_front().unwrap();
            self.online_choice_sum -= self.online_choice_queue.pop_front().unwrap();
        }

        let rating: f32 = self.online_reward_sum + self.online_reward_queue.len() as f32;
            
        // record copy of current online sequences and save to disk
        //let random_name: String = gen_random_name(20);
        //let new_data_seq: DataSeq = DataSeq::new(
        //    self.online_input_seq.clone(), 
        //    self.online_output_seq.clone()
        //);

        // add random state from the online state queue to the replay heap
        let random_idx: usize = rand::thread_rng().gen_range(0..self.online_state_queue.len());
        let new_state: (ArrayD<f32>, ArrayD<f32>) = self.online_state_queue[random_idx].clone();

        if self.seq_heap_dict.len() >= self.max_buffer_len
        {
            let first_node: &mut HeapNode<f32, (ArrayD<f32>, ArrayD<f32>)> = 
                self.seq_heap_dict.mut_ref_at(0);
                
            let lowest_rating: f32 = first_node.key;

            if rating >= lowest_rating
            {
                //let first_ref: &mut HeapNode<f32, Vec<u8>> = self.seq_heap_dict.mut_ref_at(0);
                //first_node.key = rating;
                //first_node.data.write_bytes(serialized_seq);
                //self.seq_heap_dict.trickle_down();
                //let serialized_seq: Vec<u8> = bincode::serialize(&new_data_seq).unwrap().to_vec();
                self.seq_heap_dict.replace_first(rating, new_state);
            }
        }
        else
        {
            //let serialized_seq: Vec<u8> = bincode::serialize(&new_data_seq).unwrap().to_vec();
            //let memory_map: MemoryMap = MemoryMap::new(serialized_seq);
            self.seq_heap_dict.add(rating, new_state);
        }
        /**/
    }

    /*
    pub fn mov_seq_idx(&mut self) -> usize
    {
        let current_idx: usize = self.current_idx as usize;
        
        if self.current_idx == 0
        {
            // allows new entries to be accepted while allowing
            // current data to still be trained on
            for node in &mut self.seq_heap_dict.heap_vec
            {
                //node.key = f32::NEG_INFINITY;
            }
        }

        if self.current_idx >= self.seq_heap_dict.len() as i32 - 1
        {
            self.current_idx = self.seq_heap_dict.len() as i32 - 1;
            self.ascend = false;
        }
        else if self.current_idx <= 0
        {
            self.current_idx = 0;
            self.ascend = true;
        }

        if self.seq_heap_dict.len() > 1
        {
            if self.ascend
            {
                self.current_idx += 1;
            }
            else
            {
                self.current_idx -= 1;
            }
        }
        //println!("{:?}, {:?}", self.ascend, self.current_idx);

        self.current_idx = rand::thread_rng().gen_range(0..self.seq_heap_dict.len()) as i32;

        return current_idx;
    }
    */

    /*
    pub fn update_saved_seq(&mut self)
    {
        //let serialised_seq: &[u8] = 
        //    self.seq_heap_dict.heap_vec[self.current_idx as usize].data.read_bytes();
        let serialised_seq: Vec<u8> = self.seq_heap_dict.heap_vec[self.current_idx as usize].value.clone();

        let seq: DataSeq = bincode::deserialize(
            &serialised_seq
        ).unwrap();

        self.seq_in_use = seq;
    }
    */

    /*
    pub fn get_saved_seq_len(&self) -> usize
    {
        return self.seq_in_use.input_seq.len();
    }
    */

    pub fn get_state_input_at(&self, idx: usize) -> ArrayD<f32>
    {
        let input_array: ArrayD<f32> = 
            self.seq_heap_dict.heap_vec[idx].value.0.clone();
        
        return input_array;
    }

    pub fn get_state_output_at(&self, idx: usize) -> ArrayD<f32>
    {
        let output_array: ArrayD<f32> = 
            self.seq_heap_dict.heap_vec[idx].value.1.clone();
        
        return output_array;
    }

    pub fn reset_online_queue(&mut self)
    {  
        self.online_state_queue.clear();
        self.online_reward_queue.clear();
        self.online_choice_queue.clear();
        self.online_count = 0;
        self.online_reward_sum = 0.0;
        self.online_choice_sum = 0.0;
    }

    pub fn get_ave_rating(&mut self) -> f32
    {
        //let mut rating_sorted: Vec<f32> = self.ratings.clone();
        //rating_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        //let q1_idx: usize = (rating_sorted.len() + 1) * (1 / 4);

        let mut rating: f32 = 0.0;
        for i in 0..self.seq_heap_dict.len()
        {
            rating += self.seq_heap_dict.heap_vec[i].key;
        }

        //return rating_sorted[q1_idx];//rating / (self.ratings.len() as f32);
        return rating / (self.seq_heap_dict.len() as f32);
    }

    pub fn get_count(&self) -> usize
    {
        return self.seq_heap_dict.len();
    }

}