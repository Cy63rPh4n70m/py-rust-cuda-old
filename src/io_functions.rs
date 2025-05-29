use std::{fs::{File, OpenOptions}, io::Write};
use memmap2::MmapMut;

use crate::{neuralnet::NeuralNet, storage::storage_seq::DataSeq};

pub fn save_model(path: &str, neuralnet: &mut NeuralNet, checkpoint: bool)
{
    let mut file: File = File::create(path).unwrap();
    
    // save any cuda pointers to ndarray
    neuralnet.store_cuda_params();
    neuralnet.config_for_save(checkpoint);

    let serialized_nn: String = serde_json::to_string(neuralnet).unwrap();

    //println!("Model path: {:?}", path);
    //println!("Model size: {:?} Bytes", serialized_nn.len());
    let _ = file.write_all(serialized_nn.as_bytes());
    let _ = file.flush();

    if checkpoint
    {
        // restore all io pointers to allocated
        neuralnet.restore_for_checkpoint();
    }
}

pub fn load_model(path: &str) -> NeuralNet
{
    let file: File = File::open(path).unwrap();
    let nn_data: String = std::io::read_to_string(file).unwrap();
    //println!("Model path: {:?}", path);
    //println!("Model size: {:?} MB", nn_data.len() / 1_000_000);
    let neuralnet: NeuralNet = serde_json::from_str(&nn_data).unwrap();

    return neuralnet;
}

pub fn write_seq_to_mem(seq: DataSeq, dir_path: String, id: String) -> MmapMut
{
    let file: File = OpenOptions::new()
        .create(true)
        .write(true)
        .read(true)
        .open(dir_path + "/" + &id).unwrap();

    let serialized_seq: String = serde_json::to_string(&seq).unwrap();
    file.set_len(serialized_seq.len() as u64).unwrap();

    let mut memorymap: MmapMut = unsafe { MmapMut::map_mut(&file).unwrap() };
    memorymap[..].copy_from_slice(serialized_seq.as_bytes());
    memorymap.flush().unwrap();

    return memorymap;
}

pub fn read_seq_from_mem(memory_map: &MmapMut, byte_range: usize) -> DataSeq
{   
    let mem_data: &[u8] = memory_map.get(0..byte_range).unwrap();
    let data_seq: DataSeq = bincode::deserialize(mem_data).unwrap();
    //let serialized_data: String = String::from_utf8_lossy(mem_data).to_string();
    //let data_seq: DataSeq = serde_json::from_str(&serialized_data).unwrap();
    //println!("{:?}\n{:?}\n{:?}", mem_data, data_seq.input_seq, data_seq.output_seq);
    return data_seq;
}