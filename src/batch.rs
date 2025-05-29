use ndarray::{Array1, Array3};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Batch3D
{
    pub input_data_vec: Vec<Array3<f32>>,
    pub output_data_vec: Vec<Array1<f32>>
}