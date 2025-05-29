use std::collections::VecDeque;

use ndarray::ArrayD;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataSeq
{
    pub input_seq: VecDeque<ArrayD<u16>>,
    pub output_seq: VecDeque<ArrayD<u16>>,
}

impl DataSeq
{
    pub fn new(input_seq: VecDeque<ArrayD<u16>>, output_seq: VecDeque<ArrayD<u16>>) -> Self
    {
        return Self
        {
            input_seq,
            output_seq,
        }
    }
}