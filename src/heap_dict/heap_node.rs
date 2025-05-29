use std::fmt::Debug;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HeapNode<K, V>
{
    pub key: K,
    pub value: V
}

impl<K, V> HeapNode<K, V>
{
    pub fn new(key: K, value: V) -> Self
    {
        return Self
        {
            key,
            value
        }
    }
}