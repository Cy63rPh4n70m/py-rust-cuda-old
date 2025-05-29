use std::fmt::Debug;

use rand::Rng;
use serde::{Deserialize, Serialize};

use super::heap_node::HeapNode;

// implement min heap
#[derive(Serialize, Deserialize)]
pub struct HeapDict<K, V>
{
    pub heap_vec: Vec<HeapNode<K, V>>,
}
impl<K, V> HeapDict<K, V>
where
    K: Clone + Copy + Debug + PartialOrd + Serialize + for<'a> Deserialize<'a>,
    V: Debug + Serialize + for<'a> Deserialize<'a>
{
    pub fn new() -> Self
    {
        return Self
        {
            heap_vec: Vec::new(),
        }
    }

    pub fn add(&mut self, value: K, data: V)
    {
        let new_heap_node: HeapNode<K, V> = HeapNode::new(value, data);
        self.heap_vec.push(new_heap_node);

        // perform heapify up
        let mut current_idx: usize = self.heap_vec.len() - 1;
        for _ in 0..self.heap_vec.len()
        {
            if current_idx == 0
            {
                break;
            }

            let parent_idx: usize = (current_idx - 1) / 2;
            if self.heap_vec[parent_idx].key > self.heap_vec[current_idx].key
            {
                self.heap_vec.swap(current_idx, parent_idx);
                current_idx = parent_idx;
            }
            else // parent is smaller than current
            {
                break;
            }
        }
    }

    pub fn replace_first(&mut self, value: K, data: V)
    {
        let new_heap_node: HeapNode<K, V> = HeapNode::new(value, data);
        self.heap_vec[0] = new_heap_node;

        self.trickle_down();

    }
    /*
    pub fn pop_at(&mut self, idx: usize) -> HeapNode<K, V>
    {
        // obtain first and replace with last
        let first_node: HeapNode<K, V> = self.heap_vec[idx].clone();
        let last_idx: usize = self.heap_vec.len() - 1;
        self.heap_vec.swap(idx, last_idx);
        self.heap_vec.pop().unwrap();

        self.trickle_down();

        return first_node;
    }

    pub fn clone_at(&self, idx: usize) -> HeapNode<K, V>
    {
        return self.heap_vec[idx].clone();
    }
    */

    pub fn mut_ref_at(&mut self, idx: usize) -> &mut HeapNode<K, V>
    {
        return &mut self.heap_vec[idx];
    }

    pub fn display(&self)
    {
        for node in &self.heap_vec
        {
            println!("{:?}", node);
        }
    }

    // trickle down from root
    pub fn trickle_down(&mut self)
    {
        let mut current_idx: usize = 0;
        for _ in 0..self.heap_vec.len()
        {
            let left_idx: usize = (current_idx * 2) + 1;
            let right_idx: usize = (current_idx * 2) + 2;

            // checking both left and right out of bounds
            if left_idx >= self.heap_vec.len() && right_idx >= self.heap_vec.len()
            {
                break;
            }
            // if only left bound is available
            else if !(left_idx >= self.heap_vec.len()) && right_idx >= self.heap_vec.len()
            {
                if self.heap_vec[left_idx].key >= self.heap_vec[current_idx].key
                {
                    break;
                }
                else
                {
                    self.heap_vec.swap(current_idx, left_idx);
                    current_idx = left_idx;
                }
            }
            else // both left and right child are available
            {
                // checking if both are larger (min heap conditions satisfied)
                if self.heap_vec[left_idx].key >= self.heap_vec[current_idx].key &&
                    self.heap_vec[right_idx].key >= self.heap_vec[current_idx].key
                {
                    break;
                }

                if self.heap_vec[left_idx].key <= self.heap_vec[right_idx].key
                {
                    self.heap_vec.swap(current_idx, left_idx);
                    current_idx = left_idx;
                }
                else if self.heap_vec[left_idx].key > self.heap_vec[right_idx].key
                {
                    self.heap_vec.swap(current_idx, right_idx);
                    current_idx = right_idx;
                }
            }
        }
    }

    pub fn get_random(&mut self, chance: f32) -> usize
    {
        let mut current_idx: usize = 0;
        for _ in 0..self.heap_vec.len()
        {
            if rand::thread_rng().gen_bool(chance as f64)
            {
                break;
            }

            let left_idx: usize = (current_idx * 2) + 1;
            let right_idx: usize = (current_idx * 2) + 2;

            // checking both left and right out of bounds
            if left_idx >= self.heap_vec.len() && right_idx >= self.heap_vec.len()
            {
                break;
            }
            // if only left bound is available
            else if !(left_idx >= self.heap_vec.len()) && right_idx >= self.heap_vec.len()
            {
                current_idx = left_idx;
            }
            else // both left and right child are available
            {
                // checking if both are larger (min heap conditions satisfied)
                if rand::thread_rng().gen_bool(0.5)
                {
                    current_idx = left_idx;
                }
                else
                {
                    current_idx = right_idx;
                }
            }
        }

        return current_idx;

    }

    pub fn len(&self) -> usize
    {
        return self.heap_vec.len();
    }
    pub fn clear(&mut self)
    {
        self.heap_vec.clear();
    }
}