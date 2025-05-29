use ndarray::{Array2, Axis};
use rayon::iter::IntoParallelIterator;
use rayon::prelude::*;

pub fn par_matmul(arr1: &Array2<f64>, arr2: &Array2<f64>, arr3: &mut Array2<f64>)
{
    let shape: usize = arr1.shape()[1];
    arr1.axis_iter(Axis(0))
        .into_par_iter()
        .zip(arr3.axis_iter_mut(Axis(0)))
        //.enumerate()
        .for_each(
            |(a1, mut a3)|
            {
                a3 += &(&a1.into_shape((shape, 1)).unwrap() + arr2).sum_axis(Axis(0));
            }
        );
    //println!("{:?}", arr3);
}