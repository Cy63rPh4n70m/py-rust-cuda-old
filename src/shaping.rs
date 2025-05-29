use rand::Rng;

use crate::types::*;

pub fn shape_1d_to_3d(vec1d: &Vec<f32>, shape3d: (usize, usize, usize)) -> Vec<Vec<Vec<f32>>>
{
    let mut vec3d: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut index: usize = 0;

    for _z in 0..shape3d.0
    {
        let mut vec2d: Vec<Vec<f32>> = Vec::new();
        for _y in 0..shape3d.1
        {
            let mut row: Vec<f32> = Vec::new();
            for _x in 0..shape3d.2
            {
                row.push(vec1d[index]);
                index += 1
            }
            vec2d.push(row);
        }
        vec3d.push(vec2d);
    }

    return vec3d;
}

pub fn shape_1d_to_2d(vec1d: &Vec<f32>, shape2d: (usize, usize)) -> Matrix2D
{
    let mut vec2d: Vec<Vec<f32>> = Vec::new();
    let mut index: usize = 0;
    for _y in 0..shape2d.0
    {
        let mut row: Vec<f32> = Vec::new();
        for _x in 0..shape2d.1
        {
            row.push(vec1d[index]);
            index += 1;
        }
        vec2d.push(row);
    }

    return vec2d;
}

pub fn shape_3d_to_1d(vec3d: &Vec<Matrix2D>) -> Vec<f32>
{
    let mut vec1dshaped: Vec<f32> = Vec::new();
    for vec2d in vec3d
    {
        for vec1d in vec2d
        {
            for val in vec1d
            {
                vec1dshaped.push(*val);
            }
        }
    }
    return vec1dshaped;
}

pub fn shape_2d_to_1d(vec2: &Vec<Vec<f32>>) -> Vec<f32>
{
    let mut vec1dshaped: Vec<f32> = Vec::new();
    for vec1d in vec2
    {
        for val in vec1d
        {
            vec1dshaped.push(*val);
        }
    }

    return vec1dshaped;
}

pub fn obtain_shape_of_2d_mat(vec2d: &Matrix2D) -> (usize, usize)
{
    let y_shape: usize = vec2d.len();
    let x_shape: usize = vec2d[0].len();

    return (y_shape, x_shape);
}

pub fn obtain_shape_of_3d_mat(vec3d: &Vec<Matrix2D>) -> (usize, usize, usize)
{
    let z_shape: usize = vec3d.len();
    let y_shape: usize = vec3d[0].len();
    let x_shape: usize = vec3d[0][0].len();

    return (z_shape, y_shape, x_shape);
}

pub fn zero_3d_mat(z: usize, y: usize, x: usize, f_val: f32, random: bool) -> Vec<Matrix2D>
{
    let mut zero_3d: Vec<Vec<Vec<f32>>> = Vec::with_capacity(z);
    for _ in 0..z
    {
        let mut zero_2d: Vec<Vec<f32>> = Vec::with_capacity(y);
        for _ in 0..y
        {
            let mut zero_1d: Vec<f32> = Vec::with_capacity(x);
            for _ in 0..x 
            {
                if random
                {
                    zero_1d.push(rand::thread_rng().gen_range(-f_val..f_val));
                }
                else
                {
                    zero_1d.push(f_val);
                }
            }
            zero_2d.push(zero_1d);
        }
        zero_3d.push(zero_2d);
    }

    return zero_3d;
}