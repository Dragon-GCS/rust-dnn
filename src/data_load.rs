#![allow(unused_imports, unused_variables, dead_code)]
use std::{fs::File, io::{Read, Result}};
use mlp::{mat, functions, Matrix};

fn read_file(path: &str) -> Vec<u8> {
    let mut file = File::open(path).expect("file not found");
    let mut data = Vec::new();
    file.read_to_end(&mut data).expect("read error");
    data
}

pub fn read_label(path: &str) -> Vec<usize> {
    let data = read_file(path);
    let num = u32::from_be_bytes(data[4..8].try_into().unwrap());
    println!("Load {} labels", num);
    let labels = data[8..].iter().map(|x| *x as usize).collect::<Vec<usize>>();
    assert!(labels.len() == num as usize);
    labels
}

pub fn read_img(path: &str) -> Matrix<f64> {
    let data = read_file(path);
    let num = u32::from_be_bytes(data[4..8].try_into().unwrap()) as usize;
    let (rows, cols) = (
        u32::from_be_bytes(data[8..12].try_into().unwrap()) as usize, 
        u32::from_be_bytes(data[12..16].try_into().unwrap()) as usize);
    println!("Load {} images, {}x{}", num, rows, cols);
    let images = data[16..].iter().map(|x| f64::from(*x) / 255.0).collect::<Vec<f64>>();
    assert!(images.len() == num * rows * cols);
    Matrix::new(images, num , rows * cols)
}