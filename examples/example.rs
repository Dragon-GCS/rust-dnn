#![allow(unused_imports, unused_variables, dead_code)]
use std::{fs::File, io::{Read, Result}};
use mlp::{mat, functions, Matrix};
const DATA_DIR: &str = "E:\\ProjectFiles\\Python\\04_DeepLearning\\Datasets\\mnist\\raw";
const TRAIN_IMG: &str = "train-images-idx3-ubyte";
const TRAIN_LABEL: &str = "train-labels-idx1-ubyte";
const TEST_IMG: &str = "t10k-images-idx3-ubyte";
const TEST_LABEL: &str = "t10k-labels-idx1-ubyte";

fn read_file(path: &str) -> Vec<u8> {
    let mut file = File::open(path).expect("file not found");
    let mut data = Vec::new();
    file.read_to_end(&mut data).expect("read error");
    data
}

fn read_label(path: &str) -> Vec<u8> {
    let data = read_file(path);
    let num = u32::from_be_bytes(data[4..8].try_into().unwrap());
    println!("Load {} labels", num);
    let labels = data[8..].to_vec();
    assert!(labels.len() == num as usize);
    labels
}

fn read_img(path: &str) -> Matrix<f64> {
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

fn main(){
    let label = read_label(&format!("{}/{}", DATA_DIR, TEST_LABEL));
    for i in 0..10 {
        print!("{} ", label[i]);
    }
    let image = read_img(&format!("{}/{}", DATA_DIR, TEST_IMG));
    println!("{} x {}", image.rows, image.cols);
    for i in 0..28 {
        for j in 0..28 {
            print!("{:.1} ", image.v[i * 28 + j]);
        }
        println!();
    }
}