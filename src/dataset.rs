#![allow(unused_imports, unused_variables, dead_code)]
use std::{fs::File, io::{Read, Result}, slice::SliceIndex, ops::Index};
use rand::{thread_rng, seq::SliceRandom};
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

pub struct Datasets<T, U> {
    x: Matrix<T>,
    y: Vec<U>,
    i: usize,
    idx: Vec<usize>,
    batch_size: usize,
}

impl<T, U> Datasets<T, U> {
    pub fn new (x: Matrix<T>, y: Vec<U>, batch_size: usize) -> Self {
        let mut idx = (0..x.rows).collect::<Vec<usize>>();
        idx.shuffle(&mut rand::thread_rng());
        Datasets {
            x,
            y,
            i: 0,
            idx,
            batch_size
        }
    }

    pub fn set_batch_size(&mut self, batch_size: usize) {
        self.batch_size = batch_size;
    }

    pub fn shuffle(&mut self) {
        self.i = 0;
        self.idx.shuffle(&mut rand::thread_rng());
    }

    pub fn len(&self) -> usize {
        (self.idx.len() - 1) / self.batch_size + 1
    }
}

impl<T, U> Iterator for Datasets<T, U>
where
    T: Copy,
    U: Copy,
{
    type Item = (usize, Matrix<T>, Vec<U>);
    fn next(&mut self) -> Option<Self::Item> {
        let start = self.i * self.batch_size;
        if start >= self.idx.len() {
            return None;
        }
        let end = self.idx.len().min((self.i + 1) *  self.batch_size);
        let mut batch_x = Vec::with_capacity(self.x.cols * self.batch_size);
        let mut batch_y = Vec::with_capacity(self.batch_size);
        for i in start..end {
            batch_x.extend(self.x[self.idx[i]].to_vec());
            batch_y.push(self.y[self.idx[i]]);
        }
        self.i += 1;
        Some((self.i - 1, Matrix::new(batch_x, batch_y.len(), self.x.cols), batch_y))
    }
}
