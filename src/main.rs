// Filename: main.rs
// Dragon's Rust code
// Created at 2022/05/19 10:18
// Edit with VS Code

#![allow(dead_code, unused_imports, unused_variables)]
mod data_load;

use std::io::Read;
use mlp::{MLP, layers::Linear, mat, functions, Matrix};
use data_load::{read_label, read_img};

fn load_file(filename: &str) -> Vec<u8> {
    let mut f = std::fs::File::open(filename).unwrap();
    let mut buff = Vec::new();
    f.read_to_end(&mut buff).unwrap();
    buff
}

fn train(model: &mut MLP, x: &Matrix<f64>, y: &Matrix<f64>, lr: f64, epoch: usize, batch_size: usize) {
    for i in 0..epoch {
        for batch in 0..x.rows / batch_size {
            // let start = batch * batch_size;
            // let end = start + batch_size;
            // let x_batch = x.slice(start..end, ..);
            // let y_batch = y.slice(start..end, ..);
            // let (z, loss) = model.forward(&x_batch, &y_batch);
            // model.backward(&y_batch, lr);
            // println!("epoch: {}, batch: {}, loss: {}", i, batch, loss);
            ..;
        }
        if i % 10 == 0 {
            println!("epoch {}", i);
        }
    }
    model.backward(y, 0.1);
}

fn test_train() {
    let x = mat![[1.,2.,-3.], [1.,-2.,1.], [-3.,2.,1.], [1.,-2.,1.]];
    let label = vec![2, 1, 0, 1];
    let y = functions::one_hot(&label, 3);
    let lr = 1e-1;

    let mut layers = Vec::new();
    layers.push(Linear::new(3, 5, true));
    layers.push(Linear::new(5, 5, true));
    layers.push(Linear::new(5, 3, false));
    let mut model = MLP::new(layers);

    for _ in 0..100 {
        let (output, loss) = model.forward(&x, &y);
        let pred = output.argmax(1);
        let acc = functions::accuracy(&pred, &label);
        println!("loss: {:.4} accuracy: {:.4}", loss, acc);
        model.backward(&y, lr);
    }
}

pub fn test_mnist() {
    let data_dir: &str = "E:\\ProjectFiles\\Python\\04_DeepLearning\\Datasets\\mnist\\raw";
    let train_img: &str = "train-images-idx3-ubyte";
    let train_label: &str = "train-labels-idx1-ubyte";
    let test_img: &str = "t10k-images-idx3-ubyte";
    let test_label: &str = "t10k-labels-idx1-ubyte";

    let test_image = read_img(&format!("{}/{}",data_dir, test_img));
    let test_label = read_label(&format!("{}/{}", data_dir, test_label));
    let mut lr = 1e-2;
    let batch = 25;

    let mut layers = Vec::new();
    layers.push(Linear::new(784, 24, true));
    layers.push(Linear::new(24, 24, true));
    layers.push(Linear::new(24, 24, true));
    layers.push(Linear::new(24, 10, false));
    let mut model = MLP::new(layers);

    println!("Start training");
    for i in 0..10 {
        for b in 0..(test_image.rows / batch) - 1 {
            let start = b * batch;
            let end = (b + 1) * batch;
            let x_batch = Matrix::new(test_image.v[start * 784..end * 784].to_vec(), batch, 784);
            let label = test_label[start..end].to_vec();
            let y_batch = functions::one_hot(&label, 10);

            let (z, loss) = model.forward(&x_batch, &y_batch);
            let pred = z.argmax(1);
            let acc = functions::accuracy(&pred, &label);
            if b % 50 == 0 {
                println!("epoch: {}, batch: {:3}, loss: {:.4}, accuracy: {:.4}", i, b, loss, acc);
            }
            model.backward(&y_batch, lr);
        }
        lr *= 0.9;
    }

}
fn main() {
    test_mnist()
    // let a = Matrix::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2, 5);
    // let b = Matrix::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2, 5);
    // println!("{:}", a.clone() & b.transpose());
    // assert_eq!(
    //     a.clone() & c.clone(),
    //     Matrix::new(vec![55, 130, 130, 330], 2, 2)
    // );
}
