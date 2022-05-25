// Filename: main.rs
// Dragon's Rust code
// Created at 2022/05/19 10:18
// Edit with VS Code

#![allow(dead_code, unused_imports, unused_variables)]
mod dataset;

use std::io::Read;
use mlp::{MLP, layers::Linear, mat, functions, Matrix};
use dataset::{read_label, read_img, Datasets};

fn load_file(filename: &str) -> Vec<u8> {
    let mut f = std::fs::File::open(filename).unwrap();
    let mut buff = Vec::new();
    f.read_to_end(&mut buff).unwrap();
    buff
}

fn train(model: &mut MLP, dataset: &mut Datasets<f64,usize>, lr: f64, epoch: usize) {
    let mut step = 0;
    let batches = dataset.len();
    for i in 0..epoch {
        dataset.shuffle();
        for (batch, image, label) in &mut *dataset {
            let y = functions::one_hot(&label, 10);
            let (output, loss) = model.forward(&image, &y);
            let pred = output.argmax(1);
            let acc = functions::accuracy(&pred, &label);
            if batch % 100 == 99 || batch == 0 {
                println!("[{}/{}]step: {}/{}, loss: {:.6}, acc: {:.4}",i + 1, epoch, batch + 1, batches, loss, acc);
            }
            model.backward(&y, lr);
            step += 1
        }
    }
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

    let image = read_img(&format!("{}/{}",data_dir, train_img));
    let labels = read_label(&format!("{}/{}", data_dir, train_label));
    let batch = 25;
    let mut dataset = Datasets::new(image, labels, batch);
    
    let mut layers = Vec::new();
    layers.push(Linear::new(784, 24, true));
    layers.push(Linear::new(24, 24, true));
    layers.push(Linear::new(24, 24, true));
    layers.push(Linear::new(24, 10, false));
    let mut model = MLP::new(layers);
    
    let lr = 1e-3;
    train(&mut model, &mut dataset, lr, 10);

}
fn main() {
    test_mnist()
}
