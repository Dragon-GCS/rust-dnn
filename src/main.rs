// Filename: main.rs
// Dragon's Rust code
// Created at 2022/05/19 10:18
// Edit with VS Code

#![allow(dead_code, unused_imports, unused_variables)]
mod dataset;

use dataset::{read_img, read_label, Datasets};
use mlp::{functions, layers::Linear, mat, Matrix, MLP};
use std::io::{self, Read, Write};
use std::time::Instant;

fn train(model: &mut MLP, dataset: &mut Datasets<f64, usize>, lr: f64, epoch: usize) {
    let mut step = 0;
    let batches = dataset.len();
    let mut metric = functions::Metric::new();
    for i in 0..epoch {
        dataset.shuffle();
        metric.reset();
        let now = Instant::now();
        for (batch, image, label) in &mut *dataset {
            let y = functions::one_hot(&label, 10);
            let (output, loss) = model.forward(&image, &y);
            let pred = output.argmax(1);
            let acc = metric.update(&pred, &label);
            if batch % 100 == 99 || batch == 0 {
                print!(
                    "\r[{}/{}]batch: {}/{}, loss: {:.6}, acc: {:.4} {:2}ms/batch",
                    i + 1,
                    epoch,
                    batch + 1,
                    batches,
                    loss,
                    acc,
                    now.elapsed().as_millis() / (batch as u128 + 1)
                );
                io::stdout().flush().unwrap();
            }
            model.backward(&y, lr);
            step += 1
        }
        println!();
    }
}

fn main() {
    let data_dir: &str = "E:\\ProjectFiles\\Python\\04_DeepLearning\\Datasets\\mnist\\raw";
    let train_img: &str = "train-images-idx3-ubyte";
    let train_label: &str = "train-labels-idx1-ubyte";

    let image = read_img(&format!("{}/{}", data_dir, train_img));
    let labels = read_label(&format!("{}/{}", data_dir, train_label));
    let batch = 25;
    let mut dataset = Datasets::new(image, labels, batch);

    let mut layers = Vec::new();
    layers.push(Linear::new(784, 32, true));
    layers.push(Linear::new(32, 32, true));
    layers.push(Linear::new(32, 10, false));
    let mut model = MLP::new(layers);

    let lr = 1e-3;
    train(&mut model, &mut dataset, lr, 20);
}
