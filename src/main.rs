// Filename: main.rs
// Dragon's Rust code
// Created at 2022/05/19 10:18
// Edit with VS Code

mod dataset;

use dataset::{read_img, read_label, Datasets};
use mlp::{functions, layers::Linear, MLP};
use std::io::{stdout, Write};
use std::time::Instant;

fn train(model: &mut MLP, dataset: &mut Datasets<f64, usize>, lr: f64, epoch: usize) {
    let batches = dataset.len();
    let mut metric = functions::Metric::new();
    for i in 1..(epoch + 1) {
        dataset.shuffle();
        metric.reset();
        let now = Instant::now();
        for (batch, (image, label)) in dataset.into_iter().enumerate() {
            let batch = batch + 1;
            let y = functions::one_hot(&label, 10);
            let (output, loss) = model.forward(&image, &y);
            let pred = output.argmax(1);
            let acc = metric.update(&pred, &label);
            if batch % 100 == 0 {
                print!(
                    "\r[{i}/{epoch}]batch: {batch}/{batches}, loss: {loss:.6}, acc: {acc:.4} {:}us/batch",
                    now.elapsed().as_micros() / batch as u128
                );
                stdout().flush().unwrap();
            }
            model.backward(&y, lr);
        }
        println!();
    }
}

fn main() {
    let data_dir: &str = "D:\\Projects\\DeepLearning\\Datasets\\mnist\\raw";
    let train_img: &str = "train-images-idx3-ubyte";
    let train_label: &str = "train-labels-idx1-ubyte";

    let image = read_img(&format!("{}/{}", data_dir, train_img));
    let labels = read_label(&format!("{}/{}", data_dir, train_label));
    let batch = 25;
    let mut dataset = Datasets::new(image, labels, batch);
    let layers = vec![
        Linear::new(784, 32, true),
        Linear::new(32, 32, true),
        Linear::new(32, 10, false),
    ];
    let mut model = MLP::new(layers);

    let lr = 1e-3;
    train(&mut model, &mut dataset, lr, 20);
}
