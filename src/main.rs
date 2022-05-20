// Filename: main.rs
// Dragon's Rust code
// Created at 2022/05/19 10:18
// Edit with VS Code

#![allow(dead_code, unused_imports)]
use mlp::{*, functions as F};
fn main() {
    let a = mat![[1.,2.,3.], [1.,2.,1.], [3.,2.,1.]];
    let label = vec![0, 1, 2];
    let y = F::one_hot(&label, 3);
    let b = F::softmax(&a);
    let loss = F::cross_entropy(&b, &y);
    println!("{:}", loss);

}
