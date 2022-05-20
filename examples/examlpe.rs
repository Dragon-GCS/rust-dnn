
use mlp::{mat, activation};

fn main(){
    let m = mat![[1.,2.,3.], [2.,2.,2.], [3.,2.,1.]];
    println!("{:}", activation::softmax(m));
}