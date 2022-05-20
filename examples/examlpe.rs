
use mlp::{mat, functions};

fn main(){
    let m = mat![[1.,2.,3.], [2.,2.,2.], [3.,2.,1.]];
    println!("{:}", functions::softmax(&m));
}