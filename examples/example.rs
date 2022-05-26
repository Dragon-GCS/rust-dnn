use mlp::{MLP, layers::Linear, mat, functions};

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
    let mut metric = functions::Metric::new();
    for _ in 0..100 {
        let (output, loss) = model.forward(&x, &y);
        let pred = output.argmax(1);
        let acc = metric.update(&pred, &label);
        println!("loss: {:.4} accuracy: {:.4}", loss, acc);
        model.backward(&y, lr);
        metric.reset();
    }
}

fn main() {
    test_train();
    let v = vec![1,2,3,4];
    print!("{:?}", v);
}