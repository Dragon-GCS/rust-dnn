use rand::Rng;

pub mod functions;
pub mod layers;
mod matrix;
pub use self::matrix::Matrix;

pub fn init_matrix(rows: usize, cols: usize) -> Matrix<f64> {
    let mut v = vec![0.; rows * cols];
    let mut rng = rand::thread_rng();
    for elem in v.iter_mut() {
        *elem = rng.gen::<f64>() * 2. - 1.;
    }
    Matrix::new(v, rows, cols)
}

#[macro_export]
macro_rules! mat {
    ($([$($x:expr),* $(,)*]),+ $(,)*) => {
        $crate::Matrix::from_vec({
        let mut buff = Vec::new();
        $(
            let v = vec![$($x),*];
            buff.push(v);
        )*
        buff
    })};
}

pub struct MLP {
    pub linear: Vec<layers::Linear>,
    pub output: layers::CEOutput,
}

impl MLP {
    pub fn new(layers: Vec<layers::Linear>) -> Self {
        let output = layers::CEOutput::new();
        MLP {
            linear: layers,
            output,
        }
    }

    pub fn forward(&mut self, x: &Matrix<f64>, y: &Matrix<f64>) -> (Matrix<f64>, f64) {
        let mut z = x.clone();
        for layer in self.linear.iter_mut() {
            z = layer.forward(&z);
        }
        self.output.forward(&z, y)
    }

    pub fn backward(&mut self, y: &Matrix<f64>, lr: f64) {
        let mut grad = self.output.backward(y);
        for layer in self.linear.iter_mut().rev() {
            grad = layer.backward(&grad, lr);
        }
    }
}
