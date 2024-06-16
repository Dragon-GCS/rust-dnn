use rand::Rng;

pub mod functions;
pub mod layers;
mod matrix;
pub use self::matrix::Matrix;

pub fn init_matrix(rows: usize, cols: usize) -> Matrix<f64> {
    let mut rng = rand::thread_rng();
    let v = (0..rows * cols)
        .map(|_| rng.gen::<f64>() * 2. - 1.)
        .collect();
    Matrix::new(v, rows, cols)
}

// macro_export is used to make the macro available to other modules
// use create::mat
#[macro_export]
macro_rules! mat {
    ($([$($x:expr),*]),+) => {
        $crate::Matrix::from_vec(
        vec![$(vec![$($x),*]),*]
   )};
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

    pub fn forward(&mut self, x: &Matrix<f64>, y: &Matrix<f64>) -> (&Matrix<f64>, f64) {
        let mut inputs = x;
        for layer in self.linear.iter_mut() {
            inputs = layer.forward(inputs);
        }
        self.output.forward(inputs, y)
    }

    pub fn backward(&mut self, y: &Matrix<f64>, lr: f64) {
        let mut grad = self.output.backward(y);
        for layer in self.linear.iter_mut().rev() {
            grad = layer.backward(&grad, lr);
        }
    }
}

#[cfg(test)]
mod test {
    use crate::mat;

    #[test]
    pub fn test_mat_macro() {
        let m = mat![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]];
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 3);
        assert_eq!(m.v, vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
    }
}
