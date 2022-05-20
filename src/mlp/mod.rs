use rand;

mod matrix;
pub mod activation;
pub use self::matrix::Matrix;

pub fn init_matrix(rows: usize, cols: usize) -> Matrix<f64> {
    let mut v = Vec::with_capacity(rows * cols);
    for _ in 0..rows * cols {
        v.push(rand::random::<f64>());
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