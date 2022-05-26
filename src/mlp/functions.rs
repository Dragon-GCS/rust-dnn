use super::Matrix;

pub fn one_hot(label: &Vec<usize>, depth: usize) -> Matrix<f64> {
    let rows = label.len();
    let mut v = Vec::new();
    for i in 0..rows {
        for j in 0..depth {
            if label[i] == j {
                v.push(1.0);
            } else {
                v.push(0.0);
            }
        }
    }
    Matrix::new(v, rows, depth)
}

/// Relu function
/// $$f(x) =\begin{cases} x&x>0\\0 & x\le0\end{cases}$$
pub fn relu (mat: &Matrix<f64>) -> Matrix<f64>
{
    let mut v = Vec::new();
    for i in 0..mat.v.len() {
        v.push(if mat.v[i] > 0.0 { mat.v[i] } else { 0.0 });
    }
    Matrix::new(v, mat.rows, mat.cols)
}

/// Prime of the ReLU function
/// $$f\prime(x) =\begin{cases} 1&x>0\\0&x\le0\end{cases}$$
pub fn relu_prime (mat: &Matrix<f64>) -> Matrix<f64> {
    let mut v = Vec::new();
    for i in 0..mat.v.len() {
        v.push(if mat.v[i] > 0.0 { 1.0 } else { 0.0 });
    }
    Matrix::new(v, mat.rows, mat.cols)
}

/// Softmax function
/// $$\sigma\prime(x)=\sigma(x)(1-\sigma(x))$$
pub fn sigmoid (mat: &Matrix<f64>) -> Matrix<f64> {
    let mut v = Vec::new();
    for i in 0..mat.v.len() {
        v.push(1.0 / (1.0 + (-mat.v[i]).exp()));
    }
    Matrix::new(v, mat.rows, mat.cols)
}

/// Prime of sigmoid function
/// $$\sigma\prime(x)=\sigma(x)(1-\sigma(x))$$
pub fn sigmoid_prime (mat: &Matrix<f64>) -> Matrix<f64> {
    let mut v = Vec::new();
    for i in 0..mat.v.len() {
        v.push(mat.v[i] * (1.0 - mat.v[i]));
    }
    Matrix::new(v, mat.rows, mat.cols)
}

/// Calculate the softmax of a matrix
/// $$f(x) = \frac{e^{x_i}}{\sum_{j}e^{x_j}}$$
pub fn softmax (mat: &Matrix<f64>, dim: usize) -> Matrix<f64> {
    let mut v = Vec::new();
    let mut sum = Vec::new();
    let (mut dim1, mut dim2) = (mat.rows, mat.cols);
    if dim == 0 {(dim1, dim2) = (mat.cols, mat.rows);}
    for i in 0..dim1 {
        sum.push(0.0);
        for j in 0..dim2 {
            let exp = if dim == 0 {mat.v[j * mat.cols + i].exp()} else {mat.v[i * mat.cols + j].exp()};
            v.push(exp);
            sum[i] += exp;
        }
    }
    for i in 0..dim1 {
        for j in 0..dim2 {
            let index = if dim == 0 {j * mat.cols + i} else {i * mat.cols + j};
            v[index] /= sum[i];
        }
    }
    Matrix::new(v, mat.rows, mat.cols)
}

/// Prime of softmax function
/// $$f\prime(x)= f(x_i)(1 - f(x_i)) + \sum_{j\neq i} f(x_j) f(x_i) $$
pub fn softmax_prime(mat: &Matrix<f64>) -> Matrix<f64> {
    let mut v = Vec::new();
    for elem in mat.v.iter() {
        v.push(*elem * (1.0 - *elem));
    }
    for i in 0..mat.rows {
        for j in 0..mat.cols {
            for k in 0..mat.rows {
                if j != i {
                    v[i * mat.cols + j] += mat.v[i * mat.cols + j] * mat.v[i * mat.cols + k];
                }
            }
        }
    }
    Matrix::new(v, mat.rows, mat.cols)
}

/// Calculate the cross entropy loss of two matrix which not be softmaxed.
/// $$f(\hat{y}, y) = -\frac{1}{n}\sum^n_{i=1}y_i\ln\hat{y}_i$$
pub fn cross_entropy (mat: &Matrix<f64>, target: &Matrix<f64>) -> f64 {
    let (mut sum, rows) = (0.0, mat.rows);
    for (pred, real) in mat.v.iter().zip(target.v.iter()) {
        if *real > 0. { sum += real * pred.ln() }
    }
    -sum / rows as f64
}

/// Prime of two matrix's cross entropy loss with soft.
/// $$f_i\prime(\hat{y}, y) = \hat{y_i} - y_i$$
pub fn cross_entropy_prime(mat: &Matrix<f64>, target: &Matrix<f64>) -> Matrix<f64> {
    let rows = mat.rows;
    let mut v = Vec::new();
    for (pred, real) in mat.v.iter().zip(target.v.iter()) {
        v.push((pred - real) / rows as f64);
    }
    Matrix::new(v, mat.rows, mat.cols)
}

pub struct  Metric {
    right: f64,
    total: f64,
}

impl Metric {
    pub fn new() -> Self {
        Self { right: 0., total: 0. }
    }

    pub fn update(&mut self, pred: &Vec<usize>, label: &Vec<usize>) -> f64 {
        for (&p, &l) in pred.iter().zip(label.iter()) {
            if p == l { self.right += 1. }
            self.total += 1.;
        }
        self.right / self.total
    }

    pub fn reset(&mut self) {
        self.right = 0.;
        self.total = 0.;
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::mat;

    // all the test cases are verified by PyTorch.
    #[test]
    pub fn test_one_hot() {
        let label = vec![1, 3, 2];
        assert_eq!(one_hot(&label, 4), mat![
            [0., 1., 0., 0.],
            [0., 0., 0., 1.],
            [0., 0., 1., 0.]
        ]);
    }

    #[test]
    pub fn test_relu() {
        let m = mat![[1.,2.,3.], [-2.,-2.,-2.], [3.,-2.,-1.]];
        let r = mat![[1.,2.,3.], [0., 0., 0.], [3., 0., 0.]];
        assert_eq!(relu(&m), r);
        let p = mat![[1.,1.,1.], [0., 0., 0.], [1., 0., 0.]];
        assert_eq!(relu_prime(&r), p);
    }

    #[test]
    
    pub fn test_sigmoid() {
        let m = mat![[1.,2.,3.], [-2.,-2.,-2.]];
        let r = mat![
            [0.731058, 0.880797, 0.952574], 
            [0.119202, 0.119202, 0.1192029]];
        for (pred, real) in sigmoid(&m).v.iter().zip(r.v.iter()) {
            assert!((pred - real).abs() < 1e-6);
        }
        let p = mat![
            [0.1966119, 0.1049936, 0.0451766], 
            [0.1049936, 0.1049936, 0.1049936]];
        for (pred, real) in sigmoid_prime(&r).v.iter().zip(p.v.iter()) {
            assert!((pred - real).abs() < 1e-6);
        }
    }

    #[test]
    pub fn test_softmax() {
        let m = mat![[1.,2.,3.], [1.,2.,1.], [3.,2.,1.]];
        let r = mat![
            [0.090031, 0.244728, 0.665241], 
            [0.211941, 0.576116, 0.211941], 
            [0.665241, 0.244728, 0.090031]];
        for (pred, real) in softmax(&m, 1).v.iter().zip(r.v.iter()) {
            println!("{:?} {:?}", pred, real);
            assert!((pred - real).abs() < 1e-6);
        }
        let p = mat![
            [0.081925, 0.4295642, 0.887936],
            [0.378962, 0.2442063, 0.378962],
            [0.8879364, 0.429564, 0.081925]];
        for (pred, real) in softmax_prime(&r).v.iter().zip(p.v.iter()) {
            assert!((pred - real).abs() < 1e-6);
        }
    }

    #[test]
    pub fn test_cross_entropy() {
        let m =  mat![
            [0.090031, 0.244728, 0.665241], 
            [0.211941, 0.576116, 0.211941], 
            [0.665241, 0.244728, 0.090031]];
        let y = mat![
            [1.,0.,0.],
            [0.,1.,0.],
            [0.,0.,1.]];
        assert!((cross_entropy(&m, &y) - 1.788882).abs() < 1e-6);
        let p = mat![
            [-0.303323, 0.081576, 0.221747], 
            [0.070647, -0.141294, 0.070647],
            [0.221747, 0.081576, -0.303323]];
        for (pred, real) in cross_entropy_prime(&m, &y).v.iter().zip(p.v.iter()) {
            assert!((pred - real).abs() < 1e-6);
        }
    }
}