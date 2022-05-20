use super::Matrix;

pub fn relu (mat: Matrix<f64>) -> Matrix<f64>
{
    let mut v = Vec::new();
    for i in 0..mat.v.len() {
        v.push(if mat.v[i] > 0.0 { mat.v[i] } else { 0.0 });
    }
    Matrix::new(v, mat.rows, mat.cols)
}

pub fn relu_primer (mat: Matrix<f64>) -> Matrix<f64> {
    let mut v = Vec::new();
    for i in 0..mat.v.len() {
        v.push(if mat.v[i] > 0.0 { 1.0 } else { 0.0 });
    }
    Matrix::new(v, mat.rows, mat.cols)
}

pub fn sigmoid (mat: Matrix<f64>) -> Matrix<f64> {
    let mut v = Vec::new();
    for i in 0..mat.v.len() {
        v.push(1.0 / (1.0 + (-mat.v[i]).exp()));
    }
    Matrix::new(v, mat.rows, mat.cols)
}

pub fn sigmoid_primer (mat: Matrix<f64>) -> Matrix<f64> {
    let mut v = Vec::new();
    for i in 0..mat.v.len() {
        v.push(mat.v[i] * (1.0 - mat.v[i]));
    }
    Matrix::new(v, mat.rows, mat.cols)
}

pub fn softmax (mat:Matrix<f64>) -> Matrix<f64> {
    let mut v = Vec::new();
    let mut sum = Vec::new();
    for i in 0..mat.rows {
        sum.push(0.0);
        for j in 0..mat.cols {
            let exp = mat.v[i * mat.cols + j].exp();
            v.push(exp);
            sum[i] += exp;
        }
    }
    for i in 0..mat.rows {
        for j in 0..mat.cols {
            v[i * mat.cols + j] /= sum[i];
        }
    }
    Matrix::new(v, mat.rows, mat.cols)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::mat;

    #[test]
    pub fn test_relu() {
        let m = mat![[1.,2.,3.], [-2.,-2.,-2.], [3.,-2.,-1.]];
        let r = mat![[1.,2.,3.], [0., 0., 0.], [3., 0., 0.]];
        assert_eq!(relu(m), r);
    }

    #[test]
    pub fn test_relu_primer() {
        let m = mat![[1.,2.,3.], [-2.,-2.,-2.], [3.,-2.,-1.]];
        let r = mat![[1.,1.,1.], [0., 0., 0.], [1., 0., 0.]];
        assert_eq!(relu_primer(m), r);
    }

    #[test]
    pub fn test_sigmoid() {
        let m = mat![[1.,2.,3.], [-2.,-2.,-2.]];
        let r = mat![
            [0.731058, 0.880797, 0.952574], 
            [0.119202, 0.119202, 0.1192029]];
        for (pred, real) in sigmoid(m).v.iter().zip(r.v.iter()) {
            assert!((pred - real).abs() < 1e-6);
        }
    }

    #[test]
    pub fn test_sigmoid_primer() {
        let m = mat![[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]];
        let r = mat![
            [0.09, 0.16, 0.21], 
            [0.21, 0.16, 0.09]];
        for (pred, real) in sigmoid_primer(m).v.iter().zip(r.v.iter()) {
            assert!((pred - real).abs() < 1e-6);
        }
    }

    #[test]
    pub fn test_softmax() {
        let m = mat![[1.,2.,3.], [2.,2.,2.], [3.,2.,1.]];
        let r = mat![
            [0.090031, 0.244728, 0.665241], 
            [0.333333, 0.333333, 0.333333], 
            [0.665241, 0.244728, 0.090031]];
        for (pred, real) in softmax(m).v.iter().zip(r.v.iter()) {
            assert!((pred - real).abs() < 1e-6);
        }
    }
}