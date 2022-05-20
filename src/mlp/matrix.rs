// Filename: matrix.rs
// Dragon's Rust code
// Created at 2022/05/19 10:18
// Edit with VS Code

use std::fmt::{self, Display};
use std::ops::{Add, BitAnd, Mul, Neg, Sub};


/// Use a vector to represent a matrix
/// matrix index \[i\]\[j\] => vector index \[i * cols + j\]
/// a transpose matrix iter each row by cols to create a new vector.
/// Rewrite ops + - * and neg, matmul(&).
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T> {
    pub v: Vec<T>,
    pub rows: usize,
    pub cols: usize,
}

impl<T: Copy> Matrix<T> {
    pub fn new(v: Vec<T>, rows: usize, cols: usize) -> Self {
        assert!(v.len() == rows * cols);
        Matrix {
            v,
            rows,
            cols,
        }
    }

    pub fn from_vec(vec: Vec<Vec<T>>) -> Self {
        let rows = vec.len();
        let cols = vec[0].len();
        let mut v = Vec::with_capacity(rows * cols);
        for row in vec.iter() {
            for elem in row.iter() {
                v.push(*elem);
            }
        }
        Matrix {
            v,
            rows,
            cols,
        }
    }

    pub fn transpose(&self) -> Self {
        let mut v = Vec::with_capacity(self.cols * self.rows);
        for j in 0..self.cols {
            for i in 0..self.rows {
                v.push(self.v[i * self.cols + j]);
            }
        }
        Matrix::new(v, self.cols, self.rows)
    }
}

impl<T> Add for Matrix<T>
where
    T: Add<Output = T> + Copy,
{
    type Output = Self;
    fn add(self, other: Self) -> Self {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut v = Vec::new();
        for i in 0..self.v.len() {
            v.push(self.v[i] + other.v[i]);
        }
        Matrix::new(v, self.rows, self.cols)
    }
}

impl<T> Mul for Matrix<T>
where
    T: Mul<Output = T> + Copy,
{
    type Output = Self;

    fn mul(self, other: Matrix<T>) -> Matrix<T> {
        let mut v = Vec::new();
        for i in 0..self.v.len() {
            v.push(self.v[i] * other.v[i]);
        }
        Matrix::new(v, self.rows, self.cols)
    }
}

impl<T> Sub for Matrix<T>
where
    T: Sub<Output = T> + Copy,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let mut v = Vec::new();
        for i in 0..self.v.len() {
            v.push(self.v[i] - other.v[i]);
        }
        Matrix::new(v, self.rows, self.cols)
    }
}

impl<T> Neg for Matrix<T>
where
    T: Neg<Output = T> + Copy,
{
    type Output = Self;
    fn neg(self) -> Self {
        let mut v = Vec::new();
        for i in 0..self.v.len() {
            v.push(-self.v[i]);
        }
        Matrix::new(v, self.rows, self.cols)
    }
}

// impl matrix multiplication by '&'
impl<T> BitAnd for Matrix<T>
where
    T: Mul<Output = T> + Copy,
{
    type Output = Self;
    fn bitand(self, other: Self) -> Self {
        assert_eq!(self.cols, other.rows);

        let mut v = Vec::new();
        for i in 0..self.rows {
            for j in 0..self.cols {
                v.push(self.v[i * self.cols + j] * other.v[j * other.cols + i]);
            }
        }
        Matrix::new(v, self.rows, self.cols)
    }
}

impl<T: Display> fmt::Display for Matrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for i in 0..self.rows {
            if i == 0 {
                write!(f, "⌈")?
            } else if i == self.rows - 1 {
                write!(f, "⌊")?
            } else {
                write!(f, "|")?
            }
            for j in 0..self.cols {
                write!(f, "{:.8}", self.v[i * self.cols + j])?;
                if j != self.cols - 1 {
                    write!(f, ", ")?;
                }
            }
            if i == 0 {
                writeln!(f, "⌉")?
            } else if i == self.rows - 1 {
                writeln!(f, "⌋")?
            } else {
                writeln!(f, "|")?
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod test{
    use super::*;

    #[test]
    fn test_matrix_ops() {
        let a = Matrix::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2, 5);
        let b = Matrix::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2, 5);
        let c = b.transpose();
        assert_eq!(a.clone() + b.clone(), Matrix::new(vec![2, 4, 6, 8, 10, 12, 14, 16, 18, 20], 2, 5));
        assert_eq!(a.clone() * b.clone(), Matrix::new(vec![1, 4, 9, 16, 25, 36, 49, 64, 81, 100], 2, 5));
        assert_eq!(a.clone() - b.clone(), Matrix::new(vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 2, 5));
        assert_eq!(-a.clone(), Matrix::new(vec![-1, -2, -3, -4, -5, -6, -7, -8, -9, -10], 2, 5));
        assert_eq!(c, Matrix::new(vec![1, 6, 2, 7, 3, 8, 4, 9, 5, 10], 5, 2));
        assert_eq!(a.clone() & c.clone(), Matrix::new(vec![1, 4, 9, 16, 25, 36, 49, 64, 81, 100], 2, 5));
    }
}