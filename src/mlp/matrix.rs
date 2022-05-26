// Filename: matrix.rs
// Dragon's Rust code
// Created at 2022/05/19 10:18
// Edit with VS Code

use std::fmt::{self, Display};
use std::ops::{Add, AddAssign, BitAnd, Index, Mul, Neg, Range, Sub};

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

impl<T> Matrix<T> {
    pub fn new(v: Vec<T>, rows: usize, cols: usize) -> Self {
        assert!(v.len() == rows * cols);
        Matrix { v, rows, cols }
    }
}

impl<T: Copy> Matrix<T> {
    pub fn from_vec(vec: Vec<Vec<T>>) -> Self {
        let rows = vec.len();
        let cols = vec[0].len();
        let mut v = Vec::new();
        for i in 0..vec.len() {
            v.extend(&vec[i])
        }
        Self::new(v, rows, cols)
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

impl<T> Matrix<T>
where
    T: Add<Output = T> + Copy,
{
    pub fn dim_sum(&self, dim: usize) -> Matrix<T> {
        let mut v = Vec::new();
        let (mut rows, mut cols) = (self.rows, 1);
        let (mut dim1, mut dim2) = (self.rows, self.cols);
        if dim == 0 {
            (rows, cols) = (1, self.cols);
            (dim1, dim2) = (self.cols, self.rows);
        }
        for i in 0..dim1 {
            let start_idx = if dim == 0 { i } else { i * self.cols };
            v.push(self.v[start_idx]);
            for j in 1..dim2 {
                v[i] = v[i] + self.v[start_idx + j * cols];
            }
        }
        Matrix::new(v, rows, cols)
    }
}

impl Matrix<f64>
{
    pub fn square(&self) -> Self {
        Matrix {
            v: self.v.iter().map(|&x| x * x).collect(),
            rows: self.rows,
            cols: self.cols,
        }
    }

    pub fn sqrt(&self) -> Self {
        Matrix {
            v: self.v.iter().map(|&x| x.sqrt()).collect(),
            rows: self.rows,
            cols: self.cols,
        }
    }

    pub fn add_num(&self, num: f64) -> Self {
        Matrix {
            v: self.v.iter().map(|&x| x + num).collect(),
            rows: self.rows,
            cols: self.cols,
        }
    }

    pub fn mul_num(&self, num: f64) -> Self {
        Matrix {
            v: self.v.iter().map(|&x| x * num).collect(),
            rows: self.rows,
            cols: self.cols,
        }
    }

    pub fn div_num(&self, num: f64) -> Self {
        Matrix {
            v: self.v.iter().map(|&x| x / num).collect(),
            rows: self.rows,
            cols: self.cols,
        }
    }

    pub fn sub_num(&self, num: f64) -> Self {
        Matrix {
            v: self.v.iter().map(|&x| x - num).collect(),
            rows: self.rows,
            cols: self.cols,
        }
    }
}

impl<T> Matrix<T>
where
T: PartialOrd + Copy,
{
    pub fn argmax(&self, dim: usize) -> Vec<usize> {
        let mut v = Vec::new();
        let cols = if dim == 0 { self.rows } else { 1 };
        let (mut dim1, mut dim2) = (self.rows, self.cols);
        if dim == 0 {
            (dim1, dim2) = (self.cols, self.rows);
        }
        for i in 0..dim1 {
            let start_idx = if dim == 0 { i } else { i * self.cols };
            let (mut max_index, mut max) = (0, self.v[start_idx]);
            for j in 0..dim2 {
                let elem = self.v[start_idx + j * cols];
                if elem > max {
                    max = elem;
                    max_index = j;
                }
            }
            v.push(max_index)
        }
        v
    }
}

impl<T> Index<usize> for Matrix<T> {
    type Output = [T];

    fn index<'a>(&self, index: usize) -> &[T] {
        &self.v[index * self.cols..(index + 1) * self.cols]
    }
}

impl<T> Index<Range<usize>> for Matrix<T> {
    type Output = [T];

    fn index(&self, index: Range<usize>) -> &[T] {
        &self.v[index.start * self.cols..index.end * self.cols]
    }
}

impl<T> Add for Matrix<T>
where
T: Add<Output = T> + Copy,
{
    type Output = Self;
    fn add(self, other: Self) -> Self {
        assert!(
            (self.rows == other.rows || self.cols == other.cols)
            && (self.v.len() % other.v.len() == 0)
        );
        let mut v = Vec::with_capacity(self.v.len());

        for i in 0..self.v.len() {
            let other_idx = if other.cols == 1 {
                i / self.cols
            } else if other.rows == 1 {
                i % self.cols
            } else {
                i
            };
            v.push(self.v[i] + other.v[other_idx]);
        }
        Matrix::<T>::new(v, self.rows, self.cols)
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
        Matrix::<T>::new(v, self.rows, self.cols)
    }
}

// impl matrix multiplication by '&', iterate order by ikj
impl<T> BitAnd for Matrix<T>
where
    T: Add<Output = T> + Mul<Output = T> + AddAssign + Copy,
{
    type Output = Self;
    fn bitand(self, other: Self) -> Self {
        assert_eq!(self.cols, other.rows);

        let mut v = Vec::new();
        for i in 0..self.rows {
            for j in 0..other.cols {
                v.push(self.v[i * self.cols] * other.v[j]);
            }
            for k in 1..self.cols {
                let t = self.v[i * self.cols + k];
                for j in 0..other.cols {
                    v[i * other.cols + j] += t * other.v[k * other.cols + j];
                }
            }
        }
        Matrix::new(v, self.rows, other.cols)
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
mod test {
    use super::*;

    #[test]
    fn test_matrix_ops() {
        let a = Matrix::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2, 5);
        let b = Matrix::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2, 5);
        let c = b.transpose();
        let d = Matrix::new(vec![1, 1, 1, 1, 1], 1, 5);
        assert_eq!(
            a.clone() + d.clone(),
            Matrix::new(vec![2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 2, 5)
        );
        assert_eq!(
            a.clone() + b.clone(),
            Matrix::new(vec![2, 4, 6, 8, 10, 12, 14, 16, 18, 20], 2, 5)
        );
        assert_eq!(
            a.clone() * b.clone(),
            Matrix::new(vec![1, 4, 9, 16, 25, 36, 49, 64, 81, 100], 2, 5)
        );
        assert_eq!(
            a.clone() - b.clone(),
            Matrix::new(vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 2, 5)
        );
        assert_eq!(
            -a.clone(),
            Matrix::new(vec![-1, -2, -3, -4, -5, -6, -7, -8, -9, -10], 2, 5)
        );
        assert_eq!(c, Matrix::new(vec![1, 6, 2, 7, 3, 8, 4, 9, 5, 10], 5, 2));
        assert_eq!(
            a.clone() & c.clone(),
            Matrix::new(vec![55, 130, 130, 330], 2, 2)
        );
    }

    #[test]
    fn test_dim() {
        let a = Matrix::new(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 3, 4);
        assert_eq!(a.dim_sum(0), Matrix::new(vec![12, 15, 18, 21], 1, 4));
        assert_eq!(a.dim_sum(1), Matrix::new(vec![6, 22, 38], 3, 1));
        assert_eq!(a.argmax(0), vec![2, 2, 2, 2]);
        assert_eq!(a.argmax(1), vec![3, 3, 3]);
    }

    #[test]
    fn test_index() {
        let a = Matrix::new(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 3, 4);
        assert_eq!(a[0..2].to_vec(), vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }
}
