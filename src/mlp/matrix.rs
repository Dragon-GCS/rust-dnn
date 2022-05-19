// Filename: matrix.rs
// Dragon's Rust code
// Created at 2022/05/19 10:18
// Edit with VS Code

use std::ops::{Add, Mul, Neg, Sub, BitAnd};

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T>{
    v: Vec<T>,
    rows: usize,
    cols: usize,
}


impl <T: Copy> Matrix<T> {
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

#[macro_export]
macro_rules! mat {
    ($([$($x:expr),* $(,)*]),+ $(,)*) => {{
        let (mut rows, mut cols) = (0, 0);
        let mut buff = Vec::new();
        $(
            $(buff.push($x); cols += 1;)*
            rows += 1;
        )*
        $crate::Matrix::new(buff, rows, cols / rows)
    }};
}

impl<T> Add for Matrix<T> 
where T: Add<Output = T> + Copy {
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

impl <T> Mul for Matrix<T> 
where T: Mul<Output = T> + Copy {
    type Output = Self;

    fn mul(self, other: Matrix<T>) -> Matrix<T> {
        let mut v = Vec::new();
        for i in 0..self.v.len() {
            v.push(self.v[i] * other.v[i]);
        }
        Matrix::new(v, self.rows, self.cols)
    }
}

impl <T> Sub for Matrix<T> 
where T: Sub<Output = T> + Copy {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let mut v = Vec::new();
        for i in 0..self.v.len() {
            v.push(self.v[i] - other.v[i]);
        }
        Matrix::new(v, self.rows, self.cols)
    }
}

impl <T> Neg for Matrix<T> 
where T: Neg<Output = T> + Copy {
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
impl <T> BitAnd for Matrix<T> 
where T: Mul<Output = T> + Copy {
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

