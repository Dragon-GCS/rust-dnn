// Filename: matrix.rs
// Dragon's Rust code
// Created at 2022/05/19 10:18
// Edit with VS Code

use std::fmt::{self, Display};
use std::ops::{Add, BitAnd, Index, Mul, Neg, Range, Sub};

use num_traits::{Float, NumAssign, NumOps};

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
    pub fn from_vec(vec: Vec<Vec<T>>) -> Self {
        let rows = vec.len();
        let cols = vec[0].len();
        let v = vec.into_iter().flatten().collect();
        Self::new(v, rows, cols)
    }
}
impl<T: Copy> Matrix<T> {
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
    T: NumAssign + Copy + Default,
{
    pub fn dim_sum(&self, dim: usize) -> Matrix<T> {
        let (length, rows, cols) = if dim == 0 {
            (self.cols, 1, self.cols)
        } else {
            (self.rows, self.rows, 1)
        };
        let mut v = vec![T::default(); length];
        self.v.iter().enumerate().for_each(|(i, &x)| {
            let idx = if dim == 0 {
                i % self.cols
            } else {
                i / self.cols
            };
            v[idx] += x;
        });
        Matrix { v, rows, cols }
    }

    pub fn square(&self) -> Self {
        Matrix {
            v: self.v.iter().map(|&x| x * x).collect(),
            rows: self.rows,
            cols: self.cols,
        }
    }
}

macro_rules! scale_ops {
    ($ops:tt, $mth:ident) => {
        impl<T: NumOps + Copy> Matrix<T> {
            pub fn $mth(self, num: T) -> Self {
            Matrix{
                v: self.v.iter().map(|&x| x $ops num).collect(),
                rows: self.rows,
                cols: self.cols,
            }
        }
    }
    };
}
scale_ops!(+, add_num);
scale_ops!(-, sub_num);
scale_ops!(*, mul_num);
scale_ops!(/, div_num);

impl<T: Float> Matrix<T> {
    pub fn sqrt(&self) -> Self {
        Matrix {
            v: self.v.iter().map(|&x| x.sqrt()).collect(),
            rows: self.rows,
            cols: self.cols,
        }
    }
    pub fn exp(&self) -> Self {
        Matrix {
            v: self.v.iter().map(|&x| x.exp()).collect(),
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
        let (stride, num_elements, num_iterations) = if dim == 0 {
            (self.rows, self.cols, self.rows)
        } else {
            (1, self.rows, self.cols)
        };

        (0..num_elements)
            .map(|i| {
                let start_idx = if dim == 0 { i } else { i * self.cols };
                (0..num_iterations)
                    .map(|j| (j, self.v[start_idx + j * stride]))
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(max_index, _)| max_index)
                    .unwrap()
            })
            .collect()
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

macro_rules! impl_func {
    ($trt:ident, $ops:tt, $mth:ident, $struct:ident, $inner_impl:ident) => {
        impl<T: NumOps + Copy> $trt for $struct<T> {
            type Output = Matrix<T>;
            $inner_impl!($ops, $mth);
        }
        impl<T: NumOps + Copy> $trt for &$struct<T> {
            type Output = Matrix<T>;
            $inner_impl!($ops, $mth);
        }
    };
    ($trt:ident, $struct:ident, $inner_impl:ident) => {
        impl<T: NumAssign + Default + Copy> $trt for $struct<T> {
            type Output = Matrix<T>;
            $inner_impl!();
        }
        impl<T: NumAssign + Default + Copy> $trt for &$struct<T> {
            type Output = Matrix<T>;
            $inner_impl!();
        }
    };
}
macro_rules! operations {
    ($ops:tt, $mth:ident) => {
        fn $mth(self, other: Self) -> Self::Output {
            assert!(
                (self.rows == other.rows || self.cols == other.cols)
                    && (self.v.len() % other.v.len() == 0)
            );
            let v = self
                .v
                .iter()
                .enumerate()
                .map(|(i, &a)| {
                    a $ops if other.cols == 1 {
                        other.v[i / self.cols]
                    } else if other.rows == 1 {
                        other.v[i % self.cols]
                    } else {
                        other.v[i]
                    }
                })
                .collect();
            Matrix::new(v, self.rows, self.cols)
        }
    };
}
macro_rules! mat_mul {
    () => {
        fn bitand(self, other: Self) -> Self::Output {
            assert_eq!(self.cols, other.rows);

            let mut v = vec![T::default(); self.rows * other.cols];
            for row in 0..self.rows {
                for col in 0..self.cols {
                    let val = self.v[row * self.cols + col];
                    for other_col in 0..other.cols {
                        let result_index = row * other.cols + other_col;
                        let other_index = col * other.cols + other_col;
                        v[result_index] += val * other.v[other_index];
                    }
                }
            }
            Matrix::new(v, self.rows, other.cols)
        }
    };
}

impl_func!(Add, +, add, Matrix, operations);
impl_func!(Sub, -, sub, Matrix, operations);
impl_func!(Mul, *, mul, Matrix, operations);
impl_func!(BitAnd, Matrix, mat_mul);

impl<T: Neg<Output = T> + Copy> Neg for &Matrix<T> {
    type Output = Matrix<T>;
    fn neg(self) -> Self::Output {
        let v = self.v.iter().map(|&x| -x).collect();
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
mod test {
    use super::*;

    #[test]
    fn test_matrix_ops() {
        let a = &Matrix::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2, 5);
        let b = &Matrix::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2, 5);
        let c = &b.transpose();
        let d = &Matrix::new(vec![1, 1, 1, 1, 1], 1, 5);
        assert_eq!(
            a + d,
            Matrix::new(vec![2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 2, 5)
        );
        assert_eq!(
            a + b,
            Matrix::new(vec![2, 4, 6, 8, 10, 12, 14, 16, 18, 20], 2, 5)
        );
        assert_eq!(
            a * b,
            Matrix::new(vec![1, 4, 9, 16, 25, 36, 49, 64, 81, 100], 2, 5)
        );
        assert_eq!(a - b, Matrix::new(vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 2, 5));
        assert_eq!(
            -a,
            Matrix::new(vec![-1, -2, -3, -4, -5, -6, -7, -8, -9, -10], 2, 5)
        );
        assert_eq!(c, &Matrix::new(vec![1, 6, 2, 7, 3, 8, 4, 9, 5, 10], 5, 2));
        assert_eq!(a & c, Matrix::new(vec![55, 130, 130, 330], 2, 2));
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
