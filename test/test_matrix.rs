// Filename: test_matrix.rs
// Dragon's Rust code
// Created at 2022/05/19 10:18
// Edit with VS Code

use crate::Matrix;
// 测试模块[cfg(test)]或者测试函数#[test]和宏定义放在一起$crate引用就是失效，很迷
#[test]
fn test_matrix_ops() {
    let a = Matrix::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2, 5);
    let b = Matrix::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2, 5);
    assert_eq!(a.clone() + b.clone(), Matrix::new(vec![2, 4, 6, 8, 10, 12, 14, 16, 18, 20], 2, 5));
    assert_eq!(a.clone() * b.clone(), Matrix::new(vec![1, 4, 9, 16, 25, 36, 49, 64, 81, 100], 2, 5));
    assert_eq!(a.clone() - b.clone(), Matrix::new(vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 2, 5));
    assert_eq!(-a.clone(), Matrix::new(vec![-1, -2, -3, -4, -5, -6, -7, -8, -9, -10], 2, 5));
    assert_eq!(b.transpose(), Matrix::new(vec![1, 6, 2, 7, 3, 8, 4, 9, 5, 10], 5, 2));
    assert_eq!(a.clone() & b.transpose().clone(), Matrix::new(vec![1, 4, 9, 16, 25, 36, 49, 64, 81, 100], 2, 5));
    assert_eq!(mat![[1,2,3,4,5], [6,7,8,9,10]], a)
}