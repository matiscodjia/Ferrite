use crate::scalar::{fabs, Scalar};
use crate::vector::Vector;
use core::cmp::PartialEq;
use core::ops::{Add, Index, IndexMut, Mul, Sub};

/// A Static Matrix of ROWS x COLS, stored entirely on the stack.
/// Uses a 2D array to remain 100% static and compatible with stable Rust.
#[derive(Clone, Copy, Debug)]
pub struct Matrix<const ROWS: usize, const COLS: usize> {
    data: [[Scalar; COLS]; ROWS],
}

impl<const ROWS: usize, const COLS: usize> Matrix<ROWS, COLS> {
    /// Creates a new matrix filled with zeros.
    pub fn new() -> Self {
        Self {
            data: [[0.0; COLS]; ROWS],
        }
    }

    /// Returns the number of rows.
    pub const fn rows(&self) -> usize {
        ROWS
    }

    /// Returns the number of columns.
    pub const fn cols(&self) -> usize {
        COLS
    }

    /// Extracts a column as a Static Vector of size ROWS.
    pub fn get_col(&self, col: usize) -> Option<Vector<ROWS>> {
        if col >= COLS {
            return None;
        }
        let mut col_data = [0.0; ROWS];
        for i in 0..ROWS {
            col_data[i] = self.data[i][col];
        }
        Some(Vector::new(col_data))
    }

    /// Injects a Static Vector into a matrix column.
    pub fn set_col(&mut self, col: usize, vec: &Vector<ROWS>) {
        if col >= COLS {
            panic!("Column index out of bounds");
        }
        let vec_data = vec.get_data();
        for i in 0..ROWS {
            self.data[i][col] = vec_data[i];
        }
    }

    /// Creates a Matrix from an array of column vectors.
    pub fn from_cols(cols: [Vector<ROWS>; COLS]) -> Self {
        let mut mat = Self::new();
        for j in 0..COLS {
            mat.set_col(j, &cols[j]);
        }
        mat
    }

    /// Performs the matrix product: self (M x N) * other (N x P) -> Result (M x P).
    /// Dimensions are checked at compile-time!
    pub fn multiply<const OTHER_COLS: usize>(
        &self,
        other: &Matrix<COLS, OTHER_COLS>,
    ) -> Matrix<ROWS, OTHER_COLS> {
        let mut result = Matrix::<ROWS, OTHER_COLS>::new();
        result.matmul_accumulate(self, other);
        result
    }

    /// Accumulates the product of A * B into the current matrix (self).
    /// self = self + (A * B)
    /// All dimensions are checked at compile-time.
    pub fn matmul_accumulate<const K: usize>(&mut self, a: &Matrix<ROWS, K>, b: &Matrix<K, COLS>) {
        for i in 0..ROWS {
            for j in 0..COLS {
                let mut sum = 0.0;
                for k in 0..K {
                    sum += a.data[i][k] * b.data[k][j];
                }
                self.data[i][j] += sum;
            }
        }
    }

    /// Returns the transpose as a new static matrix.
    pub fn transpose(&self) -> Matrix<COLS, ROWS> {
        let mut result = Matrix::<COLS, ROWS>::new();
        for i in 0..ROWS {
            for j in 0..COLS {
                result[(j, i)] = self.data[i][j];
            }
        }
        result
    }

    /// Multiplies the matrix by a column vector: (ROWS x COLS) * (COLS) → (ROWS).
    pub fn mul_vec(&self, v: &Vector<COLS>) -> Vector<ROWS> {
        let mut result = [0.0; ROWS];
        for i in 0..ROWS {
            let mut sum = 0.0;
            for j in 0..COLS {
                sum += self.data[i][j] * v[j];
            }
            result[i] = sum;
        }
        Vector::new(result)
    }

    /// Scales the matrix by a coefficient.
    pub fn scale(&self, coef: Scalar) -> Self {
        let mut result = Self::new();
        for i in 0..ROWS {
            for j in 0..COLS {
                result[(i, j)] = self.data[i][j] * coef;
            }
        }
        result
    }
}

impl<const SIZE: usize> Matrix<SIZE, SIZE> {
    pub fn identity() -> Self {
        let mut result = Self::new();
        for i in 0..SIZE {
            result[(i, i)] = 1.0;
        }
        result
    }
}

// Operators for Static Matrices
impl<const ROWS: usize, const COLS: usize> Add<Matrix<ROWS, COLS>> for Matrix<ROWS, COLS> {
    type Output = Matrix<ROWS, COLS>;
    fn add(self, rhs: Matrix<ROWS, COLS>) -> Self::Output {
        let mut res = Matrix::new();
        for i in 0..ROWS {
            for j in 0..COLS {
                res[(i, j)] = self.data[i][j] + rhs.data[i][j];
            }
        }
        res
    }
}

impl<const ROWS: usize, const COLS: usize> Sub<Matrix<ROWS, COLS>> for Matrix<ROWS, COLS> {
    type Output = Matrix<ROWS, COLS>;
    fn sub(self, rhs: Matrix<ROWS, COLS>) -> Self::Output {
        let mut res = Matrix::new();
        for i in 0..ROWS {
            for j in 0..COLS {
                res[(i, j)] = self.data[i][j] - rhs.data[i][j];
            }
        }
        res
    }
}

// Global Mul operator for Matrix * Matrix
impl<const M: usize, const N: usize, const P: usize> Mul<Matrix<N, P>> for Matrix<M, N> {
    type Output = Matrix<M, P>;
    fn mul(self, rhs: Matrix<N, P>) -> Self::Output {
        self.multiply(&rhs)
    }
}

impl<const ROWS: usize, const COLS: usize> PartialEq for Matrix<ROWS, COLS> {
    fn eq(&self, other: &Self) -> bool {
        let epsilon = 1e-5;
        for i in 0..ROWS {
            for j in 0..COLS {
                if fabs(self.data[i][j] - other.data[i][j]) >= epsilon {
                    return false;
                }
            }
        }
        true
    }
}

/// Retrieves a value at (row, col).
impl<const ROWS: usize, const COLS: usize> Index<(usize, usize)> for Matrix<ROWS, COLS> {
    type Output = Scalar;
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.data[row][col]
    }
}

/// Sets a value at (row, col).
/// # Panics
/// Panics if indices are out of bounds.
impl<const ROWS: usize, const COLS: usize> IndexMut<(usize, usize)> for Matrix<ROWS, COLS> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        &mut self.data[row][col]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation_and_size() {
        let m = Matrix::<2, 3>::new();
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
    }

    #[test]
    fn test_matrix_get_set() {
        let mut m = Matrix::<2, 2>::new();
        m[(0, 1)] = 42.0;
        assert_eq!(m[(0, 1)], 42.0);
        assert_eq!(m[(1, 1)], 0.0);
    }

    #[test]
    #[should_panic]
    fn test_matrix_out_of_bounds() {
        let m = Matrix::<2, 2>::new();
        let _ = m[(2, 0)];
    }

    #[test]
    fn test_matrix_addition() {
        let mut m1 = Matrix::<2, 2>::new();
        m1[(0, 0)] = 1.0;
        let mut m2 = Matrix::<2, 2>::new();
        m2[(0, 0)] = 2.0;
        assert_eq!(m1 + m2, {
            let mut res = Matrix::<2, 2>::new();
            res[(0, 0)] = 3.0;
            res
        });
    }

    #[test]
    fn test_matrix_multiplication() {
        let mut m1 = Matrix::<2, 2>::new();
        m1[(0, 0)] = 1.0;
        m1[(0, 1)] = 2.0;
        m1[(1, 0)] = 3.0;
        m1[(1, 1)] = 4.0;

        let mut m2 = Matrix::<2, 1>::new();
        m2[(0, 0)] = 5.0;
        m2[(1, 0)] = 6.0;

        let res = m1 * m2;
        assert_eq!(res[(0, 0)], 17.0);
        assert_eq!(res[(1, 0)], 39.0);
    }

    #[test]
    fn test_matmul_accumulate() {
        let mut res = Matrix::<1, 1>::new();
        res[(0, 0)] = 10.0;
        let m1 = Matrix::<1, 1>::identity();
        let m2 = Matrix::<1, 1>::identity();
        res.matmul_accumulate(&m1, &m2);
        assert_eq!(res[(0, 0)], 11.0);
    }

    #[test]
    fn test_matrix_transpose() {
        let mut m = Matrix::<1, 2>::new();
        m[(0, 0)] = 1.0;
        m[(0, 1)] = 2.0;
        let t = m.transpose();
        assert_eq!(t.rows(), 2);
        assert_eq!(t.cols(), 1);
        assert_eq!(t[(1, 0)], 2.0);
    }

    #[test]
    fn test_matrix_col_extraction() {
        let mut m = Matrix::<2, 2>::new();
        m[(0, 1)] = 5.0;
        m[(1, 1)] = 10.0;
        let col = m.get_col(1).unwrap();
        assert_eq!(col, Vector::new([5.0, 10.0]));
    }

    #[test]
    fn test_matrix_from_cols() {
        let v1 = Vector::new([1.0, 2.0]);
        let v2 = Vector::new([3.0, 4.0]);
        let m = Matrix::from_cols([v1, v2]);
        assert_eq!(m[(1, 0)], 2.0);
        assert_eq!(m[(1, 1)], 4.0);
    }

    #[test]
    fn test_matrix_identity() {
        let id = Matrix::<3, 3>::identity();
        assert_eq!(id[(0, 0)], 1.0);
        assert_eq!(id[(0, 1)], 0.0);
        assert_eq!(id[(1, 1)], 1.0);
        assert_eq!(id[(2, 2)], 1.0);
    }
}
