use crate::scalar::{fabs, sqrt, Scalar};
use core::ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub};

/// A Static Vector of size N, stored entirely on the stack.
/// No heap allocation, 100% no_std compatible.
#[derive(Debug, Clone, Copy)]
pub struct Vector<const N: usize> {
    data: [Scalar; N],
}

impl<const N: usize> Vector<N> {
    /// Creates a new vector from a static array.
    pub fn new(data: [Scalar; N]) -> Self {
        Self { data }
    }

    /// Returns the dimension of the vector.
    pub const fn dim(&self) -> usize {
        N
    }

    /// Calculates the infinity norm (max of absolute values).
    pub fn inf_norm(&self) -> Scalar {
        let mut max = 0.0;
        for &val in &self.data {
            let abs_val = fabs(val);
            if abs_val > max {
                max = abs_val;
            }
        }
        max
    }

    /// Calculates the L2 norm (Euclidean norm).
    pub fn l2_norm(&self) -> Scalar {
        sqrt(self.dot(self))
    }

    /// Calculates the L1 norm (sum of absolute values).
    pub fn l1_norm(&self) -> Scalar {
        let mut sum = 0.0;
        for &val in &self.data {
            sum += fabs(val);
        }
        sum
    }

    /// Calculates the dot product between two vectors of the same size N.
    /// The size is checked at compile-time.
    pub fn dot(&self, other: &Vector<N>) -> Scalar {
        let mut sum = 0.0;
        for i in 0..N {
            sum += self.data[i] * other.data[i];
        }
        sum
    }

    /// Calculates the orthogonal projection of `self` onto `other`.
    pub fn orthogonal_projection(&self, other: &Vector<N>) -> Vector<N> {
        let scale_factor = other.dot(other);
        if fabs(scale_factor) < 1e-8 {
            return Vector::new([0.0; N]);
        }
        let ratio = self.dot(other) / scale_factor;
        other * ratio
    }

    /// Access the raw data array.
    pub fn get_data(&self) -> &[Scalar; N] {
        &self.data
    }

    /// Sum of all elements.
    pub fn sum(&self) -> Scalar {
        let mut s = 0.0;
        for &v in &self.data {
            s += v;
        }
        s
    }

    /// Element-wise multiply (Hadamard product).
    pub fn hadamard(&self, other: &Vector<N>) -> Vector<N> {
        let mut data = [0.0; N];
        for i in 0..N {
            data[i] = self.data[i] * other.data[i];
        }
        Vector::new(data)
    }
}

// Operator implementations for Static Vectors
impl<const N: usize> Mul<Scalar> for &Vector<N> {
    type Output = Vector<N>;
    fn mul(self, rhs: Scalar) -> Self::Output {
        let mut data = [0.0; N];
        for i in 0..N {
            data[i] = self.data[i] * rhs;
        }
        Vector::new(data)
    }
}

impl<const N: usize> Sub<&Vector<N>> for &Vector<N> {
    type Output = Vector<N>;
    fn sub(self, rhs: &Vector<N>) -> Self::Output {
        let mut data = [0.0; N];
        for i in 0..N {
            data[i] = self.data[i] - rhs.data[i];
        }
        Vector::new(data)
    }
}

impl<const N: usize> Add<&Vector<N>> for &Vector<N> {
    type Output = Vector<N>;
    fn add(self, rhs: &Vector<N>) -> Self::Output {
        let mut data = [0.0; N];
        for i in 0..N {
            data[i] = self.data[i] + rhs.data[i];
        }
        Vector::new(data)
    }
}

impl<const N: usize> Mul<Scalar> for Vector<N> {
    type Output = Vector<N>;
    fn mul(self, rhs: Scalar) -> Self::Output {
        &self * rhs
    }
}

impl<const N: usize> Add<Vector<N>> for Vector<N> {
    type Output = Vector<N>;
    fn add(self, rhs: Vector<N>) -> Self::Output {
        &self + &rhs
    }
}

impl<const N: usize> Sub<Vector<N>> for Vector<N> {
    type Output = Vector<N>;
    fn sub(self, rhs: Vector<N>) -> Self::Output {
        &self - &rhs
    }
}

impl<const N: usize> Neg for Vector<N> {
    type Output = Vector<N>;
    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

impl<const N: usize> Div<Scalar> for Vector<N> {
    type Output = Vector<N>;
    fn div(self, rhs: Scalar) -> Self::Output {
        self * (1.0 / rhs)
    }
}

impl<const N: usize> PartialEq for Vector<N> {
    fn eq(&self, other: &Self) -> bool {
        let epsilon = 1e-6;
        for i in 0..N {
            if fabs(self.data[i] - other.data[i]) >= epsilon {
                return false;
            }
        }
        true
    }
}
impl<const N: usize> Index<usize> for Vector<N> {
    type Output = Scalar;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<const N: usize> IndexMut<usize> for Vector<N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_creation_and_dim() {
        let v = Vector::new([1.0, 2.0, 3.0]);
        assert_eq!(v.dim(), 3);
    }

    #[test]
    fn test_vector_l1_norm() {
        let v = Vector::new([1.0, -2.0, 3.0]);
        assert_eq!(v.l1_norm(), 6.0);
    }

    #[test]
    fn test_vector_l2_norm() {
        let v = Vector::new([3.0, 4.0]);
        assert_eq!(v.l2_norm(), 5.0);
    }

    #[test]
    fn test_vector_inf_norm() {
        let v = Vector::new([-10.0, 2.0, 5.0]);
        assert_eq!(v.inf_norm(), 10.0);
    }

    #[test]
    fn test_vector_dot_product() {
        let v1 = Vector::new([1.0, 2.0]);
        let v2 = Vector::new([3.0, 4.0]);
        assert_eq!(v1.dot(&v2), 11.0);
    }

    #[test]
    fn test_vector_addition() {
        let v1 = Vector::new([1.0, 2.0]);
        let v2 = Vector::new([3.0, 4.0]);
        assert_eq!(&v1 + &v2, Vector::new([4.0, 6.0]));
    }

    #[test]
    fn test_vector_subtraction() {
        let v1 = Vector::new([5.0, 7.0]);
        let v2 = Vector::new([2.0, 3.0]);
        assert_eq!(&v1 - &v2, Vector::new([3.0, 4.0]));
    }

    #[test]
    fn test_vector_scalar_mul() {
        let v = Vector::new([1.0, -2.0]);
        assert_eq!(&v * 3.0, Vector::new([3.0, -6.0]));
    }

    #[test]
    fn test_vector_projection() {
        let v = Vector::new([1.0, 1.0]);
        let target = Vector::new([1.0, 0.0]);
        assert_eq!(v.orthogonal_projection(&target), Vector::new([1.0, 0.0]));
    }

    #[test]
    fn test_vector_null_projection() {
        let v = Vector::new([1.0, 2.0]);
        let null_v = Vector::<2>::new([0.0, 0.0]);
        assert_eq!(v.orthogonal_projection(&null_v), Vector::new([0.0, 0.0]));
    }
}
