//! Compile-time-shaped linear algebra.
//!
//! Every shape is a const generic and every buffer lives on the stack, so a
//! `no_std` target pays no allocation for an intermediate result.

pub mod decomposition;
pub mod matrix;
pub mod tensor;
pub mod vector;

pub use decomposition::{
    gram_schmidt, jacobi_rotation, qr_decomposition, solve_linear_system, solve_upper_triangular,
    svd, svd_2x2,
};
pub use matrix::Matrix;
pub use tensor::{
    tensordot_1, tensordot_2, tensordot_3, Rank6, Tensor, Tensor3D, Tensor4D, Tensor6D, TensorView,
    TensorView6D,
};
pub use vector::Vector;
