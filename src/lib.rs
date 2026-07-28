#![no_std] // We are officially an embedded library now.

#[cfg(any(feature = "std", test))]
#[macro_use]
extern crate std; // Use full std for tests and dev on MacOS.

pub mod algorithms;
pub mod autodiff;
pub mod matrix;
pub mod scalar;
pub mod sp;
pub mod tensor;
pub mod vector;
pub use algorithms::{gram_schmidt, qr_decomposition, solve_linear_system};
pub use matrix::Matrix;
pub use scalar::Scalar;
pub use sp::conv2d;
/// Re-export main types for simplified usage
pub use vector::Vector;
