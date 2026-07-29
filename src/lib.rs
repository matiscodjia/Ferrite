#![no_std] // We are officially an embedded library now.

#[cfg(any(feature = "std", test))]
#[macro_use]
extern crate std; // Use full std for tests and dev on MacOS.

pub mod autodiff;
pub mod linalg;
pub mod scalar;
pub mod sp;

/// Re-export main types for simplified usage
pub use linalg::{gram_schmidt, qr_decomposition, solve_linear_system};
pub use linalg::{Matrix, Tensor, Vector};
pub use scalar::Scalar;
pub use sp::cross_correlate2d;
