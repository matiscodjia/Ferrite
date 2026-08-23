#![no_std] // We are officially an embedded library now.

// Probe: only used for heap-backed storage (large benchmark tensors).
#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(any(feature = "std", test))]
#[macro_use]
extern crate std; // Use full std for tests and dev on MacOS.

pub mod autodiff;
/// Host-only probe: .npy reading/writing and benchmark driving. Nothing
/// here targets an embedded device, hence the `std` gate.
#[cfg(feature = "std")]
pub mod io;
pub mod linalg;
pub mod scalar;
pub mod sp;
/// Re-export main types for simplified usage
pub use linalg::{gram_schmidt, qr_decomposition, solve_linear_system};
pub use linalg::{Tensor, Vector};
pub use scalar::Scalar;
pub use sp::cross_correlate2d;
