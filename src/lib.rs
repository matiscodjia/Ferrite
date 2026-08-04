#![no_std] // We are officially an embedded library now.

// Sonde : ne sert qu'aux stockages tas (gros tenseurs de benchmark).
#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(any(feature = "std", test))]
#[macro_use]
extern crate std; // Use full std for tests and dev on MacOS.

pub mod autodiff;
/// Sonde hôte : lecture/écriture .npy et pilotage des benchmarks. Rien ici n'est
/// destiné à une cible embarquée, d'où le gate sur `std`.
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
