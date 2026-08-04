//! Compile-time-shaped, stride-based tensors, one file per rank.
//!
//! [`tensor2d`] and [`tensor3d`] hold the plain owned tensors and the 2D
//! zero-copy view. [`tensor4d`] adds pluggable storage (stack or heap) and
//! the `im2col_view` that turns a 4D tensor into a 6D receptive-field view
//! without copying. [`tensor6d`] holds the `Rank6` trait (shared by owned
//! `Tensor6D` and the `im2col` `TensorView6D`) and both rank-6 types.
//! [`contraction`] holds the `tensordot_*` family that contracts across
//! them.

mod contraction;
mod tensor2d;
mod tensor3d;
mod tensor4d;
mod tensor6d;

pub use tensor4d::Tensor4D;
#[cfg(feature = "alloc")]
pub use tensor4d::Tensor4DBoxed;

pub use contraction::{tensordot_1, tensordot_2, tensordot_3};
pub use tensor2d::{Tensor, TensorView};
pub use tensor3d::Tensor3D;
pub use tensor6d::{Rank6, Tensor6D, TensorView6D};

// `Vector` isn't re-exported from `linalg::mod.rs` yet — `linalg::vector::Vector`
// (the old type, being phased out) still claims that name at the top level.
// Callers within the crate that want the new alias use this path directly;
// `linalg/mod.rs` switches its top-level `Vector` re-export over once the old
// type is deleted.
pub use tensor2d::Vector;
