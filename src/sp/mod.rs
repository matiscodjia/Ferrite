//! Signal processing toolkit.
//!
//! Operators that read a signal — a video sequence, a stack of feature maps —
//! through a bank of filters. Everything is shaped at compile time and lives on
//! the stack, so a `no_std` target pays no allocation for the intermediate
//! layouts.

pub mod conv_streaming;
pub mod correlate;
pub mod kernels;
mod shape;
pub use conv_streaming::ConvStreaming;
pub use correlate::cross_correlate2d;
pub use kernels::{filter_bank, Gaussian3D, Sobel3D};
