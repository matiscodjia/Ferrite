//! Building blocks: the one parametric layer, and the stateless nonlinearities.

pub mod activations;
pub mod linear;

pub use activations::{LeakyReLU, ReLU, Sigmoid, Softmax, Tanh};
pub use linear::{Linear, LinearGrads};
