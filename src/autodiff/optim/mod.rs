//! Optimizers and the training step that drives them.

pub mod optimizer;
pub mod sgd;
pub mod train;

pub use optimizer::Optimizer;
pub use sgd::{sgd, Sgd};
pub use train::train_step;
