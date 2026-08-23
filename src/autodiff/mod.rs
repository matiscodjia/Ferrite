//! Static reverse-mode autodiff.
//!
//! The computation graph is the type of the network, resolved at compile time:
//! there is no tape, no allocation, and no dynamic dispatch. [`core`] holds the
//! traits and the `Then` combinator, [`layers`] the building blocks, [`optim`]
//! the descent, and [`grad_check`] the finite-difference cross-check.

pub mod core;
pub mod grad_check;
pub mod layers;
pub mod loss;
pub mod optim;

pub use self::core::{FlatGrads, Module, Params, Perturb, Then, Update};
pub use grad_check::GradChecker;
pub use layers::{LeakyReLU, Linear, ReLU, Sigmoid, Softmax, Tanh};
pub use loss::{cross_entropy, mae, mse};
pub use optim::{sgd, train_step, Optimizer, Sgd};
