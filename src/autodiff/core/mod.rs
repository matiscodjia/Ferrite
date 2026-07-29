//! The traits every layer implements, and the combinator that chains them.
//!
//! A network is a type: `Then<A, B>` composes two modules, and each trait here
//! lifts an operation from the leaves to the whole tree — forward/backward
//! ([`Module`]), gradient shape ([`Params`]), in-place descent ([`Update`]),
//! and the flat views ([`FlatGrads`], [`Perturb`]) that gradient checking needs.

pub mod flat_grads;
pub mod module;
pub mod params;
pub mod perturb;
pub mod sequential;
pub mod update;

pub use flat_grads::FlatGrads;
pub use module::Module;
pub use params::Params;
pub use perturb::Perturb;
pub use sequential::Then;
pub use update::Update;
