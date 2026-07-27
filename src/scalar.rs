#[cfg(feature = "f64")]
pub type Scalar = f64;
#[cfg(not(feature = "f64"))]
pub type Scalar = f32;

#[cfg(feature = "f64")]
pub use libm::{exp, fabs, log, sqrt, tanh};
#[cfg(not(feature = "f64"))]
pub use libm::{expf as exp, fabsf as fabs, logf as log, sqrtf as sqrt, tanhf as tanh};
