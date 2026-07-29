use crate::autodiff::core::params::Params;
use crate::autodiff::core::sequential::Then;
use crate::scalar::Scalar;

pub trait FlatGrads: Params {
    fn write_grads(grads: &Self::Gradients, buf: &mut [Scalar], offset: &mut usize);
}

impl<A: FlatGrads, B: FlatGrads> FlatGrads for Then<A, B> {
    fn write_grads(grads: &Self::Gradients, buf: &mut [Scalar], offset: &mut usize) {
        A::write_grads(&grads.0, buf, offset);
        B::write_grads(&grads.1, buf, offset);
    }
}
