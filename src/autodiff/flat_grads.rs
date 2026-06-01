use crate::autodiff::sequential::Then;
use crate::scalar::Scalar;

pub trait FlatGrads {
    type Gradients;
    fn write_grads(grads: &Self::Gradients, buf: &mut [Scalar], offset: &mut usize);
}

impl<A: FlatGrads, B: FlatGrads> FlatGrads for Then<A, B> {
    type Gradients = (A::Gradients, B::Gradients);
    fn write_grads(grads: &Self::Gradients, buf: &mut [Scalar], offset: &mut usize) {
        A::write_grads(&grads.0, buf, offset);
        B::write_grads(&grads.1, buf, offset);
    }
}
