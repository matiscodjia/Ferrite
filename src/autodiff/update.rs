use crate::autodiff::params::Params;
use crate::autodiff::sequential::Then;
use crate::scalar::Scalar;

pub trait Update: Params {
    fn update(&mut self, grads: &Self::Gradients, lr: Scalar);
}

impl<A, B> Update for Then<A, B>
where
    A: Update,
    B: Update,
{
    fn update(&mut self, grads: &Self::Gradients, lr: Scalar) {
        self.first.update(&grads.0, lr);
        self.second.update(&grads.1, lr);
    }
}
