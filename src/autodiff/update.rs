use crate::autodiff::sequential::Then;
use crate::scalar::Scalar;

pub trait Update {
    type Gradients;
    fn update(&mut self, grads: &Self::Gradients, lr: Scalar);
}

impl<A, B> Update for Then<A, B>
where
    A: Update,
    B: Update,
{
    type Gradients = (A::Gradients, B::Gradients);
    fn update(&mut self, grads: &Self::Gradients, lr: Scalar) {
        self.first.update(&grads.0, lr);
        self.second.update(&grads.1, lr);
    }
}
