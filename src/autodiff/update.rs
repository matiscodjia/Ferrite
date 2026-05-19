use crate::autodiff::sequential::Then;

pub trait Update {
    type Gradients;
    fn update(&mut self, grads: &Self::Gradients, lr: f32);
}

impl<A, B> Update for Then<A, B>
where
    A: Update,
    B: Update,
{
    type Gradients = (A::Gradients, B::Gradients);
    fn update(&mut self, grads: &Self::Gradients, lr: f32) {
        self.first.update(&grads.0, lr);
        self.second.update(&grads.1, lr);
    }
}

