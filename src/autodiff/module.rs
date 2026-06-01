use crate::autodiff::params::Params;
use crate::autodiff::sequential::Then;

pub trait Module<Input>: Params {
    type Output;
    type Context;

    fn then<Next>(self, next: Next) -> Then<Self, Next>
    where
        Self: Sized,
        Next: Module<Self::Output>,
    {
        Then {
            first: self,
            second: next,
        }
    }

    fn forward(&self, x: Input) -> (Self::Output, Self::Context);
    fn backward(&self, grad_out: Self::Output, ctx: &Self::Context) -> (Input, Self::Gradients);
}
