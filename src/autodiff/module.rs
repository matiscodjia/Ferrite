use crate::autodiff::sequential::Then;

pub trait Module<Input>: Sized {
    type Output;
    type Context;
    type Gradients;
    fn then<Next>(self, next: Next) -> Then<Self, Next>
    where
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
