use crate::autodiff::module::Module;
use crate::autodiff::params::Params;

#[derive(Clone)]
pub struct Then<A, B> {
    pub(super) first: A,
    pub(super) second: B,
}

impl<A, B> Then<A, B> {
    pub fn new(first: A, second: B) -> Self {
        Self { first, second }
    }
}

impl<A: Params, B: Params> Params for Then<A, B> {
    type Gradients = (A::Gradients, B::Gradients);
}

impl<Input, A, B> Module<Input> for Then<A, B>
where
    A: Module<Input>,
    B: Module<A::Output>,
{
    type Output = B::Output;
    type Context = (A::Context, B::Context);

    fn then<Next>(self, next: Next) -> Then<Self, Next>
    where
        Next: Module<Self::Output>,
    {
        Then {
            first: self,
            second: next,
        }
    }

    fn forward(&self, x: Input) -> (Self::Output, Self::Context) {
        let (out1, ctx1) = self.first.forward(x);
        let (out2, ctx2) = self.second.forward(out1);
        (out2, (ctx1, ctx2))
    }

    fn backward(&self, grad_out: Self::Output, ctx: &Self::Context) -> (Input, Self::Gradients) {
        let (ctx1, ctx2) = ctx;
        let (grad1, grads2) = self.second.backward(grad_out, ctx2);
        let (grad0, grads1) = self.first.backward(grad1, ctx1);
        (grad0, (grads1, grads2))
    }
}

#[macro_export]
macro_rules! seq {
    ($only:expr) => {
        $only
    };
    ($first:expr, $($rest:expr),+) => {
        $crate::autodiff::sequential::Then::new($first, $crate::seq!($($rest),+))
    };
}
