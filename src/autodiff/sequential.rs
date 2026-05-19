use crate::autodiff::module::Module;

pub struct Then<A, B> {
    pub(super) first: A,
    pub(super) second: B,
}
impl<A, B> Then<A, B> {
    pub fn new(first: A, second: B) -> Self {
        Self { first, second }
    }
}

impl<Input, A, B> Module<Input> for Then<A, B>
where
    A: Module<Input>,
    B: Module<A::Output>,
{
    type Output = B::Output;
    type Context = (A::Context, B::Context);
    type Gradients = (A::Gradients, B::Gradients);

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

#[cfg(test)]
mod tests {
    use crate::autodiff::activations::{ReLU, Tanh};
    use crate::autodiff::linear::Linear;
    use crate::autodiff::loss::mse;
    use crate::autodiff::module::Module;
    use crate::autodiff::optims::sgd::sgd;
    use crate::Vector;

    #[test]
    fn seq_2_layers_forward_backward() {
        let net = seq!(Linear::<2, 4>::from_seed(42), ReLU::<4> {});
        let input = Vector::new([1.0, 0.5]);
        let (output, ctx) = net.forward(input);
        let _ = net.backward(output, &ctx);
    }

    #[test]
    fn seq_3_layers_loss() {
        let net = seq!(
            Linear::<2, 4>::from_seed(42),
            Tanh::<4> {},
            Linear::<4, 1>::from_seed(99)
        );
        let input = Vector::new([1.0, 0.5]);
        let target = Vector::new([1.0]);
        let (output, ctx) = net.forward(input);
        let (_, loss_grad) = mse(output, target);
        let _ = net.backward(loss_grad, &ctx);
    }

    #[test]
    fn seq_training_step_decreases_loss() {
        let mut net = seq!(
            Linear::<2, 8>::from_seed(42),
            Tanh::<8> {},
            Linear::<8, 1>::from_seed(137)
        );
        let input = Vector::new([1.0, 0.5]);
        let target = Vector::new([1.0]);

        let (output, ctx) = net.forward(input);
        let (loss, loss_grad) = mse(output, target);
        let (_, grads) = net.backward(loss_grad, &ctx);
        sgd(&mut net, &grads, 0.01);

        let (output2, _) = net.forward(input);
        let (loss2, _) = mse(output2, target);
        assert!(
            loss2 < loss,
            "loss should decrease after one SGD step: {loss2} >= {loss}"
        );
    }

    #[test]
    fn seq_single_layer() {
        let net = seq!(Linear::<3, 2>::from_seed(7));
        let input = Vector::new([1.0, 0.0, -1.0]);
        let (output, ctx) = net.forward(input);
        let _ = net.backward(output, &ctx);
    }
}

