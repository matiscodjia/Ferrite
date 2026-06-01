use crate::autodiff::module::Module;
use crate::autodiff::optims::optimizer::Optimizer;
use crate::scalar::Scalar;

pub fn train_step<Net, Input, Opt>(
    net: &mut Net,
    optimizer: &mut Opt,
    input: Input,
    target: <Net as Module<Input>>::Output,
    loss_fn: fn(
        <Net as Module<Input>>::Output,
        <Net as Module<Input>>::Output,
    ) -> (Scalar, <Net as Module<Input>>::Output),
) -> Scalar
where
    Net: Module<Input>,
    Opt: Optimizer<Net>,
    <Net as Module<Input>>::Output: Copy,
    Input: Copy,
{
    let (output, ctx) = net.forward(input);
    let (loss, loss_grad) = loss_fn(output, target);
    let (_, grads) = net.backward(loss_grad, &ctx);
    optimizer.step(net, &grads);
    loss
}
