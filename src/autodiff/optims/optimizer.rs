use crate::autodiff::params::Params;

pub trait Optimizer<Net: Params> {
    fn step(&mut self, net: &mut Net, grads: &Net::Gradients);
}
