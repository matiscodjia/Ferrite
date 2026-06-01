use crate::autodiff::optims::optimizer::Optimizer;
use crate::autodiff::update::Update;
use crate::scalar::Scalar;

pub struct Sgd {
    pub lr: Scalar,
}

impl Sgd {
    pub fn new(lr: Scalar) -> Self {
        Self { lr }
    }
}

impl<Net: Update> Optimizer<Net> for Sgd {
    fn step(&mut self, net: &mut Net, grads: &Net::Gradients) {
        net.update(grads, self.lr);
    }
}

pub fn sgd<N: Update>(network: &mut N, grads: &N::Gradients, lr: Scalar) {
    network.update(grads, lr);
}
