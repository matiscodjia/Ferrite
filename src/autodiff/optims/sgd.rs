use crate::autodiff::update::Update;
use crate::scalar::Scalar;

pub fn sgd<N: Update>(network: &mut N, grads: &N::Gradients, lr: Scalar) {
    network.update(grads, lr);
}
