use crate::autodiff::core::sequential::Then;
use crate::scalar::Scalar;

pub trait Perturb {
    fn num_params(&self) -> usize;
    fn perturb(&mut self, idx: usize, delta: Scalar);
}

impl<A: Perturb, B: Perturb> Perturb for Then<A, B> {
    fn num_params(&self) -> usize {
        self.first.num_params() + self.second.num_params()
    }

    fn perturb(&mut self, idx: usize, delta: Scalar) {
        let n = self.first.num_params();
        if idx < n {
            self.first.perturb(idx, delta);
        } else {
            self.second.perturb(idx - n, delta);
        }
    }
}
