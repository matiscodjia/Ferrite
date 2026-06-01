use crate::autodiff::flat_grads::FlatGrads;
use crate::autodiff::module::Module;
use crate::autodiff::perturb::Perturb;
use crate::autodiff::update::Update;
use crate::scalar::{exp, tanh, Scalar};
use crate::vector::Vector;

#[derive(Clone, Copy)]
pub struct ReLU<const N: usize> {}

impl<const N: usize> Module<Vector<N>> for ReLU<N> {
    type Output = Vector<N>;
    type Context = Vector<N>;
    type Gradients = ();

    fn forward(&self, x: Vector<N>) -> (Self::Output, Self::Context) {
        let mut result = [0.0; N];
        for i in 0..N {
            if x[i] > 0.0 {
                result[i] = x[i];
            }
        }
        (Vector::new(result), x)
    }

    fn backward(
        &self,
        grad_out: Self::Output,
        ctx: &Self::Context,
    ) -> (Vector<N>, Self::Gradients) {
        let x = ctx;
        let mut result = [0.0; N];
        for i in 0..N {
            if x[i] > 0.0 {
                result[i] = grad_out[i];
            }
        }
        (Vector::new(result), ())
    }
}
impl<const N: usize> Update for ReLU<N> {
    type Gradients = ();
    fn update(&mut self, _grads: &Self::Gradients, _lr: Scalar) {}
}
impl<const N: usize> Perturb for ReLU<N> {
    fn num_params(&self) -> usize { 0 }
    fn perturb(&mut self, _idx: usize, _delta: Scalar) {}
}
impl<const N: usize> FlatGrads for ReLU<N> {
    type Gradients = ();
    fn write_grads(_grads: &Self::Gradients, _buf: &mut [Scalar], _offset: &mut usize) {}
}

#[derive(Clone, Copy)]
pub struct LeakyReLU<const N: usize> {
    pub alpha: Scalar,
}

impl<const N: usize> LeakyReLU<N> {
    pub fn new(alpha: Scalar) -> Self {
        Self { alpha }
    }
}
impl<const N: usize> Update for LeakyReLU<N> {
    type Gradients = ();
    fn update(&mut self, _grads: &Self::Gradients, _lr: Scalar) {}
}
impl<const N: usize> Perturb for LeakyReLU<N> {
    fn num_params(&self) -> usize { 0 }
    fn perturb(&mut self, _idx: usize, _delta: Scalar) {}
}
impl<const N: usize> FlatGrads for LeakyReLU<N> {
    type Gradients = ();
    fn write_grads(_grads: &Self::Gradients, _buf: &mut [Scalar], _offset: &mut usize) {}
}
impl<const N: usize> Module<Vector<N>> for LeakyReLU<N> {
    type Output = Vector<N>;
    type Context = Vector<N>;
    type Gradients = ();

    fn forward(&self, x: Vector<N>) -> (Self::Output, Self::Context) {
        let mut result = [0.0; N];
        for i in 0..N {
            if x[i] > 0.0 {
                result[i] = x[i];
            } else {
                result[i] = x[i] * self.alpha;
            }
        }
        (Vector::new(result), x)
    }

    fn backward(
        &self,
        grad_out: Self::Output,
        ctx: &Self::Context,
    ) -> (Vector<N>, Self::Gradients) {
        let x = ctx;
        let mut result = [0.0; N];
        for i in 0..N {
            if x[i] > 0.0 {
                result[i] = grad_out[i];
            } else {
                result[i] = grad_out[i] * self.alpha;
            }
        }
        (Vector::new(result), ())
    }
}

#[derive(Clone, Copy)]
pub struct Sigmoid<const N: usize> {}

impl<const N: usize> Module<Vector<N>> for Sigmoid<N> {
    type Output = Vector<N>;
    type Context = Vector<N>;
    type Gradients = ();

    fn forward(&self, x: Vector<N>) -> (Self::Output, Self::Context) {
        let mut result = [0.0; N];
        for i in 0..N {
            result[i] = 1.0 / (1.0 + exp(-x[i]));
        }
        let output = Vector::new(result);
        (output, output)
    }

    fn backward(
        &self,
        grad_out: Self::Output,
        ctx: &Self::Context,
    ) -> (Vector<N>, Self::Gradients) {
        let sigmoid = ctx;
        let mut result = [0.0; N];
        for i in 0..N {
            result[i] = grad_out[i] * sigmoid[i] * (1.0 - sigmoid[i]);
        }
        (Vector::new(result), ())
    }
}

impl<const N: usize> Update for Sigmoid<N> {
    type Gradients = ();
    fn update(&mut self, _grads: &Self::Gradients, _lr: Scalar) {}
}
impl<const N: usize> Perturb for Sigmoid<N> {
    fn num_params(&self) -> usize { 0 }
    fn perturb(&mut self, _idx: usize, _delta: Scalar) {}
}
impl<const N: usize> FlatGrads for Sigmoid<N> {
    type Gradients = ();
    fn write_grads(_grads: &Self::Gradients, _buf: &mut [Scalar], _offset: &mut usize) {}
}

#[derive(Clone, Copy)]
pub struct Tanh<const N: usize> {}

impl<const N: usize> Module<Vector<N>> for Tanh<N> {
    type Output = Vector<N>;
    type Context = Vector<N>;
    type Gradients = ();

    fn forward(&self, x: Vector<N>) -> (Self::Output, Self::Context) {
        let mut result = [0.0; N];
        for i in 0..N {
            result[i] = tanh(x[i]);
        }
        let output = Vector::new(result);
        (output, output)
    }

    fn backward(
        &self,
        grad_out: Self::Output,
        ctx: &Self::Context,
    ) -> (Vector<N>, Self::Gradients) {
        let t = ctx;
        let mut result = [0.0; N];
        for i in 0..N {
            result[i] = grad_out[i] * (1.0 - t[i] * t[i]);
        }
        (Vector::new(result), ())
    }
}

impl<const N: usize> Update for Tanh<N> {
    type Gradients = ();
    fn update(&mut self, _grads: &Self::Gradients, _lr: Scalar) {}
}
impl<const N: usize> Perturb for Tanh<N> {
    fn num_params(&self) -> usize { 0 }
    fn perturb(&mut self, _idx: usize, _delta: Scalar) {}
}
impl<const N: usize> FlatGrads for Tanh<N> {
    type Gradients = ();
    fn write_grads(_grads: &Self::Gradients, _buf: &mut [Scalar], _offset: &mut usize) {}
}

#[derive(Clone, Copy)]
pub struct Softmax<const N: usize> {}

impl<const N: usize> Module<Vector<N>> for Softmax<N> {
    type Output = Vector<N>;
    type Context = Vector<N>;
    type Gradients = ();

    fn forward(&self, x: Vector<N>) -> (Self::Output, Self::Context) {
        let mut max = x[0];
        for i in 1..N {
            if x[i] > max {
                max = x[i];
            }
        }
        let mut data = [0.0; N];
        let mut sum = 0.0;
        for i in 0..N {
            data[i] = exp(x[i] - max);
            sum += data[i];
        }
        for i in 0..N {
            data[i] /= sum;
        }
        let output = Vector::new(data);
        (output, output)
    }

    fn backward(
        &self,
        grad_out: Self::Output,
        ctx: &Self::Context,
    ) -> (Vector<N>, Self::Gradients) {
        let s = ctx;
        let dot = grad_out.dot(s);
        let mut result = [0.0; N];
        for i in 0..N {
            result[i] = s[i] * (grad_out[i] - dot);
        }
        (Vector::new(result), ())
    }
}

impl<const N: usize> Update for Softmax<N> {
    type Gradients = ();
    fn update(&mut self, _grads: &Self::Gradients, _lr: Scalar) {}
}
impl<const N: usize> Perturb for Softmax<N> {
    fn num_params(&self) -> usize { 0 }
    fn perturb(&mut self, _idx: usize, _delta: Scalar) {}
}
impl<const N: usize> FlatGrads for Softmax<N> {
    type Gradients = ();
    fn write_grads(_grads: &Self::Gradients, _buf: &mut [Scalar], _offset: &mut usize) {}
}
