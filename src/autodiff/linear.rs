use crate::autodiff::flat_grads::FlatGrads;
use crate::autodiff::module::Module;
use crate::autodiff::params::Params;
use crate::autodiff::perturb::Perturb;
use crate::autodiff::update::Update;
use crate::matrix::Matrix;
use crate::scalar::{sqrt, Scalar};
use crate::vector::Vector;

#[derive(Clone, Copy)]
pub struct Linear<const IN: usize, const OUT: usize> {
    weights: Matrix<OUT, IN>,
    bias: Vector<OUT>,
}

impl<const IN: usize, const OUT: usize> Linear<IN, OUT> {
    pub fn zeros() -> Self {
        Linear {
            weights: Matrix::new(),
            bias: Vector::new([0.0; OUT]),
        }
    }

    pub fn from_weights(weights: Matrix<OUT, IN>, bias: Vector<OUT>) -> Self {
        Linear { weights, bias }
    }

    pub fn from_seed(seed: u64) -> Self {
        let mut state = if seed == 0 { 1 } else { seed };
        let limit = sqrt(6.0 / (IN + OUT) as Scalar);
        let mut weights = Matrix::<OUT, IN>::new();
        for i in 0..OUT {
            for j in 0..IN {
                weights[(i, j)] = xorshift_scalar(&mut state) * limit;
            }
        }
        Linear {
            weights,
            bias: Vector::new([0.0; OUT]),
        }
    }
}

fn xorshift_scalar(state: &mut u64) -> Scalar {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    (x as Scalar) / (u64::MAX as Scalar) * 2.0 - 1.0
}

#[derive(Clone, Copy)]
pub struct LinearGrads<const IN: usize, const OUT: usize> {
    weights_grads: Matrix<OUT, IN>,
    bias_grad: Vector<OUT>,
}

impl<const IN: usize, const OUT: usize> Params for Linear<IN, OUT> {
    type Gradients = LinearGrads<IN, OUT>;
}

impl<const IN: usize, const OUT: usize> Module<Vector<IN>> for Linear<IN, OUT> {
    type Output = Vector<OUT>;
    type Context = Vector<IN>;

    fn forward(&self, x: Vector<IN>) -> (Self::Output, Self::Context) {
        let result = self.weights.mul_vec(&x) + self.bias;
        (result, x)
    }

    fn backward(
        &self,
        grad_out: Self::Output,
        ctx: &Self::Context,
    ) -> (Vector<IN>, Self::Gradients) {
        let x = ctx;
        let data_grad = self.weights.transpose().mul_vec(&grad_out);
        let mut weights_grads = Matrix::<OUT, IN>::new();
        for i in 0..OUT {
            for j in 0..IN {
                weights_grads[(i, j)] = grad_out[i] * x[j];
            }
        }
        (
            data_grad,
            LinearGrads {
                weights_grads,
                bias_grad: grad_out,
            },
        )
    }
}

impl<const IN: usize, const OUT: usize> FlatGrads for Linear<IN, OUT> {
    fn write_grads(grads: &Self::Gradients, buf: &mut [Scalar], offset: &mut usize) {
        for row in 0..OUT {
            for col in 0..IN {
                buf[*offset] = grads.weights_grads[(row, col)];
                *offset += 1;
            }
        }
        for i in 0..OUT {
            buf[*offset] = grads.bias_grad[i];
            *offset += 1;
        }
    }
}

impl<const IN: usize, const OUT: usize> Perturb for Linear<IN, OUT> {
    fn num_params(&self) -> usize {
        IN * OUT + OUT
    }

    fn perturb(&mut self, idx: usize, delta: Scalar) {
        if idx < IN * OUT {
            let row = idx / IN;
            let col = idx % IN;
            self.weights[(row, col)] += delta;
        } else {
            self.bias[idx - IN * OUT] += delta;
        }
    }
}

impl<const IN: usize, const OUT: usize> Update for Linear<IN, OUT> {
    fn update(&mut self, grads: &Self::Gradients, lr: Scalar) {
        self.weights = self.weights - grads.weights_grads.scale(lr);
        self.bias = self.bias - grads.bias_grad * lr;
    }
}
