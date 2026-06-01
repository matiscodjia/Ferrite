use crate::scalar::{log, Scalar};
use crate::vector::Vector;

pub fn mse<const N: usize>(output: Vector<N>, target: Vector<N>) -> (Scalar, Vector<N>) {
    let mut loss = 0.0;
    let mut grad_data = [0.0; N];
    for i in 0..N {
        let diff = output[i] - target[i];
        loss += diff * diff;
        grad_data[i] = 2.0 * diff / N as Scalar;
    }
    (loss / N as Scalar, Vector::new(grad_data))
}

pub fn mae<const N: usize>(output: Vector<N>, target: Vector<N>) -> (Scalar, Vector<N>) {
    let mut loss = 0.0;
    let mut grad_data = [0.0; N];
    for i in 0..N {
        let diff = output[i] - target[i];
        loss += if diff >= 0.0 { diff } else { -diff };
        grad_data[i] = diff.signum() / N as Scalar;
    }
    (loss / N as Scalar, Vector::new(grad_data))
}

pub fn cross_entropy<const N: usize>(probs: Vector<N>, target: Vector<N>) -> (Scalar, Vector<N>) {
    let epsilon = 1e-7;
    let mut loss = 0.0;
    let mut grad_data = [0.0; N];
    for i in 0..N {
        let p = if probs[i] < epsilon { epsilon } else { probs[i] };
        loss -= target[i] * log(p);
        grad_data[i] = -target[i] / p;
    }
    (loss, Vector::new(grad_data))
}
