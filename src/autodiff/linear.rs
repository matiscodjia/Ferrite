use crate::autodiff::module::Module;
use crate::autodiff::update::Update;
use crate::matrix::Matrix;
use crate::vector::Vector;
use libm::sqrtf;

#[derive(Clone, Copy)]
pub struct Linear<const IN: usize, const OUT: usize> {
    weights: Matrix<OUT, IN>,
    bias: Vector<OUT>,
}

impl<const IN: usize, const OUT: usize> Linear<IN, OUT> {
    /// Tous les poids à zéro — le réseau ne peut pas apprendre avec ça (dead neurons sur ReLU).
    pub fn zeros() -> Self {
        Linear {
            weights: Matrix::new(),
            bias: Vector::new([0.0; OUT]),
        }
    }

    /// Poids fournis explicitement — utile pour charger un modèle pré-entraîné.
    pub fn from_weights(weights: Matrix<OUT, IN>, bias: Vector<OUT>) -> Self {
        Linear { weights, bias }
    }

    /// Xavier uniform + Xorshift64 PRNG. Passe n'importe quelle graine non-nulle.
    /// Sur MCU, lis ton RNG hardware et passe la valeur : `Linear::from_seed(hal::rng::read())`.
    pub fn from_seed(seed: u64) -> Self {
        let mut state = if seed == 0 { 1 } else { seed };

        // Xavier uniform : valeurs dans [-limit, limit], limit = sqrt(6 / (IN + OUT))
        let limit = sqrtf(6.0 / (IN + OUT) as f32);

        let mut weights = Matrix::<OUT, IN>::new();
        for i in 0..OUT {
            for j in 0..IN {
                weights[(i, j)] = xorshift_f32(&mut state) * limit;
            }
        }

        // Biais initialisés à zéro — convention standard
        Linear {
            weights,
            bias: Vector::new([0.0; OUT]),
        }
    }
}

/// Xorshift64 — PRNG minimal, no_std, zéro dépendance.
/// Retourne un f32 dans [-1.0, 1.0].
fn xorshift_f32(state: &mut u64) -> f32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    // Mappe [0, u64::MAX] → [-1.0, 1.0]
    (x as f32) / (u64::MAX as f32) * 2.0 - 1.0
}

#[derive(Clone, Copy)]
pub struct LinearGrads<const IN: usize, const OUT: usize> {
    weights_grads: Matrix<OUT, IN>,
    bias_grad: Vector<OUT>,
}

impl<const IN: usize, const OUT: usize> Module<Vector<IN>> for Linear<IN, OUT> {
    type Output = Vector<OUT>;
    type Context = Vector<IN>;
    type Gradients = LinearGrads<IN, OUT>;

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

impl<const IN: usize, const OUT: usize> Update for Linear<IN, OUT> {
    type Gradients = LinearGrads<IN, OUT>;
    fn update(&mut self, grads: &Self::Gradients, lr: f32) {
        self.weights = self.weights - grads.weights_grads.scale(lr);
        self.bias = self.bias - grads.bias_grad * lr;
    }
}
