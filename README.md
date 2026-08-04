# Ferrite

Static deep learning framework for bare-metal microcontrollers, written in Rust.

No heap. No `std`. No runtime. Everything lives on the stack, all sizes are known at compile time via const generics. Designed to train and run neural networks directly on STM32 and similar Cortex-M devices.

---

## Design principles

- **Zero allocation** — no `Vec`, no `Box`, no allocator required
- **Static graphs** — network architecture is a type, resolved entirely at compile time
- **Portable** — `no_std` by default, `std` feature for development on desktop
- **Single dependency** — `libm` for `sin`, `exp`, `sqrt` and friends

---

## Quick start

```rust
use ferrite::autodiff::{cross_entropy, sgd, Linear, Module, Softmax, Tanh};
use ferrite::seq;

let mut network = seq!(
    Linear::<4, 16, 64>::from_seed(42),
    Tanh::<16> {},
    Linear::<16, 3, 48>::from_seed(137),
    Softmax::<3> {},
);

// One training step.
let (output, ctx) = network.forward(input);
let (loss, grad)  = cross_entropy(output, target);
let (_, grads)    = network.backward(grad, &ctx);
sgd(&mut network, &grads, 0.05);
```

`seq!(L1, L2, L3)` expands to `Then<L1, Then<L2, L3>>` — a recursive type resolved entirely at compile time. The compiler sees the full graph: no indirection, no dynamic dispatch, full inlining.

---

## What is implemented

### Linear algebra

| | |
|---|---|
| `Tensor<ROWS, COLS, NUMEL>` | the crate's one elementary structure — indexing, `+ - * /`, `multiply` (matrix product), `transposed`, column extraction |
| `Vector<N>` | `= Tensor<N, 1, N>` — still just a `Tensor`, with L1 / L2 / Linf norms, dot product, projection, Hadamard product |
| Gram-Schmidt | orthonormal basis from any set of vectors |
| QR decomposition | `A = QR`, used for linear system solving |
| Linear system solver | `Ax = b` via QR + back-substitution |
| SVD | one-sided Jacobi, full `A = U Σ Vᵀ` |

### Deep learning

| | |
|---|---|
| `Linear<IN, OUT, NUMEL>` | fully connected layer, Xavier uniform init via Xorshift64 PRNG |
| `ReLU`, `Sigmoid`, `Tanh` | element-wise activations with correct backward |
| `Softmax` | numerically stable (max subtraction), full VJP backward |
| `MSE`, `MAE` | regression losses |
| `cross_entropy` | classification loss, use after Softmax |
| `SGD` | stochastic gradient descent |
| `seq!` macro | ergonomic composition — `seq!(L1, L2, L3)` → `Then<L1, Then<L2, L3>>` |

### Initialization

```rust
Linear::<IN, OUT, NUMEL>::from_seed(seed)    // Xavier uniform + Xorshift64 PRNG
Linear::<IN, OUT, NUMEL>::from_weights(w, b) // load pretrained weights
Linear::<IN, OUT, NUMEL>::zeros()            // explicit zero init
```

`NUMEL` is always `IN * OUT` — Rust has no stable way to derive it automatically from the other two, so it's spelled out at each call site.

On MCU, pass your hardware RNG output as seed:
```rust
Linear::<4, 8, 32>::from_seed(hal::rng::read())
```

---

## Performance

Benchmarks on Apple M4 pro (release mode), single sample, batch size 1:

| Network | Forward | Forward + Backward | Full step |
|---|---|---|---|
| `2 → 4 → 1` | 16.5 ns | 17.1 ns | 50.7 ns |
| `2 → 8 → 4 → 1` | 63.4 ns | 83.0 ns | 116.8 ns |

On STM32F4 at 168 MHz with FPU, expect roughly 20–50x slower — still well within range for real-time learning at 100 Hz sensor rates.

---

## Validation

Iris dataset (UCI), 4 features, 3 classes, 150 samples, 80/20 split:

```
epoch    0 | train loss 0.4574
epoch  200 | train loss 0.0272
epoch 2000 | train loss 0.0443

train accuracy : 118/120 (98.3%)
test  accuracy :   30/30 (100.0%)
```

Network: `Linear<4,16,64> → Tanh → Linear<16,3,48> → Softmax`, SGD lr=0.05, 2000 epochs.

---

## Compile for STM32

```toml
# Cargo.toml
[dependencies]
ferrite = { path = ".", default-features = false }
```

```bash
cargo build --target thumbv7em-none-eabihf --no-default-features
```

No allocator needed. The library produces no heap calls — verified by design.

---

## Gradient checking

All analytical gradients are verified against numerical differentiation using centered finite differences.

**Protocol**

For each parameter θᵢ, the numerical gradient is estimated as:

```
∂L/∂θᵢ ≈ (L(θᵢ + ε) − L(θᵢ − ε)) / 2ε
```

The relative error per parameter uses the max as denominator to avoid inflating small-gradient errors:

```
eᵢ = |∂L/∂θᵢ (analytical) − ∂L/∂θᵢ (numerical)| / max(|analytical|, |numerical|, 1e-8)
```

**Results**

| Mode | ε | mean error | max error |
|---|---|---|---|
| `f32` (default) | 1e-4 | ~6e-4 | ~2e-3 |
| `f64` (`--features f64`) | 1e-4 | <1e-6 | <1e-5 |

In `f64`, errors are well below the theoretical floor for centered differences (~ε²), confirming the analytical gradients are correct. The `f32` errors are consistent with floating-point precision limits and have no practical impact on training.

**Running the check**

```bash
# f32 (default)
cargo test grad_check -- --nocapture

# f64 — higher precision validation
cargo test grad_check --features f64 -- --nocapture
```

The checker is implemented in `src/autodiff/grad_check.rs`. It requires `N` (total parameter count) as a const generic — `IN*OUT + OUT` per `Linear` layer, 0 for activations:

```rust
// Linear<2,4,8> → Tanh<4> → Linear<4,1,4> : 12 + 0 + 5 = 17
let result = GradChecker::check::<17, _, _>(net, input, target, mse, 1e-4);
assert!(result.max_relative_error < 1e-2);
```

---

## Roadmap

- [ ] STM32 Nucleo deployment — live training on sensor data via ADC
- [ ] Benchmarks on real hardware (Cortex-M4 with FPU)
- [ ] `Conv2D` layer with static feature map dimensions
- [ ] `MaxPool2D`, `Flatten`
- [ ] Weight serialization to flash memory
- [ ] Adam optimizer

---

## Structure

```
src/
├── lib.rs
├── scalar.rs          — Scalar type alias (f32 default, f64 via --features f64)
├── linalg/
│   ├── decomposition.rs   — Gram-Schmidt, QR, SVD
│   ├── storage.rs         — Storage/Buffer traits (stack vs heap backing)
│   └── tensor/
│       ├── tensor2d.rs    — Tensor<ROWS,COLS,NUMEL>, TensorView, the Vector<N> alias
│       ├── tensor3d.rs    — Tensor3D
│       ├── tensor4d.rs    — Tensor4D, im2col_view
│       ├── tensor6d.rs    — Rank6 trait, Tensor6D, TensorView6D
│       └── contraction.rs — tensordot_1/2/3
├── sp/
│   ├── correlate.rs   — cross_correlate2d (conv2d forward pass)
│   └── kernels.rs     — Gaussian3D, Sobel3D, filter_bank
├── io/                — .npy reading (host only, `std` feature)
└── autodiff/
    ├── core/
    │   ├── module.rs      — Module<Input> trait
    │   ├── sequential.rs  — Then<A,B>, seq! macro
    │   ├── update.rs      — Update trait
    │   ├── perturb.rs     — Perturb trait (parameter indexing for grad check)
    │   ├── flat_grads.rs  — FlatGrads trait (gradient serialization)
    │   └── params.rs      — Params trait
    ├── grad_check.rs      — GradChecker
    ├── layers/
    │   ├── linear.rs      — Linear<IN, OUT, NUMEL>
    │   └── activations.rs — ReLU, LeakyReLU, Sigmoid, Tanh, Softmax
    ├── loss.rs            — mse, mae, cross_entropy
    └── optim/
        ├── sgd.rs         — Sgd, sgd()
        └── train.rs       — train_step()

tests/
├── tensors.rs         — Tensor/Vector unit tests, tensordot equivalences
├── algorithms.rs      — Gram-Schmidt/QR/SVD integration tests
├── autodiff.rs        — integration tests (real API usage, incl. Iris dataset)
├── sequential.rs      — Then/seq! composition tests, gradient check
└── sp.rs              — cross_correlate2d integration tests
```
