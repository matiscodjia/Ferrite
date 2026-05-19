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
use ferrite::autodiff::{
    activations::{Tanh, Softmax},
    linear::Linear,
    loss::cross_entropy,
    module::Module,
    optims::sgd::sgd,
};
use ferrite::seq;

let mut network = seq!(
    Linear::<4, 16>::from_seed(42),
    Tanh::<16> {},
    Linear::<16, 3>::from_seed(137),
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
| `Vector<N>` | L1 / L2 / Linf norms, dot product, projection, Hadamard product |
| `Matrix<M, N>` | mul, transpose, scale, `mul_vec`, column extraction |
| Gram-Schmidt | orthonormal basis from any set of vectors |
| QR decomposition | `A = QR`, used for linear system solving |
| Linear system solver | `Ax = b` via QR + back-substitution |
| SVD | one-sided Jacobi, full `A = U Σ Vᵀ` |

### Deep learning

| | |
|---|---|
| `Linear<IN, OUT>` | fully connected layer, Xavier uniform init via Xorshift64 PRNG |
| `ReLU`, `Sigmoid`, `Tanh` | element-wise activations with correct backward |
| `Softmax` | numerically stable (max subtraction), full VJP backward |
| `MSE`, `MAE` | regression losses |
| `cross_entropy` | classification loss, use after Softmax |
| `SGD` | stochastic gradient descent |
| `seq!` macro | ergonomic composition — `seq!(L1, L2, L3)` → `Then<L1, Then<L2, L3>>` |

### Initialization

```rust
Linear::<IN, OUT>::from_seed(seed)    // Xavier uniform + Xorshift64 PRNG
Linear::<IN, OUT>::from_weights(w, b) // load pretrained weights
Linear::<IN, OUT>::zeros()            // explicit zero init
```

On MCU, pass your hardware RNG output as seed:
```rust
Linear::<4, 8>::from_seed(hal::rng::read())
```

---

## Performance

Benchmarks on Apple M3 (release mode), single sample, batch size 1:

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

Network: `Linear<4,16> → Tanh → Linear<16,3> → Softmax`, SGD lr=0.05, 2000 epochs.

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
├── vector.rs          — Vector<N>
├── matrix.rs          — Matrix<M, N>
├── algorithms.rs      — Gram-Schmidt, QR, SVD
└── autodiff/
    ├── module.rs      — Module<Input> trait, then() combinator
    ├── sequential.rs  — Then<A,B>, seq! macro
    ├── update.rs      — Update trait
    ├── linear.rs      — Linear<IN, OUT>
    ├── activations.rs — ReLU, Sigmoid, Tanh, Softmax
    ├── loss.rs        — mse, mae, cross_entropy
    └── optims/
        └── sgd.rs     — sgd()

tests/
└── autodiff.rs        — integration tests (real API usage)
```
