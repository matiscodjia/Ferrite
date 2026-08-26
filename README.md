# frugal_ml

Ahead-of-time linear algebra and reverse-mode automatic differentiation for
bare-metal microcontrollers, written in Rust.

No heap. No `std`. No runtime. No dynamic dispatch. Every tensor shape and
every network topology is a fact the compiler knows and checks. The
computation graph is not a data structure built at runtime; it is a Rust
*type*, monomorphized once, at compile time, into a fixed sequence of
instructions. Designed to train and run small networks directly on STM32
and similar Cortex-M devices, with nothing between the source and the
flashed binary.

---

## Why ahead-of-time

Three well-known execution strategies for a differentiable-programming
engine, and where this one sits:

- **Eager** (PyTorch default, JAX untraced, micrograd): each operation
  executes immediately and records itself onto a runtime graph, a
  **Wengert list** ([Wengert, 1964][wengert64]), walked in reverse to
  compute gradients. Flexible (the graph can depend on runtime control
  flow), but the graph is a real, heap-allocated data structure that
  exists while the program runs.
- **Trace-then-compile** (`torch.compile`, JAX `jit`): run once eagerly to
  capture a graph, then hand it to a compiler (XLA, Inductor) that emits
  optimized code for that specific trace.
- **Ahead-of-time, graph-as-type** (this crate): there is no tracing step
  because there is nothing to trace. `Then<A, B>` (`src/autodiff/core/sequential.rs`)
  composes two `Module`s into a new type; `seq!(L1, L2, L3)` expands to
  `Then<L1, Then<L2, L3>>`. The Rust compiler resolves that whole nested
  type: every `forward`/`backward` call, every shape check, during
  ordinary monomorphization. By the time the binary exists, the "graph"
  has been erased: what remains is a fixed sequence of function calls,
  no different in principle from code written by hand with no framework
  at all.

This is a design point closer to classical algorithmic differentiation
([Griewank & Walther, 2008][griewank08]) than to today's mainstream ML
frameworks, and it exists because of a hard constraint most ML frameworks
never have to satisfy: **no heap**. Reverse-mode AD normally needs a tape,
a runtime record of the forward pass, walked backward. Without an
allocator, that tape has nowhere to live. Encoding the graph in the type
system instead means the "tape" is a compile-time artifact that costs
nothing at runtime, at the price of requiring every shape to be static.

### What this buys you, measured, not claimed

- **Dimension mismatches are compile errors, not runtime panics.**
  Composing two layers whose shapes don't line up fails with
  `E0599: the method 'forward' exists ... but its trait bounds were not satisfied`,
  naming the exact incompatible types, before the program exists, let
  alone runs on the target. PyTorch's equivalent is a `RuntimeError` at
  the exact line of the mismatched `matmul`, at inference time, on the
  device.
- **`Then::forward`/`backward` fully inline or resolve to a fixed,
  finite call chain**: never genuine recursion, never a vtable lookup.
  `Module` cannot be used as `dyn Module<Input>` at all: its associated
  types (`Output`, `Context`) would need to be pinned to one concrete
  type per trait object (`error[E0191]`), which would defeat the point of
  letting each layer have a different shape. Dynamic dispatch isn't
  merely avoided here, it's structurally impossible.
- **Real, on-target numbers** (STM32F446RE, Cortex-M4F @ 168 MHz, `probe-rs`
  + DWT cycle counting; see [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for
  full methodology and raw logs):
  - A 16-tap adaptive linear filter (`Linear<16,1,16>`, no hidden layer,
    an ADALINE per [Widrow & Hoff, 1960][widrow60]) trained online, one
    sample at a time: **3.42 µs per `forward`+`backward`+`update`**, a
    **36.5×** margin against the real-time budget at an 8 kHz sample rate
    (bounds-checked indexing; see `docs/BENCHMARKS.md` for the unchecked-
    indexing variant, ~8% faster end to end, on a separate branch pending merge).
  - A `64 → 40 → 10` classifier, trained for 300 epochs entirely on-device:
    **300/300** final accuracy, weights+gradients measured at **~50 KB**
    real peak stack usage against 128 KB total RAM.
  - Swapping bounds-checked tensor indexing (`get`/`set`, `debug_assert!`,
    free in release) for unchecked indexing (`get_unchecked`/`set_unchecked`,
    sound by construction: the same static shapes that back `get`/`set`
    already prove every index in range) measured **1.25×** faster on the
    isolated matmul, **1.087×** end-to-end on a full training step
    (`try_unchecked` branch, not yet merged; the numbers above this line
    are from the checked path currently on `main`).

### What it costs you: the honest column

- **No dynamic architectures.** The network's shape is fixed at compile
  time. There is no equivalent of PyTorch's `if condition: layer_a(x)
  else: layer_b(x)` inside a `forward`; the graph cannot depend on a
  runtime value. Anything requiring architecture search, early exit, or
  per-sample routing needs a different design.
- **`Then` is a strictly sequential chain**, not a general graph. It has
  no way to express a value used by two different downstream branches
  (a ResNet skip connection, an attention residual). Composing that
  today would need reshaping the API, not just writing more layers.
- **The naive mental model ("checked costs nothing in release, so it's
  free") is wrong, measured.** `Then::forward`/`backward` inlining
  everything into one caller (the common case) means that caller's stack
  frame is sized once, in one prologue, for the union of every
  intermediate value in the whole call chain, not the sum of what's
  logically alive at any one instant. On the `64-wide` hidden-layer
  classifier this meant **~127 KB of real stack use for a network whose
  weights+gradients need only ~50 KB in theory**, a ~2.5× gap between
  the naive parameter-counting estimate and reality, found by bisecting
  actual crashes on hardware, not by static analysis. Forcing separate,
  non-inlined stack frames per layer (`#[inline(never)]`) was tried as a
  fix and *increased* the footprint (127.83 KB vs 118.77 KB, same
  network): inlining lets the optimizer coalesce short-lived locals
  across layer boundaries, and opaque call boundaries block exactly that.
  The real fix is a different API shape (write into a caller-provided
  buffer instead of returning by value), tracked but not yet done.
- **`f32` by default**, tied to whatever hardware FPU precision the
  target has. `--features f64` exists for host-side gradient-check
  precision, not for deployment.

---

## Quick start

```rust
use frugal_ml::autodiff::{cross_entropy, sgd, Linear, Module, Softmax, Tanh};
use frugal_ml::seq;

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

`seq!(L1, L2, L3)` expands to `Then<L1, Then<L2, L3>>`, a recursive type
resolved entirely at compile time. The compiler sees the full graph: no
indirection, no dynamic dispatch.

---

## What is implemented

### Linear algebra

| | |
|---|---|
| `Tensor<ROWS, COLS, NUMEL>` | the crate's one elementary structure: indexing, `+ - * /`, `multiply`/`multiply_unchecked` (matrix product), `transposed`, column extraction |
| `Vector<N>` | `= Tensor<N, 1, N>`, still just a `Tensor`, with L1 / L2 / Linf norms, dot product, projection, Hadamard product |
| Gram-Schmidt | orthonormal basis from any set of vectors |
| QR decomposition | `A = QR`, used for linear system solving |
| Linear system solver | `Ax = b` via QR + back-substitution |
| SVD | one-sided Jacobi, full `A = U Σ Vᵀ` |
| `tensordot_1/2/3`, `ConvStreaming`, `cross_correlate2d` | fixed-kernel convolution (Sobel, Gaussian), inference only, not yet a trainable `Module` |

### Differentiable programming

| | |
|---|---|
| `Linear<IN, OUT, NUMEL>` | fully connected layer, Xavier-uniform init via a Xorshift64 PRNG |
| `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh` | element-wise activations, diagonal-Jacobian backward |
| `Softmax` | numerically stable (max subtraction), full dense-Jacobian VJP backward |
| `mse`, `mae` | regression losses |
| `cross_entropy` | classification loss, used after `Softmax` |
| `Sgd` | stochastic gradient descent, the only optimizer; no momentum/Adam state, deliberately, see below |
| `seq!` macro | `seq!(L1, L2, L3)` → `Then<L1, Then<L2, L3>>` |
| `GradChecker` | centered finite-difference cross-check against every analytical gradient |

**Why SGD only, for now:** Adam's `m`/`v` running moments each cost one
more buffer the exact size of the parameters, roughly doubling optimizer
state on top of the gradient SGD already needs. On a target where a
network's *entire* stack budget was measured at ~50 KB, that is not a
free upgrade; it directly competes with the RAM budget documented above.
Momentum (one extra buffer, not two) is the next planned step if plain
SGD's instability on non-stationary data becomes a real blocker.

### Initialization

```rust
Linear::<IN, OUT, NUMEL>::from_seed(seed)    // Xavier uniform + Xorshift64 PRNG
Linear::<IN, OUT, NUMEL>::from_weights(w, b) // load pretrained weights
Linear::<IN, OUT, NUMEL>::zeros()            // explicit zero init
```

`NUMEL` is always `IN * OUT`. Rust has no stable way to derive it
automatically from the other two ([`generic_const_exprs`][gce] is
unstable), so it is spelled out at each call site, and checked: every
`Tensor` carries a `const` item whose evaluation panics **at compile
time** if `NUMEL != ROWS * COLS` (`src/linalg/tensor/tensor2d.rs`), a
correctness guarantee about a compile-time constant, enforced by forcing
the compiler to evaluate it, not a runtime `assert!`.

On MCU, pass your hardware RNG output as seed:
```rust
Linear::<4, 8, 32>::from_seed(hal::rng::read())
```

---

## Gradient checking

All analytical gradients are verified against numerical differentiation
using centered finite differences, the same check, formalized, that a
first course in the subject starts with: `(L(θ+ε) − L(θ−ε)) / 2ε`.

```
eᵢ = |∂L/∂θᵢ (analytical) − ∂L/∂θᵢ (numerical)| / max(|analytical|, |numerical|, 1e-8)
```

| Mode | ε | mean error | max error |
|---|---|---|---|
| `f32` (default) | 1e-4 | ~6e-4 | ~2e-3 |
| `f64` (`--features f64`) | 1e-4 | <1e-6 | <1e-5 |

```bash
cargo test grad_check -- --nocapture
cargo test grad_check --features f64 -- --nocapture   # tighter precision floor
```

---

## Compile for STM32

```toml
[dependencies]
frugal_ml = { path = ".", default-features = false }
```

```bash
cargo build --target thumbv7em-none-eabihf --no-default-features
```

No allocator dependency. No heap calls anywhere in the crate: a
structural property of the design, not a claim checked after the fact.

---

## References

The design leans on established results in algorithmic differentiation
and adaptive filtering rather than inventing new theory. The contribution
here is the compile-time-graph engineering, not the mathematics:

- Widrow, B., & Hoff, M. E. (1960). *Adaptive Switching Circuits.*
  IRE WESCON Convention Record. ADALINE and the LMS update rule;
  `Linear<N,1,NUMEL>` + `mse` + `Sgd`, undecorated, **is** an LMS filter.
- Wengert, R. E. (1964). *A simple automatic derivative evaluation
  program.* Communications of the ACM, 7(8), 463-464. The tape/list
  this crate deliberately does not build at runtime.
- Griewank, A., & Walther, A. (2008). *Evaluating Derivatives:
  Principles and Techniques of Algorithmic Differentiation* (2nd ed.).
  SIAM. The standard reference for forward/reverse-mode AD outside the
  ML-framework tradition.
- Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018).
  *Automatic Differentiation in Machine Learning: a Survey.* Journal of
  Machine Learning Research, 18, 1-43.
- David, R., et al. (2021). *TensorFlow Lite Micro: Embedded Machine
  Learning on TinyML Systems.* Proceedings of Machine Learning and
  Systems (MLSys). The closest mainstream point of comparison; TFLite
  Micro interprets a serialized graph at runtime, this crate has no
  interpreter to compare against.

[wengert64]: https://dl.acm.org/doi/10.1145/355586.364791
[griewank08]: https://epubs.siam.org/doi/book/10.1137/1.9780898717761
[widrow60]: https://www-isl.stanford.edu/~widrow/papers/c1960adaptiveswitching.pdf
[gce]: https://github.com/rust-lang/rust/issues/76560

---

## Structure

```
src/
├── lib.rs
├── scalar.rs: Scalar type alias (f32 default, f64 via --features f64)
├── linalg/
│   ├── decomposition.rs: Gram-Schmidt, QR, SVD
│   ├── storage.rs: Storage/Buffer traits (stack vs heap backing)
│   └── tensor/
│       ├── tensor2d.rs: Tensor<ROWS,COLS,NUMEL>, TensorView, the Vector<N> alias
│       ├── tensor3d.rs: Tensor3D
│       ├── tensor4d.rs: Tensor4D, im2col_view
│       ├── tensor6d.rs: Rank6 trait, Tensor6D, TensorView6D
│       └── contraction.rs: tensordot_1/2/3
├── sp/
│   ├── correlate.rs: cross_correlate2d (fixed-kernel conv2d forward)
│   ├── conv_streaming.rs: ConvStreaming, O(KH·W) RAM row-at-a-time convolution
│   ├── shape.rs: conv_shape! macro
│   └── kernels.rs: Gaussian3D, Sobel3D, filter_bank
├── io/: .npy reading (host only, `std` feature)
└── autodiff/
    ├── core/
    │   ├── module.rs: Module<Input> trait
    │   ├── sequential.rs: Then<A,B>, seq! macro
    │   ├── update.rs: Update trait
    │   ├── perturb.rs: Perturb trait (parameter indexing for grad check)
    │   ├── flat_grads.rs: FlatGrads trait (gradient serialization)
    │   └── params.rs: Params trait (not re-exported: implement it to write a
    │                         custom layer, nothing else needs to name it directly)
    ├── grad_check.rs: GradChecker
    ├── layers/
    │   ├── linear.rs: Linear<IN, OUT, NUMEL>
    │   └── activations.rs: ReLU, LeakyReLU, Sigmoid, Tanh, Softmax
    ├── loss.rs: mse, mae, cross_entropy
    └── optim/
        ├── sgd.rs: Sgd, sgd()
        └── train.rs: train_step()

tests/
├── tensors.rs: Tensor/Vector unit tests, tensordot equivalences
├── algorithms.rs: Gram-Schmidt/QR/SVD integration tests
├── autodiff.rs: integration tests, real API usage
├── sequential.rs: Then/seq! composition tests, gradient check
└── sp.rs: cross_correlate2d integration tests
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for a concept-by-concept
walkthrough of every trait in `autodiff/core` and why each one exists, and
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for full on-target methodology,
raw logs, and plots.

---

## Roadmap

- [x] Live on-device training on real hardware: STM32F446RE, `probe-rs`,
      cycle-accurate measurement (`docs/BENCHMARKS.md`)
- [ ] `Conv2D` as a trainable `Module` (the `sp` primitives are inference-only today)
- [ ] Write-into-provided-buffer `Module` API, the fix for the measured stack-inlining cost
- [ ] `MaxPool2D`, `Flatten`
- [ ] Momentum optimizer (before Adam, see the RAM-cost argument above)
- [ ] Weight serialization to flash memory

---

## Status

Early, actively developed, breaking changes expected. Published on
crates.io to secure the name; treat pre-1.0 as exactly that.

## License

MIT
