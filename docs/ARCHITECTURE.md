# Architecture: `autodiff/core`, trait by trait

This is a concept-by-concept tour of every trait in `src/autodiff/core/`,
in dependency order, answering the question each one exists to answer.
Nothing here is a data structure built at runtime: every relationship
described is resolved by the compiler during monomorphization.

## `Params`: "what shape is my gradient?"

```rust
pub trait Params {
    type Gradients;
}
```

No methods. One associated type. Every trainable type answers this
question once: `Linear<IN,OUT,NUMEL>` says `type Gradients =
LinearGrads<IN,OUT,NUMEL>`; a stateless activation says `type Gradients =
()`. `Params` is deliberately *not* re-exported from the crate root:
grep the whole codebase and the only places it is named are the `impl
Params for X` blocks that provide the answer. Nothing consumes it
directly; it exists so that `Module`, `Update`, and `FlatGrads` each have
something to write `Self::Gradients` against.

## `Module<Input>`: forward, backward, and why `Input` is generic but `Output`/`Context` are not

```rust
pub trait Module<Input>: Params {
    type Output;
    type Context;
    fn then<Next>(self, next: Next) -> Then<Self, Next> where Self: Sized, Next: Module<Self::Output> { ... }
    fn forward(&self, x: Input) -> (Self::Output, Self::Context);
    fn backward(&self, grad_out: Self::Output, ctx: &Self::Context) -> (Input, Self::Gradients);
}
```

`Module: Params` is a supertrait bound, not decoration: `backward`'s
signature names `Self::Gradients`, and that name only resolves because
the bound guarantees it exists. Remove the bound and the trait fails to
compile with `E0220: associated type 'Gradients' not found for 'Self'`.

`Input` is a generic *parameter* of the trait; `Output`/`Context` are
associated *types*. The difference matters: a generic parameter lets the
same type implement the trait more than once (`Convert<f64>` and
`Convert<String>` for the same `Celsius`, in the standard library's own
idiom: `From<T>` works the same way). An associated type forces exactly
one answer per implementor. `Input` needs the former because nothing
prevents a layer from one day accepting more than one input shape;
`Output`/`Context` need the latter because, once `Input` is fixed, they
are a *function* of it: there is nothing left to choose.

`Context` is the explicit stand-in for what a closure captures implicitly
in a tape-based engine (micrograd's `_backward`, PyTorch's autograd node):
whatever `forward` needs to remember for `backward` to run later,
written out as a real, named type instead of an opaque captured
environment.

`.then()` is a **default method**: every `Module` gets it for free,
including `Then` itself, which is why `sequential.rs` used to redefine it
and no longer does (measured: byte-identical behavior with or without the
redefinition; removed as dead code). It must live here, in `module.rs`,
because Rust only allows a trait's default methods to be defined
alongside the trait itself, not an arbitrary choice but a language
constraint.

## `Then<A, B>`: composition, with nothing behind it but two fields

```rust
pub struct Then<A, B> {
    first: A,
    second: B,
}
```

Two distinct generic parameters, not one reused twice: `Then<A>` with a
single parameter would force `first` and `second` to be the *same
concrete type*, which would make `seq!(Linear, ReLU)` a type error before
anything else. `Then::new` carries no bounds at all: constructing a
`Then` out of two unrelated types compiles fine (`Then::new(5,
"hello")` is a legal, useless value). The bounds live entirely on the
trait impls that give `Then` behavior:

```rust
impl<A: Params, B: Params> Params for Then<A, B> {
    type Gradients = (A::Gradients, B::Gradients);
}

impl<Input, A, B> Module<Input> for Then<A, B>
where
    A: Module<Input>,
    B: Module<A::Output>,
{
    type Output = B::Output;
    type Context = (A::Context, B::Context);

    fn forward(&self, x: Input) -> (Self::Output, Self::Context) {
        let (out1, ctx1) = self.first.forward(x);
        let (out2, ctx2) = self.second.forward(out1);
        (out2, (ctx1, ctx2))
    }

    fn backward(&self, grad_out: Self::Output, ctx: &Self::Context) -> (Input, Self::Gradients) {
        let (ctx1, ctx2) = ctx;
        let (grad1, grads2) = self.second.backward(grad_out, ctx2);
        let (grad0, grads1) = self.first.backward(grad1, ctx1);
        (grad0, (grads1, grads2))
    }
}
```

`B: Module<A::Output>` (not `B: Module<Input>`) is the whole point: it
forces the second layer's expected input to match the first layer's
actual output, checked before the program exists. Attempting
`seq!(Linear::<2,4,8>, Linear::<5,1,5>)`, a 4-wide output feeding a
5-wide expectation, fails with `E0599`, naming the exact unsatisfied
bound (`Linear<5,1,5>: Module<Tensor<4,1,4>>`), not a `RuntimeError` on
target.

`backward` calls `second` before `first`: not stylistic, it *is* the
chain rule. Each `Module::backward` computes a vector-Jacobian product:
`grad_in[j] = Σᵢ grad_out[i] · ∂yᵢ/∂x_j`. Composing two layers means
applying that formula twice, and the second layer's `grad_in` is exactly
the first layer's `grad_out`: reverse order is a structural consequence
of the math, not a convention. What was produced first in `forward` is
consumed last in `backward`, symmetric around the loss.

## `Update`, `FlatGrads`, `Perturb`: one pattern, three capabilities

All three follow the identical shape: a trait with no relationship to
`Input`, plus a recursive impl for `Then` that splits an index/gradient
tuple across `first`/`second`:

```rust
impl<A, B> Update for Then<A, B> where A: Update, B: Update {
    fn update(&mut self, grads: &Self::Gradients, lr: Scalar) {
        self.first.update(&grads.0, lr);
        self.second.update(&grads.1, lr);
    }
}
```

`Update` applies gradient descent in place. `FlatGrads` serializes the
nested `Gradients` tuple into one flat buffer, in the same order
`Perturb` walks parameters: the two exist together so `GradChecker` can
compare an analytical gradient (via `FlatGrads`) against a numerical one
(via `Perturb`, nudging one scalar parameter at a time, wherever it lives
in the tree) term by term. None of the three needs to know what `Input`
was, which is precisely why they depend on `Params` (`Input`-independent)
rather than `Module<Input>`: tying them to `Module` would force every
caller to fix an arbitrary `Input` just to update weights, and would
introduce a real inference ambiguity the moment any type implemented
`Module` for more than one `Input` (verified: `E0283`, the same ambiguity
`Convert<f64>`/`Convert<String>` produces without an explicit type
annotation).

`Perturb` carries no bounds check across the `Then` boundary: routing an
out-of-range index lands wherever the arithmetic sends it: silently a
no-op if it lands in a stateless layer (`ReLU::perturb` ignores its
index unconditionally), a real out-of-bounds panic if it lands inside a
`Linear`'s bias array. Not a bug (`GradChecker`, the only caller, always
stays in range), but a real gap if you call it directly.

## Why no `dyn Module`

`Module` cannot be used as `dyn Module<Input>` without pinning `Output`
and `Context` to one concrete type each (`dyn Module<Input, Output = X,
Context = Y>`), verified as `error[E0191]`. Since every layer has a
different `Output`/`Context`, a single trait object could never represent
a heterogeneous chain the way `Then` does today. This is not a
performance choice enforced by convention; it is the associated-type
design making dynamic dispatch structurally impossible for this trait.
