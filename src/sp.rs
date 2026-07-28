//! Signal processing toolkit.
//!
//! Operators that read a signal — a video sequence, a stack of feature maps —
//! through a bank of filters. Everything is shaped at compile time and lives on
//! the stack, so a `no_std` target pays no allocation for the intermediate
//! layouts.

use crate::scalar::Scalar;
use crate::tensor::{tensordot_3, Tensor3D, Tensor4D};

/// The 3x3 taps of a kernel, row by row, before any channel replication.
type Taps3x3 = [Scalar; 9];

/// Spreads a 2D kernel over the C channels of a (C x 3 x 3) tensor, scaling
/// every tap by `gain`.
///
/// The depth axis is what makes these kernels usable by `conv2d`: a filter must
/// span every input channel, because the contraction sums over them. Replicating
/// the same 2D taps means the filter treats all channels alike — the C-channel
/// response is the sum of the per-channel responses.
fn replicate<const C: usize, const NUMEL: usize>(
    taps: &Taps3x3,
    gain: Scalar,
) -> Tensor3D<C, 3, 3, NUMEL> {
    let mut kernel = Tensor3D::<C, 3, 3, NUMEL>::new();
    for c in 0..C {
        for i in 0..3 {
            for j in 0..3 {
                kernel.set(c, i, j, taps[i * 3 + j] * gain);
            }
        }
    }
    kernel
}

/// 3x3 binomial approximation of a Gaussian, replicated over C channels.
///
/// Taps are `[1 2 1; 2 4 2; 1 2 1] / 16`, further divided by C so that summing
/// the channel responses keeps a unit DC gain: a constant sequence comes out at
/// its own level, whatever its number of channels.
pub struct Gaussian3D;

impl Gaussian3D {
    const TAPS: Taps3x3 = [1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0];

    /// ```
    /// use ferrite::sp::Gaussian3D;
    /// use ferrite::tensor::Tensor3D;
    ///
    /// let k: Tensor3D<1, 3, 3, 9> = Gaussian3D::kernel();
    /// assert_eq!(0.25, k.get(0, 1, 1)); // 4/16
    /// ```
    pub fn kernel<const C: usize, const NUMEL: usize>() -> Tensor3D<C, 3, 3, NUMEL> {
        replicate::<C, NUMEL>(&Self::TAPS, 1.0 / (16.0 * C as Scalar))
    }
}

/// 3x3 Sobel gradient operator, replicated over C channels.
///
/// `x` differentiates along the columns (vertical edges), `y` along the rows.
/// Taps are the classic integer ones, divided by C: for a single channel the
/// kernel is exactly `[-1 0 1; -2 0 2; -1 0 1]`, and for more the response is
/// the mean gradient across channels rather than C times it.
pub struct Sobel3D;

impl Sobel3D {
    const TAPS_X: Taps3x3 = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
    const TAPS_Y: Taps3x3 = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

    /// Horizontal gradient: responds to vertical edges.
    pub fn x<const C: usize, const NUMEL: usize>() -> Tensor3D<C, 3, 3, NUMEL> {
        replicate::<C, NUMEL>(&Self::TAPS_X, 1.0 / C as Scalar)
    }

    /// Vertical gradient: responds to horizontal edges.
    pub fn y<const C: usize, const NUMEL: usize>() -> Tensor3D<C, 3, 3, NUMEL> {
        replicate::<C, NUMEL>(&Self::TAPS_Y, 1.0 / C as Scalar)
    }
}

/// Stacks K kernels of shape (C x KH x KW) into the (K x C x KH x KW) bank that
/// `conv2d` expects — kernel `k` becomes output channel `k`.
///
/// This is the one place the kernels are copied: `conv2d` then reads the bank in
/// place. `NUMEL_BANK == K * C * KH * KW` is checked by `Tensor4D::new`.
pub fn filter_bank<
    const K: usize,
    const C: usize,
    const KH: usize,
    const KW: usize,
    const NUMEL_KERNEL: usize,
    const NUMEL_BANK: usize,
>(
    kernels: [&Tensor3D<C, KH, KW, NUMEL_KERNEL>; K],
) -> Tensor4D<K, C, KH, KW, NUMEL_BANK> {
    let mut bank = Tensor4D::<K, C, KH, KW, NUMEL_BANK>::new();
    for k in 0..K {
        for c in 0..C {
            for i in 0..KH {
                for j in 0..KW {
                    bank.set(k, c, i, j, kernels[k].get(c, i, j));
                }
            }
        }
    }
    bank
}

/// Convolves a (N x C x H x W) sequence with K filters of shape (C x KH x KW),
/// producing a (N x H_OUT x W_OUT x K) sequence — one new channel per filter.
///
/// `sequence` is the video: N frames of C channels, each H x W. `filters` is the
/// bank: K filters, each spanning all C input channels over a KH x KW window.
/// Each filter contracts a whole receptive field down to one value, so the input
/// channels disappear and the K filters become the output channels — the result
/// is channel-last, (N x H_OUT x W_OUT x K).
///
/// The window slides by `stride` in both directions. No padding: the output is
/// the usual `(H - KH) / stride + 1` by `(W - KW) / stride + 1`.
///
/// `H_OUT`, `W_OUT` and `NUMEL_Y` must come from the call site — deriving them
/// from `stride` would need `generic_const_exprs` — and are checked by
/// `im2col_view` and `Tensor4D::new`. Everything else is inferred from the
/// operands, and the shared C, KH, KW are enforced by the signature.
///
/// The patches are never materialised: `im2col_view` only remaps strides onto
/// the input buffer, and `tensordot_3` contracts that view in place. Cost is
/// N * H_OUT * W_OUT * K * C * KH * KW multiply-adds and zero extra storage.
///
/// ```
/// use ferrite::sp::conv2d;
/// use ferrite::tensor::Tensor4D;
///
/// // one 3x3 frame, single channel
/// let mut frames = Tensor4D::<1, 1, 3, 3, 9>::new();
/// frames.load_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
///
/// // one 2x2 box filter
/// let mut filters = Tensor4D::<1, 1, 2, 2, 4>::new();
/// filters.load_data([1.0, 1.0, 1.0, 1.0]);
///
/// let out: Tensor4D<1, 2, 2, 1, 4> = conv2d(&frames, &filters, 1);
/// assert_eq!(12.0, out.get(0, 0, 0, 0)); // 1+2+4+5
/// ```
pub fn conv2d<
    const N: usize,
    const C: usize,
    const H: usize,
    const W: usize,
    const K: usize,
    const KH: usize,
    const KW: usize,
    const H_OUT: usize,
    const W_OUT: usize,
    const NUMEL_X: usize,
    const NUMEL_F: usize,
    const NUMEL_Y: usize,
>(
    sequence: &Tensor4D<N, C, H, W, NUMEL_X>,
    filters: &Tensor4D<K, C, KH, KW, NUMEL_F>,
    stride: usize,
) -> Tensor4D<N, H_OUT, W_OUT, K, NUMEL_Y> {
    // (N x C x H x W) seen as (N x H_OUT x W_OUT x C x KH x KW), no copy
    let patches = sequence.im2col_view::<H_OUT, W_OUT, KH, KW>(stride);
    // contract (C x KH x KW) against each of the K filters
    tensordot_3(&patches, filters)
}
