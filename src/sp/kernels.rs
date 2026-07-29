//! Hand-written filter banks.
//!
//! Kernels are built at call time from const taps, so they cost no static
//! storage. Note that [`cross_correlate2d`](super::cross_correlate2d) does not
//! flip them: for the symmetric [`Gaussian3D`] that is immaterial, but the
//! antisymmetric [`Sobel3D`] responses come out negated relative to a true
//! convolution.

use crate::linalg::tensor::{Tensor3D, Tensor4D};
use crate::scalar::Scalar;

/// The 3x3 taps of a kernel, row by row, before any channel replication.
type Taps3x3 = [Scalar; 9];

/// Spreads a 2D kernel over the C channels of a (C x 3 x 3) tensor, scaling
/// every tap by `gain`.
///
/// The depth axis is what makes these kernels usable by `cross_correlate2d`: a filter must
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
    /// use ferrite::linalg::tensor::Tensor3D;
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
/// `cross_correlate2d` expects — kernel `k` becomes output channel `k`.
///
/// This is the one place the kernels are copied: `cross_correlate2d` then reads the bank in
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
