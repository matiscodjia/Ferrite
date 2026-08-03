//! 2D cross-correlation over a batch of multi-channel frames.

use crate::linalg::storage::{OwnedStorage, Storage};
use crate::linalg::tensor::{tensordot_3, Tensor4D};
use crate::scalar::Scalar;

/// Cross-correlates a (N x C x H x W) sequence with K filters of shape
/// (C x KH x KW), producing a (N x H_OUT x W_OUT x K) sequence — one new channel
/// per filter.
///
/// The kernel is *not* flipped, so this is a correlation, not a convolution —
/// the same choice every deep learning framework makes behind the name `conv2d`.
/// For learned weights the distinction is vacuous, training absorbs the flip.
/// For the hand-written kernels in [`kernels`](super::kernels) it is not:
/// [`Sobel3D::x`](super::Sobel3D::x) measures -d/dx here, where a true
/// convolution would give +d/dx.
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
/// use ferrite::linalg::tensor::Tensor4D;
/// use ferrite::sp::cross_correlate2d;
///
/// // one 3x3 frame, single channel
/// let mut frames = Tensor4D::<1, 1, 3, 3, 9>::new();
/// frames.load_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
///
/// // one 2x2 box filter
/// let mut filters = Tensor4D::<1, 1, 2, 2, 4>::new();
/// filters.load_data([1.0, 1.0, 1.0, 1.0]);
///
/// let out: Tensor4D<1, 2, 2, 1, 4> = cross_correlate2d(&frames, &filters, 1);
/// assert_eq!(12.0, out.get(0, 0, 0, 0)); // 1+2+4+5
/// ```
pub fn cross_correlate2d<
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
    SX,
    SF,
    SY,
>(
    sequence: &Tensor4D<N, C, H, W, NUMEL_X, SX>,
    filters: &Tensor4D<K, C, KH, KW, NUMEL_F, SF>,
    stride: usize,
) -> Tensor4D<N, H_OUT, W_OUT, K, NUMEL_Y, SY>
where
    SX: Storage<[Scalar; NUMEL_X]>,
    SF: Storage<[Scalar; NUMEL_F]>,
    SY: OwnedStorage<[Scalar; NUMEL_Y]>,
{
    // (N x C x H x W) seen as (N x H_OUT x W_OUT x C x KH x KW), no copy
    let patches = sequence.im2col_view::<H_OUT, W_OUT, KH, KW>(stride);
    // contract (C x KH x KW) against each of the K filters
    tensordot_3(&patches, filters)
}
