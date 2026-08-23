use crate::linalg::storage::{Buffer, LenMismatch, OwnedStorage, StackStorage, Storage};
use crate::scalar::Scalar;

/// Read-only access to a rank-6 tensor, whichever way it holds its elements:
/// `Tensor6D` owns them, `TensorView6D` only aliases someone else's buffer.
///
/// The six dimensions are const parameters rather than runtime values, so a
/// contraction over an implementor keeps checking its shared axes at compile
/// time: the trait erases ownership, not shape.
pub trait Rank6<
    const D0: usize,
    const D1: usize,
    const D2: usize,
    const D3: usize,
    const D4: usize,
    const D5: usize,
>
{
    fn get(self: &Self, i0: usize, i1: usize, i2: usize, i3: usize, i4: usize, i5: usize)
        -> Scalar;
    /// # Safety
    /// The caller guarantees i0 < D0, i1 < D1, i2 < D2, i3 < D3, i4 < D4, i5 < D5.
    unsafe fn get_unchecked(
        self: &Self,
        i0: usize,
        i1: usize,
        i2: usize,
        i3: usize,
        i4: usize,
        i5: usize,
    ) -> Scalar;
    /// The flat, untransformed backing buffer. Every `Rank6` implementor stores its
    /// last axis (D5) with stride 1, so a caller that wants to walk that axis can
    /// index this buffer directly with `row_offset(..) + i5` instead of paying for
    /// a full `get_unchecked` (which recomputes every stride term) on each step.
    fn get_raw_buffer(self: &Self) -> &[Scalar];
    /// Flat offset of (i0, i1, i2, i3, i4, 0) into `get_raw_buffer()`: the start of
    /// the contiguous row along the last axis.
    /// # Safety
    /// The caller guarantees i0 < D0, i1 < D1, i2 < D2, i3 < D3, i4 < D4.
    unsafe fn row_offset(
        self: &Self,
        i0: usize,
        i1: usize,
        i2: usize,
        i3: usize,
        i4: usize,
    ) -> usize;
    fn shape(self: &Self) -> [usize; 6];
}

impl<
        const BATCHES: usize,
        const GROUPS: usize,
        const CHANNELS: usize,
        const DEPTH: usize,
        const ROWS: usize,
        const COLS: usize,
        const NUMEL: usize,
        S: Storage<[Scalar; NUMEL]>,
    > Rank6<BATCHES, GROUPS, CHANNELS, DEPTH, ROWS, COLS>
    for Tensor6D<BATCHES, GROUPS, CHANNELS, DEPTH, ROWS, COLS, NUMEL, S>
{
    fn get(self: &Self, b: usize, g: usize, c: usize, d: usize, i: usize, j: usize) -> Scalar {
        Tensor6D::get(self, b, g, c, d, i, j)
    }
    unsafe fn get_unchecked(
        self: &Self,
        b: usize,
        g: usize,
        c: usize,
        d: usize,
        i: usize,
        j: usize,
    ) -> Scalar {
        Tensor6D::get_unchecked(self, b, g, c, d, i, j)
    }
    fn get_raw_buffer(self: &Self) -> &[Scalar] {
        self.data.as_flat()
    }
    unsafe fn row_offset(self: &Self, b: usize, g: usize, c: usize, d: usize, i: usize) -> usize {
        b * self.batch_stride
            + g * self.group_stride
            + c * self.channel_stride
            + d * self.depth_stride
            + i * self.row_stride
    }
    fn shape(self: &Self) -> [usize; 6] {
        self.shape
    }
}

impl<
        'a,
        const N: usize,
        const C: usize,
        const H: usize,
        const W: usize,
        const H_OUT: usize,
        const W_OUT: usize,
        const KH: usize,
        const KW: usize,
    > Rank6<N, H_OUT, W_OUT, C, KH, KW> for TensorView6D<'a, N, C, H, W, H_OUT, W_OUT, KH, KW>
{
    fn get(self: &Self, n: usize, i: usize, j: usize, c: usize, p: usize, q: usize) -> Scalar {
        TensorView6D::get(self, n, i, j, c, p, q)
    }
    unsafe fn get_unchecked(
        self: &Self,
        n: usize,
        i: usize,
        j: usize,
        c: usize,
        p: usize,
        q: usize,
    ) -> Scalar {
        TensorView6D::get_unchecked(self, n, i, j, c, p, q)
    }
    fn get_raw_buffer(self: &Self) -> &[Scalar] {
        self.data
    }
    unsafe fn row_offset(self: &Self, n: usize, i: usize, j: usize, c: usize, p: usize) -> usize {
        n * self.n_stride
            + i * self.h_out_stride
            + j * self.w_out_stride
            + c * self.channel_stride
            + p * self.kh_stride
            + self.reference_index
    }
    fn shape(self: &Self) -> [usize; 6] {
        self.shape
    }
}
impl<
        'a,
        const N: usize,
        const C: usize,
        const H: usize,
        const W: usize,
        const H_OUT: usize,
        const W_OUT: usize,
        const KH: usize,
        const KW: usize,
    > TensorView6D<'a, N, C, H, W, H_OUT, W_OUT, KH, KW>
{
    /// # Safety
    /// The caller guarantees n < N, i < H_OUT, j < W_OUT, c < C, p < KH, q < KW.
    pub unsafe fn get_unchecked(
        &self,
        n: usize,
        i: usize,
        j: usize,
        c: usize,
        p: usize,
        q: usize,
    ) -> Scalar {
        let flat_index: usize = n * self.n_stride
            + i * self.h_out_stride
            + j * self.w_out_stride
            + c * self.channel_stride
            + p * self.kh_stride
            + q * self.kw_stride;
        let index: usize = flat_index + self.reference_index;

        *self.data.get_unchecked(index)
    }
}

/// An im2col view over a (N x C x H x W) tensor: one (C x KH x KW) receptive
/// field per output pixel, laid out as (N x H_OUT x W_OUT x C x KH x KW).
///
/// The const parameters go source tensor first (N, C, H, W), then window
/// geometry (H_OUT, W_OUT, KH, KW). No data is copied: the view only remaps
/// strides onto the input buffer, so the same element is aliased by every
/// window that overlaps it.
pub struct TensorView6D<
    'a,
    const N: usize,
    const C: usize,
    const H: usize,
    const W: usize,
    const H_OUT: usize,
    const W_OUT: usize,
    const KH: usize,
    const KW: usize,
> {
    pub(super) data: &'a [Scalar],
    pub(super) reference_index: usize,
    pub(super) n_stride: usize,
    pub(super) h_out_stride: usize,
    pub(super) w_out_stride: usize,
    pub(super) channel_stride: usize,
    pub(super) kh_stride: usize,
    pub(super) kw_stride: usize,
    pub(super) shape: [usize; 6],
}
impl<
        'a,
        const N: usize,
        const C: usize,
        const H: usize,
        const W: usize,
        const H_OUT: usize,
        const W_OUT: usize,
        const KH: usize,
        const KW: usize,
    > TensorView6D<'a, N, C, H, W, H_OUT, W_OUT, KH, KW>
{
    /// Logical axes: (N x H_OUT x W_OUT x C x KH x KW), the operand order
    /// expected by `tensordot_3`. The underlying buffer stays the (N x C x H x W)
    /// input tensor: only the strides move.
    pub fn get(self: &Self, n: usize, i: usize, j: usize, c: usize, p: usize, q: usize) -> Scalar {
        debug_assert!(
            n < self.shape[0]
                && i < self.shape[1]
                && j < self.shape[2]
                && c < self.shape[3]
                && p < self.shape[4]
                && q < self.shape[5]
        );
        let flat_index: usize = n * self.n_stride
            + i * self.h_out_stride
            + j * self.w_out_stride
            + c * self.channel_stride
            + p * self.kh_stride
            + q * self.kw_stride;
        let index: usize = flat_index + self.reference_index;
        self.data[index]
    }
}

pub struct Tensor6D<
    const BATCHES: usize,
    const GROUPS: usize,
    const CHANNELS: usize,
    const DEPTH: usize,
    const ROWS: usize,
    const COLS: usize,
    const NUMEL: usize,
    S: Storage<[Scalar; NUMEL]> = StackStorage<[Scalar; NUMEL]>,
> {
    data: S,
    batch_stride: usize,
    group_stride: usize,
    channel_stride: usize,
    depth_stride: usize,
    row_stride: usize,
    col_stride: usize,
    shape: [usize; 6],
}

impl<
        const BATCHES: usize,
        const GROUPS: usize,
        const CHANNELS: usize,
        const DEPTH: usize,
        const ROWS: usize,
        const COLS: usize,
        const NUMEL: usize,
        S: Storage<[Scalar; NUMEL]>,
    > Tensor6D<BATCHES, GROUPS, CHANNELS, DEPTH, ROWS, COLS, NUMEL, S>
{
    const STRUCTURE_CORRECTNESS: () =
        assert!(NUMEL == BATCHES * GROUPS * CHANNELS * DEPTH * ROWS * COLS);

    /// Zero-initialized accumulator for the crate's own algorithms. Kept out
    /// of the public API on purpose: an external caller should never get a
    /// silently-zeroed tensor by omitting a load step.
    pub(crate) fn zeroed() -> Self
    where
        S: OwnedStorage<[Scalar; NUMEL]>,
    {
        let _ = Self::STRUCTURE_CORRECTNESS;
        Self {
            data: S::zeroed(),
            batch_stride: GROUPS * CHANNELS * DEPTH * ROWS * COLS,
            group_stride: CHANNELS * DEPTH * ROWS * COLS,
            channel_stride: DEPTH * ROWS * COLS,
            depth_stride: ROWS * COLS,
            row_stride: COLS,
            col_stride: 1,
            shape: [BATCHES, GROUPS, CHANNELS, DEPTH, ROWS, COLS],
        }
    }
    /// Builds a tensor from data known upfront, the caller never needs `mut`
    /// or a separate load step.
    pub fn new(data: [Scalar; NUMEL]) -> Self
    where
        S: OwnedStorage<[Scalar; NUMEL]>,
    {
        let mut t = Self::zeroed();
        t.load_data(data);
        t
    }
    pub fn get(self: &Self, b: usize, g: usize, c: usize, d: usize, i: usize, j: usize) -> Scalar {
        debug_assert!(
            b < self.shape[0]
                && g < self.shape[1]
                && c < self.shape[2]
                && d < self.shape[3]
                && i < self.shape[4]
                && j < self.shape[5]
        );
        let flat_index: usize = b * self.batch_stride
            + g * self.group_stride
            + c * self.channel_stride
            + d * self.depth_stride
            + i * self.row_stride
            + j * self.col_stride;
        self.data[flat_index]
    }
    pub fn set(
        self: &mut Self,
        b: usize,
        g: usize,
        c: usize,
        d: usize,
        i: usize,
        j: usize,
        value: Scalar,
    ) -> () {
        debug_assert!(
            b < self.shape[0]
                && g < self.shape[1]
                && c < self.shape[2]
                && d < self.shape[3]
                && i < self.shape[4]
                && j < self.shape[5]
        );
        let flat_index: usize = b * self.batch_stride
            + g * self.group_stride
            + c * self.channel_stride
            + d * self.depth_stride
            + i * self.row_stride
            + j * self.col_stride;
        self.data[flat_index] = value;
    }
    /// # Safety
    /// The caller guarantees b < self.shape[0], g < self.shape[1], c < self.shape[2],
    /// d < self.shape[3], i < self.shape[4], j < self.shape[5].
    pub unsafe fn get_unchecked(
        self: &Self,
        b: usize,
        g: usize,
        c: usize,
        d: usize,
        i: usize,
        j: usize,
    ) -> Scalar {
        let flat_index: usize = b * self.batch_stride
            + g * self.group_stride
            + c * self.channel_stride
            + d * self.depth_stride
            + i * self.row_stride
            + j * self.col_stride;
        *self.data.get_unchecked(flat_index)
    }
    /// # Safety
    /// The caller guarantees b < self.shape[0], g < self.shape[1], c < self.shape[2],
    /// d < self.shape[3], i < self.shape[4], j < self.shape[5].
    pub unsafe fn set_unchecked(
        self: &mut Self,
        b: usize,
        g: usize,
        c: usize,
        d: usize,
        i: usize,
        j: usize,
        value: Scalar,
    ) -> () {
        let flat_index: usize = b * self.batch_stride
            + g * self.group_stride
            + c * self.channel_stride
            + d * self.depth_stride
            + i * self.row_stride
            + j * self.col_stride;
        *self.data.get_unchecked_mut(flat_index) = value;
    }
    /// Loads a full, statically-sized buffer, for small tensors and known
    /// data. For a tensor too big to build `[Scalar; NUMEL]` on the stack, see
    /// [`Self::load_slice`] (`no_std`, no `alloc`) or [`Self::from_vec`].
    pub fn load_data(self: &mut Self, data: [Scalar; NUMEL]) -> () {
        self.data.as_flat_mut().copy_from_slice(&data);
    }
    /// Copies from a runtime-sized slice, never materializes `[Scalar; NUMEL]`
    /// on the stack.
    pub fn load_slice(self: &mut Self, data: &[Scalar]) -> Result<(), LenMismatch> {
        if data.len() != NUMEL {
            return Err(LenMismatch);
        }
        self.data.as_flat_mut().copy_from_slice(data);
        Ok(())
    }
    /// Builds a tensor straight from a `Vec`, no compile-time-sized array
    /// ever materialized.
    #[cfg(feature = "alloc")]
    pub fn from_vec(data: alloc::vec::Vec<Scalar>) -> Result<Self, alloc::vec::Vec<Scalar>>
    where
        S: OwnedStorage<[Scalar; NUMEL]>,
    {
        if data.len() != NUMEL {
            return Err(data);
        }
        let mut t = Self::zeroed();
        t.data.as_flat_mut().copy_from_slice(&data);
        Ok(t)
    }
}
