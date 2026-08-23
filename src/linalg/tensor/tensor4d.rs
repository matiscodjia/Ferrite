use super::tensor6d::TensorView6D;
use crate::linalg::storage::{Buffer, LenMismatch, OwnedStorage, StackStorage, Storage};
use crate::scalar::Scalar;

/// `S` determines where the buffer lives — the stack by default, so
/// existing instantiations (`Tensor4D::<1, 3, 32, 32, 3072>`) are
/// unchanged. See [`Tensor4DBoxed`] for the heap variant.
pub struct Tensor4D<
    const BATCHES: usize,
    const CHANNELS: usize,
    const ROWS: usize,
    const COLS: usize,
    const NUMEL: usize,
    S: Storage<[Scalar; NUMEL]> = StackStorage<[Scalar; NUMEL]>,
> {
    data: S,
    batch_stride: usize,
    channel_stride: usize,
    row_stride: usize,
    col_stride: usize,
    pub(super) shape: [usize; 4],
}

/// `Tensor4D` whose buffer lives on the heap — for shapes the stack
/// cannot carry (scaling-up benchmarks).
#[cfg(feature = "alloc")]
pub type Tensor4DBoxed<
    const BATCHES: usize,
    const CHANNELS: usize,
    const ROWS: usize,
    const COLS: usize,
    const NUMEL: usize,
> = Tensor4D<
    BATCHES,
    CHANNELS,
    ROWS,
    COLS,
    NUMEL,
    crate::linalg::storage::HeapStorage<[Scalar; NUMEL]>,
>;

impl<
        const BATCHES: usize,
        const CHANNELS: usize,
        const ROWS: usize,
        const COLS: usize,
        const NUMEL: usize,
        S: Storage<[Scalar; NUMEL]>,
    > Tensor4D<BATCHES, CHANNELS, ROWS, COLS, NUMEL, S>
{
    const STRUCTURE_CORRECTNESS: () = assert!(NUMEL == BATCHES * CHANNELS * ROWS * COLS);

    /// Zero-initialized accumulator for the crate's own algorithms
    /// (`tensordot_3`, `sp::kernels::filter_bank`), which build a result
    /// element by element. Kept out of the public API on purpose: an external
    /// caller should never get a silently-zeroed tensor by omitting a load
    /// step.
    pub(crate) fn zeroed() -> Self
    where
        S: OwnedStorage<[Scalar; NUMEL]>,
    {
        let _ = Self::STRUCTURE_CORRECTNESS;
        Self {
            data: S::zeroed(),
            batch_stride: CHANNELS * ROWS * COLS,
            channel_stride: ROWS * COLS,
            row_stride: COLS,
            col_stride: 1,
            shape: [BATCHES, CHANNELS, ROWS, COLS],
        }
    }
    /// Builds a tensor from data known upfront — the caller never needs `mut`
    /// or a separate load step.
    pub fn new(data: [Scalar; NUMEL]) -> Self
    where
        S: OwnedStorage<[Scalar; NUMEL]>,
    {
        let mut t = Self::zeroed();
        t.load_data(data);
        t
    }
    /// Copies from a runtime-sized slice — never materializes `[Scalar; NUMEL]`
    /// on the stack, so it's the door for a large tensor without `alloc`.
    pub fn load_slice(self: &mut Self, data: &[Scalar]) -> Result<(), LenMismatch> {
        if data.len() != NUMEL {
            return Err(LenMismatch);
        }
        self.data.as_flat_mut().copy_from_slice(data);
        Ok(())
    }
    /// Builds a tensor straight from a `Vec` in one step — the `.npy`
    /// pipeline's door, without the caller needing a `mut` local for the
    /// zeroed()+load_vec() two-step.
    #[cfg(feature = "alloc")]
    pub fn from_vec(data: alloc::vec::Vec<Scalar>) -> Result<Self, alloc::vec::Vec<Scalar>>
    where
        S: OwnedStorage<[Scalar; NUMEL]>,
    {
        let mut t = Self::zeroed();
        t.load_vec(data)?;
        Ok(t)
    }
    pub fn get_data(&self) -> &[Scalar] {
        self.data.as_flat()
    }

    pub fn get_shape(&self) -> &[usize; 4] {
        &self.shape
    }

    pub fn get(self: &Self, b: usize, c: usize, i: usize, j: usize) -> Scalar {
        debug_assert!(
            b < self.shape[0] && c < self.shape[1] && i < self.shape[2] && j < self.shape[3]
        );
        let flat_index: usize = b * self.batch_stride
            + c * self.channel_stride
            + i * self.row_stride
            + j * self.col_stride;
        self.data[flat_index]
    }
    pub fn set(self: &mut Self, b: usize, c: usize, i: usize, j: usize, value: Scalar) -> () {
        debug_assert!(
            b < self.shape[0] && c < self.shape[1] && i < self.shape[2] && j < self.shape[3]
        );
        let flat_index: usize = b * self.batch_stride
            + c * self.channel_stride
            + i * self.row_stride
            + j * self.col_stride;
        self.data[flat_index] = value;
    }
    /// # Safety
    /// The caller guarantees b < self.shape[0], c < self.shape[1], i < self.shape[2],
    /// j < self.shape[3].
    pub unsafe fn get_unchecked(self: &Self, b: usize, c: usize, i: usize, j: usize) -> Scalar {
        let flat_index: usize = b * self.batch_stride
            + c * self.channel_stride
            + i * self.row_stride
            + j * self.col_stride;
        *self.data.get_unchecked(flat_index)
    }
    /// # Safety
    /// The caller guarantees b < self.shape[0], c < self.shape[1], i < self.shape[2],
    /// j < self.shape[3].
    pub unsafe fn set_unchecked(
        self: &mut Self,
        b: usize,
        c: usize,
        i: usize,
        j: usize,
        value: Scalar,
    ) -> () {
        let flat_index: usize = b * self.batch_stride
            + c * self.channel_stride
            + i * self.row_stride
            + j * self.col_stride;
        *self.data.get_unchecked_mut(flat_index) = value;
    }
    /// The flat, untransformed backing buffer. The last axis (COLS) always has
    /// stride 1, so a caller that wants to walk it can index this buffer
    /// directly with `row_offset(..) + j` instead of paying for a full
    /// `get_unchecked` (which recomputes every stride term) on each step.
    pub fn get_raw_buffer(&self) -> &[Scalar] {
        self.data.as_flat()
    }
    /// Flat offset of (b, c, i, 0) into `get_raw_buffer()` — the start of the
    /// contiguous row along the last axis.
    /// # Safety
    /// The caller guarantees b < self.shape[0], c < self.shape[1], i < self.shape[2].
    pub unsafe fn row_offset(self: &Self, b: usize, c: usize, i: usize) -> usize {
        b * self.batch_stride + c * self.channel_stride + i * self.row_stride
    }
    pub fn load_data(self: &mut Self, data: [Scalar; NUMEL]) -> () {
        *self.data = data
    }
    /// Loads a dynamically-sized buffer, never materializing
    /// `[Scalar; NUMEL]` on the stack — the entry point for large tensors.
    ///
    /// Hands the `Vec` back intact if its length doesn't match `NUMEL`,
    /// rather than dumping it into a panic message.
    #[cfg(feature = "alloc")]
    pub fn load_vec(
        self: &mut Self,
        data: alloc::vec::Vec<Scalar>,
    ) -> Result<(), alloc::vec::Vec<Scalar>> {
        if data.len() != NUMEL {
            return Err(data);
        }
        self.data.as_flat_mut().copy_from_slice(&data);
        Ok(())
    }
    /// Builds the im2col view of this (N x C x H x W) tensor for a KH x KW
    /// window sliding by `stride`, without copying any data.
    ///
    /// `H_OUT` and `W_OUT` must be passed explicitly — deriving them from
    /// `stride` would need `generic_const_exprs` — and are checked here against
    /// the usual convolution output size, `(H - KH) / stride + 1`.
    pub fn im2col_view<
        'a,
        const H_OUT: usize,
        const W_OUT: usize,
        const KH: usize,
        const KW: usize,
    >(
        self: &'a Self,
        stride: usize,
    ) -> TensorView6D<'a, BATCHES, CHANNELS, ROWS, COLS, H_OUT, W_OUT, KH, KW> {
        assert!(stride >= 1 && KH >= 1 && KW >= 1 && KH <= ROWS && KW <= COLS);
        assert!(H_OUT == (ROWS - KH) / stride + 1 && W_OUT == (COLS - KW) / stride + 1);
        TensorView6D {
            data: self.data.as_flat(),
            reference_index: 0,
            n_stride: self.batch_stride,
            // moving one output pixel slides the window by `stride` input pixels
            h_out_stride: stride * self.row_stride,
            w_out_stride: stride * self.col_stride,
            channel_stride: self.channel_stride,
            // inside a window we walk the input row by row, element by element
            kh_stride: self.row_stride,
            kw_stride: self.col_stride,
            shape: [BATCHES, H_OUT, W_OUT, CHANNELS, KH, KW],
        }
    }
}
