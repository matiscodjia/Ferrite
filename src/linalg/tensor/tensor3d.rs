use crate::linalg::storage::{Buffer, LenMismatch, OwnedStorage, StackStorage, Storage};
use crate::scalar::Scalar;

pub struct Tensor3D<
    const CHANNELS: usize,
    const ROWS: usize,
    const COLS: usize,
    const NUMEL: usize,
    S: Storage<[Scalar; NUMEL]> = StackStorage<[Scalar; NUMEL]>,
> {
    data: S,
    channel_stride: usize,
    row_stride: usize,
    col_stride: usize,
    pub(super) shape: [usize; 3],
}

impl<
        const CHANNELS: usize,
        const ROWS: usize,
        const COLS: usize,
        const NUMEL: usize,
        S: Storage<[Scalar; NUMEL]>,
    > Tensor3D<CHANNELS, ROWS, COLS, NUMEL, S>
{
    const STRUCTURE_CORRECTNESS: () = assert!(NUMEL == CHANNELS * ROWS * COLS);

    /// Zero-initialized accumulator for the crate's own algorithms (e.g.
    /// `sp::kernels::replicate`), which build a result element by element.
    /// Kept out of the public API on purpose: an external caller should never
    /// get a silently-zeroed tensor by omitting a load step.
    pub(crate) fn zeroed() -> Self
    where
        S: OwnedStorage<[Scalar; NUMEL]>,
    {
        let _ = Self::STRUCTURE_CORRECTNESS;
        Self {
            data: S::zeroed(),
            channel_stride: ROWS * COLS,
            row_stride: COLS,
            col_stride: 1,
            shape: [CHANNELS, ROWS, COLS],
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
    pub fn get(self: &Self, c: usize, i: usize, j: usize) -> Scalar {
        debug_assert!(c < self.shape[0] && i < self.shape[1] && j < self.shape[2]);
        let flat_index: usize = c * self.channel_stride + i * self.row_stride + j * self.col_stride;
        self.data[flat_index]
    }
    pub fn set(self: &mut Self, c: usize, i: usize, j: usize, value: Scalar) -> () {
        debug_assert!(c < self.shape[0] && i < self.shape[1] && j < self.shape[2]);
        let flat_index: usize = c * self.channel_stride + i * self.row_stride + j * self.col_stride;
        self.data[flat_index] = value;
    }
    /// # Safety
    /// The caller guarantees c < self.shape[0], i < self.shape[1], j < self.shape[2].
    pub unsafe fn get_unchecked(self: &Self, c: usize, i: usize, j: usize) -> Scalar {
        let flat_index: usize = c * self.channel_stride + i * self.row_stride + j * self.col_stride;
        *self.data.get_unchecked(flat_index)
    }
    /// # Safety
    /// The caller guarantees c < self.shape[0], i < self.shape[1], j < self.shape[2].
    pub unsafe fn set_unchecked(
        self: &mut Self,
        c: usize,
        i: usize,
        j: usize,
        value: Scalar,
    ) -> () {
        let flat_index: usize = c * self.channel_stride + i * self.row_stride + j * self.col_stride;
        *self.data.get_unchecked_mut(flat_index) = value;
    }
    /// The flat, untransformed backing buffer. The last axis (COLS) always has
    /// stride 1, so a caller that wants to walk it can index this buffer
    /// directly with `row_offset(..) + j` instead of paying for a full
    /// `get_unchecked` (which recomputes every stride term) on each step.
    pub fn get_raw_buffer(&self) -> &[Scalar] {
        self.data.as_flat()
    }
    /// Flat offset of (c, i, 0) into `get_raw_buffer()` — the start of the
    /// contiguous row along the last axis.
    /// # Safety
    /// The caller guarantees c < self.shape[0], i < self.shape[1].
    pub unsafe fn row_offset(self: &Self, c: usize, i: usize) -> usize {
        c * self.channel_stride + i * self.row_stride
    }
    /// Loads a full, statically-sized buffer — for small tensors and known
    /// data. For a tensor too big to build `[Scalar; NUMEL]` on the stack, see
    /// [`Self::load_slice`] (`no_std`, no `alloc`) or [`Self::from_vec`].
    pub fn load_data(self: &mut Self, data: [Scalar; NUMEL]) -> () {
        self.data.as_flat_mut().copy_from_slice(&data);
    }
    /// Copies from a runtime-sized slice — never materializes `[Scalar; NUMEL]`
    /// on the stack.
    pub fn load_slice(self: &mut Self, data: &[Scalar]) -> Result<(), LenMismatch> {
        if data.len() != NUMEL {
            return Err(LenMismatch);
        }
        self.data.as_flat_mut().copy_from_slice(data);
        Ok(())
    }
    /// Builds a tensor straight from a `Vec` — no compile-time-sized array
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
