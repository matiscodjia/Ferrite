use crate::scalar::Scalar;

pub struct Tensor3D<const CHANNELS: usize, const ROWS: usize, const COLS: usize, const NUMEL: usize>
{
    data: [Scalar; NUMEL],
    channel_stride: usize,
    row_stride: usize,
    col_stride: usize,
    pub(super) shape: [usize; 3],
}

impl<const CHANNELS: usize, const ROWS: usize, const COLS: usize, const NUMEL: usize>
    Tensor3D<CHANNELS, ROWS, COLS, NUMEL>
{
    const STRUCTURE_CORRECTNESS: () = assert!(NUMEL == CHANNELS * ROWS * COLS);

    pub fn new() -> Self {
        let _ = Self::STRUCTURE_CORRECTNESS;
        Self {
            data: [0.0; NUMEL],
            channel_stride: ROWS * COLS,
            row_stride: COLS,
            col_stride: 1,
            shape: [CHANNELS, ROWS, COLS],
        }
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
        &self.data
    }
    /// Flat offset of (c, i, 0) into `get_raw_buffer()` — the start of the
    /// contiguous row along the last axis.
    /// # Safety
    /// The caller guarantees c < self.shape[0], i < self.shape[1].
    pub unsafe fn row_offset(self: &Self, c: usize, i: usize) -> usize {
        c * self.channel_stride + i * self.row_stride
    }
    pub fn load_data(self: &mut Self, data: [Scalar; NUMEL]) -> () {
        self.data = data
    }
}
