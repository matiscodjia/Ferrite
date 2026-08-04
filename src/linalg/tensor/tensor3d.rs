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
    pub fn load_data(self: &mut Self, data: [Scalar; NUMEL]) -> () {
        self.data = data
    }
}
