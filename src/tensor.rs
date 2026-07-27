use crate::scalar::Scalar;

#[derive(Clone, Copy, Debug)]
pub struct Tensor<const ROWS: usize, const COLS: usize, const NUMEL: usize> {
    data: [Scalar; NUMEL],
    row_stride: usize,
    col_stride: usize,
    shape: (usize, usize),
}

impl<const ROWS: usize, const COLS: usize, const NUMEL: usize> Tensor<ROWS, COLS, NUMEL> {
    const STRUCTURE_CORRECTNESS: () = assert!(NUMEL == ROWS * COLS);

    pub fn new() -> Self {
        let _ = Self::STRUCTURE_CORRECTNESS;
        Self {
            data: [0.0; NUMEL],
            row_stride: COLS,
            col_stride: 1,
            shape: (ROWS, COLS),
        }
    }

    pub fn get(self: &Self, i: usize, j: usize) -> Scalar {
        let flat_index: usize = i * self.row_stride + j * self.col_stride;
        assert!(flat_index < NUMEL);
        self.data[flat_index]
    }

    pub fn set(self: &mut Self, i: usize, j: usize, value: Scalar) -> () {
        let flat_index: usize = i * self.row_stride + j * self.col_stride;
        assert!(flat_index < NUMEL);
        self.data[flat_index] = value;
    }

    pub fn transpose(self: &mut Self) -> () {
        self.shape = (self.shape.1, self.shape.0);
        let rs = self.row_stride;
        self.row_stride = self.col_stride;
        self.col_stride = rs;
    }
}
