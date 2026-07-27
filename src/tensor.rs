use crate::scalar::{fabs, Scalar};

#[derive(Clone, Copy, Debug)]
#[allow(unused_variables)]
pub struct Tensor<const ROWS: usize, const COLS: usize, const NUMEL: usize> {
    data: [Scalar; NUMEL],
    row_stride: usize,
    col_stride: usize,
    shape: (usize, usize),
}

pub struct TensorView<'a> {
    data: &'a [Scalar],
    reference_index: usize,
    row_stride: usize,
    col_stride: usize,
    shape: (usize, usize),
}
impl<'a> TensorView<'a> {
    pub fn get(self: &Self, i: usize, j: usize) -> Scalar {
        assert!(i < self.shape.0 && j < self.shape.1);
        let flat_index: usize = i * self.row_stride + j * self.col_stride;

        let index: usize = flat_index + self.reference_index;
        assert!(index < self.data.len());
        self.data[index]
    }
}

impl<const ROWS: usize, const COLS: usize, const NUMEL: usize> Tensor<ROWS, COLS, NUMEL> {
    const STRUCTURE_CORRECTNESS: () = assert!(NUMEL == ROWS * COLS);

    pub fn new() -> Self {
        Self::STRUCTURE_CORRECTNESS;
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
    pub fn load_data(self: &mut Self, data: [Scalar; NUMEL]) -> () {
        self.data = data
    }
    pub fn transpose(self: &mut Self) -> () {
        self.shape = (self.shape.1, self.shape.0);
        let rs = self.row_stride;
        self.row_stride = self.col_stride;
        self.col_stride = rs;
    }
    pub fn view<'a>(
        self: &'a Self,
        lines: (usize, usize),
        columns: (usize, usize),
    ) -> TensorView<'a> {
        let max_line = lines.1;
        let max_col = columns.1;
        assert!(max_line < ROWS && max_col < COLS && lines.0 <= lines.1 && columns.0 <= columns.1);
        let reference_index = lines.0 * self.row_stride + columns.0 * self.col_stride;
        let view_shape = (lines.1 - lines.0 + 1, columns.1 - columns.0 + 1);
        TensorView {
            data: &self.data,
            reference_index,
            row_stride: self.row_stride,
            col_stride: self.col_stride,
            shape: view_shape,
        }
    }
}
