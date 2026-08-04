use crate::scalar::Scalar;

#[derive(Clone, Copy, Debug)]
#[allow(unused_variables)]
pub struct Tensor<const ROWS: usize, const COLS: usize, const NUMEL: usize> {
    data: [Scalar; NUMEL],
    row_stride: usize,
    col_stride: usize,
    pub(super) shape: (usize, usize),
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
        debug_assert!(i < self.shape.0 && j < self.shape.1);
        let flat_index: usize = i * self.row_stride + j * self.col_stride;
        let index: usize = flat_index + self.reference_index;
        self.data[index]
    }
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
        debug_assert!(i < self.shape.0 && j < self.shape.1);
        let flat_index: usize = i * self.row_stride + j * self.col_stride;
        self.data[flat_index]
    }
    pub fn set(self: &mut Self, i: usize, j: usize, value: Scalar) -> () {
        debug_assert!(i < self.shape.0 && j < self.shape.1);
        let flat_index: usize = i * self.row_stride + j * self.col_stride;
        self.data[flat_index] = value;
    }
    /// # Safety
    /// The caller guarantees i < self.shape.0, j < self.shape.1.
    pub unsafe fn get_unchecked(self: &Self, i: usize, j: usize) -> Scalar {
        let flat_index: usize = i * self.row_stride + j * self.col_stride;
        *self.data.get_unchecked(flat_index)
    }
    /// # Safety
    /// The caller guarantees i < self.shape.0, j < self.shape.1.
    pub unsafe fn set_unchecked(self: &mut Self, i: usize, j: usize, value: Scalar) -> () {
        let flat_index: usize = i * self.row_stride + j * self.col_stride;
        *self.data.get_unchecked_mut(flat_index) = value;
    }
    /// The flat, untransformed backing buffer. The last axis (COLS) always has
    /// stride 1, so a caller that wants to walk it can index this buffer
    /// directly with `row_offset(..) + j` instead of paying for a full
    /// `get_unchecked` (which recomputes every stride term) on each step.
    pub fn get_raw_buffer(&self) -> &[Scalar] {
        &self.data
    }
    /// Flat offset of (i, 0) into `get_raw_buffer()` — the start of the
    /// contiguous row along the last axis.
    /// # Safety
    /// The caller guarantees i < self.shape.0.
    pub unsafe fn row_offset(self: &Self, i: usize) -> usize {
        i * self.row_stride
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
