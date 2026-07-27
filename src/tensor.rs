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
        assert!(i < self.shape.0 && j < self.shape.1);
        let flat_index: usize = i * self.row_stride + j * self.col_stride;
        self.data[flat_index]
    }
    pub fn set(self: &mut Self, i: usize, j: usize, value: Scalar) -> () {
        assert!(i < self.shape.0 && j < self.shape.1);
        let flat_index: usize = i * self.row_stride + j * self.col_stride;
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

pub struct Tensor3D<const CHANNELS: usize, const ROWS: usize, const COLS: usize, const NUMEL: usize>
{
    data: [Scalar; NUMEL],
    channel_stride: usize,
    row_stride: usize,
    col_stride: usize,
    shape: [usize; 3],
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
        assert!(c < self.shape[0] && i < self.shape[1] && j < self.shape[2]);
        let flat_index: usize = c * self.channel_stride + i * self.row_stride + j * self.col_stride;
        self.data[flat_index]
    }
    pub fn set(self: &mut Self, c: usize, i: usize, j: usize, value: Scalar) -> () {
        assert!(c < self.shape[0] && i < self.shape[1] && j < self.shape[2]);
        let flat_index: usize = c * self.channel_stride + i * self.row_stride + j * self.col_stride;
        self.data[flat_index] = value;
    }
    pub fn load_data(self: &mut Self, data: [Scalar; NUMEL]) -> () {
        self.data = data
    }
}

pub struct Tensor4D<
    const BATCHES: usize,
    const CHANNELS: usize,
    const ROWS: usize,
    const COLS: usize,
    const NUMEL: usize,
> {
    data: [Scalar; NUMEL],
    batch_stride: usize,
    channel_stride: usize,
    row_stride: usize,
    col_stride: usize,
    shape: [usize; 4],
}

impl<
        const BATCHES: usize,
        const CHANNELS: usize,
        const ROWS: usize,
        const COLS: usize,
        const NUMEL: usize,
    > Tensor4D<BATCHES, CHANNELS, ROWS, COLS, NUMEL>
{
    const STRUCTURE_CORRECTNESS: () = assert!(NUMEL == BATCHES * CHANNELS * ROWS * COLS);

    pub fn new() -> Self {
        let _ = Self::STRUCTURE_CORRECTNESS;
        Self {
            data: [0.0; NUMEL],
            batch_stride: CHANNELS * ROWS * COLS,
            channel_stride: ROWS * COLS,
            row_stride: COLS,
            col_stride: 1,
            shape: [BATCHES, CHANNELS, ROWS, COLS],
        }
    }
    pub fn get(self: &Self, b: usize, c: usize, i: usize, j: usize) -> Scalar {
        assert!(b < self.shape[0] && c < self.shape[1] && i < self.shape[2] && j < self.shape[3]);
        let flat_index: usize = b * self.batch_stride
            + c * self.channel_stride
            + i * self.row_stride
            + j * self.col_stride;
        self.data[flat_index]
    }
    pub fn set(self: &mut Self, b: usize, c: usize, i: usize, j: usize, value: Scalar) -> () {
        assert!(b < self.shape[0] && c < self.shape[1] && i < self.shape[2] && j < self.shape[3]);
        let flat_index: usize = b * self.batch_stride
            + c * self.channel_stride
            + i * self.row_stride
            + j * self.col_stride;
        self.data[flat_index] = value;
    }
    pub fn load_data(self: &mut Self, data: [Scalar; NUMEL]) -> () {
        self.data = data
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
> {
    data: [Scalar; NUMEL],
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
    > Tensor6D<BATCHES, GROUPS, CHANNELS, DEPTH, ROWS, COLS, NUMEL>
{
    const STRUCTURE_CORRECTNESS: () =
        assert!(NUMEL == BATCHES * GROUPS * CHANNELS * DEPTH * ROWS * COLS);

    pub fn new() -> Self {
        let _ = Self::STRUCTURE_CORRECTNESS;
        Self {
            data: [0.0; NUMEL],
            batch_stride: GROUPS * CHANNELS * DEPTH * ROWS * COLS,
            group_stride: CHANNELS * DEPTH * ROWS * COLS,
            channel_stride: DEPTH * ROWS * COLS,
            depth_stride: ROWS * COLS,
            row_stride: COLS,
            col_stride: 1,
            shape: [BATCHES, GROUPS, CHANNELS, DEPTH, ROWS, COLS],
        }
    }
    pub fn get(self: &Self, b: usize, g: usize, c: usize, d: usize, i: usize, j: usize) -> Scalar {
        assert!(
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
        assert!(
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
    pub fn load_data(self: &mut Self, data: [Scalar; NUMEL]) -> () {
        self.data = data
    }
}
