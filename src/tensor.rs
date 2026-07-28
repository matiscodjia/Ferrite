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

/// Contracts the last axis of `a` with the first axis of `b`:
/// (M x K) . (K x N) -> (M x N).
///
/// The shared dimension K is enforced by the signature, and `NUMEL_C == M * N`
/// by `Tensor::new` — both at compile time.
pub fn tensordot_1<
    const M: usize,
    const K: usize,
    const N: usize,
    const NUMEL_A: usize,
    const NUMEL_B: usize,
    const NUMEL_C: usize,
>(
    a: &Tensor<M, K, NUMEL_A>,
    b: &Tensor<K, N, NUMEL_B>,
) -> Tensor<M, N, NUMEL_C> {
    assert!(a.shape == (M, K) && b.shape == (K, N));
    let mut c = Tensor::<M, N, NUMEL_C>::new();
    for i in 0..M {
        for j in 0..N {
            let mut sum: Scalar = 0.0;
            for k in 0..K {
                sum += a.get(i, k) * b.get(k, j);
            }
            c.set(i, j, sum);
        }
    }
    c
}

/// Contracts the last two axes of `a` with the first two axes of `b`:
/// (M x K1 x K2) . (K1 x K2 x N) -> (M x N).
///
/// The shared dimensions K1 and K2 are enforced by the signature, and
/// `NUMEL_C == M * N` by `Tensor::new` — both at compile time.
pub fn tensordot_2<
    const M: usize,
    const K1: usize,
    const K2: usize,
    const N: usize,
    const NUMEL_A: usize,
    const NUMEL_B: usize,
    const NUMEL_C: usize,
>(
    a: &Tensor3D<M, K1, K2, NUMEL_A>,
    b: &Tensor3D<K1, K2, N, NUMEL_B>,
) -> Tensor<M, N, NUMEL_C> {
    assert!(a.shape == [M, K1, K2] && b.shape == [K1, K2, N]);
    let mut c = Tensor::<M, N, NUMEL_C>::new();
    for i in 0..M {
        for j in 0..N {
            let mut sum: Scalar = 0.0;
            for k in 0..K1 {
                for p in 0..K2 {
                    sum += a.get(i, k, p) * b.get(k, p, j);
                }
            }
            c.set(i, j, sum);
        }
    }
    c
}

/// Read-only access to a rank-6 tensor, whichever way it holds its elements:
/// `Tensor6D` owns them, `TensorView6D` only aliases someone else's buffer.
///
/// The six dimensions are const parameters rather than runtime values, so a
/// contraction over an implementor keeps checking its shared axes at compile
/// time — the trait erases ownership, not shape.
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
    > Rank6<BATCHES, GROUPS, CHANNELS, DEPTH, ROWS, COLS>
    for Tensor6D<BATCHES, GROUPS, CHANNELS, DEPTH, ROWS, COLS, NUMEL>
{
    fn get(self: &Self, b: usize, g: usize, c: usize, d: usize, i: usize, j: usize) -> Scalar {
        Tensor6D::get(self, b, g, c, d, i, j)
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
    fn shape(self: &Self) -> [usize; 6] {
        self.shape
    }
}

/// Contracts the last three axes of `a` with the last three axes of `b`:
/// (N x H_OUT x W_OUT x C x KH x KW) . (K x C x KH x KW) -> (N x H_OUT x W_OUT x K).
///
/// This is the conv2d contraction: `a` holds one receptive field (C x KH x KW)
/// per output pixel, `b` holds the K filters. `a` is any `Rank6`, so an
/// im2col `TensorView6D` contracts straight from the input buffer — no
/// materialised patch tensor, hence no N * H_OUT * W_OUT * C * KH * KW copy.
///
/// The shared dimensions C, KH and KW are enforced by the signature, and
/// `NUMEL_C == N * H_OUT * W_OUT * K` by `Tensor4D::new` — both at compile time.
pub fn tensordot_3<
    A,
    const N: usize,
    const H_OUT: usize,
    const W_OUT: usize,
    const C: usize,
    const KH: usize,
    const KW: usize,
    const K: usize,
    const NUMEL_B: usize,
    const NUMEL_C: usize,
>(
    a: &A,
    b: &Tensor4D<K, C, KH, KW, NUMEL_B>,
) -> Tensor4D<N, H_OUT, W_OUT, K, NUMEL_C>
where
    A: Rank6<N, H_OUT, W_OUT, C, KH, KW>,
{
    assert!(a.shape() == [N, H_OUT, W_OUT, C, KH, KW] && b.shape == [K, C, KH, KW]);
    let mut c = Tensor4D::<N, H_OUT, W_OUT, K, NUMEL_C>::new();
    for n in 0..N {
        for i in 0..H_OUT {
            for j in 0..W_OUT {
                for k in 0..K {
                    let mut sum: Scalar = 0.0;
                    for ch in 0..C {
                        for p in 0..KH {
                            for q in 0..KW {
                                sum += a.get(n, i, j, ch, p, q) * b.get(k, ch, p, q);
                            }
                        }
                    }
                    c.set(n, i, j, k, sum);
                }
            }
        }
    }
    c
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
            data: &self.data,
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
    data: &'a [Scalar],
    reference_index: usize,
    n_stride: usize,
    h_out_stride: usize,
    w_out_stride: usize,
    channel_stride: usize,
    kh_stride: usize,
    kw_stride: usize,
    shape: [usize; 6],
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
        assert!(
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
