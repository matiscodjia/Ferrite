use crate::linalg::storage::{Buffer, LenMismatch, OwnedStorage, StackStorage, Storage};
use crate::scalar::{fabs, sqrt, Scalar};
use core::ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug)]
#[allow(unused_variables)]
pub struct Tensor<
    const ROWS: usize,
    const COLS: usize,
    const NUMEL: usize,
    S: Storage<[Scalar; NUMEL]> = StackStorage<[Scalar; NUMEL]>,
> {
    data: S,
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

impl<const ROWS: usize, const COLS: usize, const NUMEL: usize, S: Storage<[Scalar; NUMEL]>>
    Tensor<ROWS, COLS, NUMEL, S>
{
    const STRUCTURE_CORRECTNESS: () = assert!(NUMEL == ROWS * COLS);

    /// Zero-initialized accumulator for the crate's own algorithms (`identity`,
    /// `multiply`, `qr_decomposition`, `tensordot_*`...), which build a result
    /// element by element and can't hand `new` the full data upfront. Kept out
    /// of the public API on purpose: an external caller of the crate should
    /// never be able to get a silently-zeroed tensor by omitting a load step.
    pub(crate) fn zeroed() -> Self
    where
        S: OwnedStorage<[Scalar; NUMEL]>,
    {
        let _ = Self::STRUCTURE_CORRECTNESS;
        Self {
            data: S::zeroed(),
            row_stride: COLS,
            col_stride: 1,
            shape: (ROWS, COLS),
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
        self.data.as_flat()
    }
    /// Flat offset of (i, 0) into `get_raw_buffer()` — the start of the
    /// contiguous row along the last axis.
    /// # Safety
    /// The caller guarantees i < self.shape.0.
    pub unsafe fn row_offset(self: &Self, i: usize) -> usize {
        i * self.row_stride
    }
    /// Loads a full, statically-sized buffer — for small tensors and known
    /// data. For a tensor too big to build `[Scalar; NUMEL]` on the stack, see
    /// [`Self::load_slice`] (`no_std`, no `alloc`) or [`Self::from_vec`].
    pub fn load_data(self: &mut Self, data: [Scalar; NUMEL]) -> () {
        self.data.as_flat_mut().copy_from_slice(&data);
    }
    /// Copies from a runtime-sized slice — never materializes `[Scalar; NUMEL]`
    /// on the stack, so it's the door for a large tensor without `alloc`
    /// (a table already sitting in flash or in a driver-filled buffer).
    pub fn load_slice(self: &mut Self, data: &[Scalar]) -> Result<(), LenMismatch> {
        if data.len() != NUMEL {
            return Err(LenMismatch);
        }
        self.data.as_flat_mut().copy_from_slice(data);
        Ok(())
    }
    /// Builds a tensor straight from a `Vec` — the door for the `.npy`
    /// pipeline, where the data doesn't exist as a compile-time-sized array in
    /// the first place.
    ///
    /// Rends le `Vec` intact si sa longueur ne correspond pas à `NUMEL`, plutôt
    /// que de le déverser dans un message de panique.
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
            data: self.data.as_flat(),
            reference_index,
            row_stride: self.row_stride,
            col_stride: self.col_stride,
            shape: view_shape,
        }
    }
    pub const fn rows(&self) -> usize {
        ROWS
    }
    pub const fn cols(&self) -> usize {
        COLS
    }
    /// Extracts a column as a `Tensor<ROWS, 1, ROWS>` (a [`Vector`]).
    pub fn get_col(&self, col: usize) -> Option<Tensor<ROWS, 1, ROWS>> {
        if col >= COLS {
            return None;
        }
        let mut result = Tensor::<ROWS, 1, ROWS>::zeroed();
        for i in 0..ROWS {
            result.set(i, 0, self.get(i, col));
        }
        Some(result)
    }
    /// Injects a `Tensor<ROWS, 1, ROWS>` (a [`Vector`]) into a matrix column.
    /// # Panics
    /// Panics if `col >= COLS`.
    pub fn set_col(&mut self, col: usize, vec: &Tensor<ROWS, 1, ROWS>) {
        assert!(col < COLS, "Column index out of bounds");
        for i in 0..ROWS {
            self.set(i, col, vec.get(i, 0));
        }
    }
    /// Builds a tensor from an array of column vectors.
    pub fn from_cols(cols: [Tensor<ROWS, 1, ROWS>; COLS]) -> Self
    where
        S: OwnedStorage<[Scalar; NUMEL]>,
    {
        let mut mat = Self::zeroed();
        for j in 0..COLS {
            mat.set_col(j, &cols[j]);
        }
        mat
    }
    /// Returns the transpose as a new tensor — unlike [`Self::transpose`]
    /// (in-place, same static shape), this changes the static shape from
    /// `(ROWS, COLS)` to `(COLS, ROWS)`.
    pub fn transposed(&self) -> Tensor<COLS, ROWS, NUMEL, S>
    where
        S: OwnedStorage<[Scalar; NUMEL]>,
    {
        let mut result = Tensor::<COLS, ROWS, NUMEL, S>::zeroed();
        for i in 0..ROWS {
            for j in 0..COLS {
                result.set(j, i, self.get(i, j));
            }
        }
        result
    }
    /// Accumulates the product of `a * b` into `self`: `self += a * b`.
    pub fn matmul_accumulate<const K: usize, const NUMEL_A: usize, const NUMEL_B: usize>(
        &mut self,
        a: &Tensor<ROWS, K, NUMEL_A>,
        b: &Tensor<K, COLS, NUMEL_B>,
    ) {
        for i in 0..ROWS {
            for j in 0..COLS {
                let mut sum: Scalar = 0.0;
                for k in 0..K {
                    sum += a.get(i, k) * b.get(k, j);
                }
                let prev = self.get(i, j);
                self.set(i, j, prev + sum);
            }
        }
    }
    /// Matrix product: `self` (ROWS x COLS) * `other` (COLS x P) -> (ROWS x P).
    /// Also serves as matrix-vector product once `Vector<N> = Tensor<N, 1, N>`.
    ///
    /// A method rather than `Mul`: the output's `NUMEL_C` can't be derived
    /// from `ROWS * P` without the unstable `generic_const_exprs`, and a
    /// free const generic on a trait impl (unlike on a plain function/method)
    /// must be constrained by `Self`/`Rhs`, so it can't be inferred from
    /// `Output` alone (`E0207`).
    pub fn multiply<const P: usize, const NUMEL_B: usize, const NUMEL_C: usize>(
        &self,
        other: &Tensor<COLS, P, NUMEL_B>,
    ) -> Tensor<ROWS, P, NUMEL_C> {
        let mut result = Tensor::<ROWS, P, NUMEL_C>::zeroed();
        for i in 0..ROWS {
            for j in 0..P {
                let mut sum: Scalar = 0.0;
                for k in 0..COLS {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }
}

impl<const SIZE: usize, const NUMEL: usize, S: Storage<[Scalar; NUMEL]>>
    Tensor<SIZE, SIZE, NUMEL, S>
{
    pub fn identity() -> Self
    where
        S: OwnedStorage<[Scalar; NUMEL]>,
    {
        let mut result = Self::zeroed();
        for i in 0..SIZE {
            result.set(i, i, 1.0);
        }
        result
    }
}

/// The column-vector shape: a "vector" is just a `Tensor` with one column.
/// See the [`Vector`] alias.
impl<const N: usize, const NUMEL: usize, S: Storage<[Scalar; NUMEL]>> Tensor<N, 1, NUMEL, S> {
    /// Builds a column tensor from a flat array — the `Vector::new(data)`
    /// equivalent (can't reuse the name `new`, already taken by the general
    /// impl's data constructor with a slightly different signature).
    pub fn from_data(data: [Scalar; N]) -> Self
    where
        S: OwnedStorage<[Scalar; NUMEL]>,
    {
        let mut t = Self::zeroed();
        for i in 0..N {
            t.set(i, 0, data[i]);
        }
        t
    }
    /// Returns the dimension of the vector (`Vector::dim`'s equivalent).
    pub const fn dim(&self) -> usize {
        N
    }
    pub fn dot(&self, other: &Self) -> Scalar {
        let mut sum: Scalar = 0.0;
        for i in 0..N {
            sum += self.get(i, 0) * other.get(i, 0);
        }
        sum
    }
    pub fn l2_norm(&self) -> Scalar {
        sqrt(self.dot(self))
    }
    pub fn l1_norm(&self) -> Scalar {
        let mut sum: Scalar = 0.0;
        for i in 0..N {
            sum += fabs(self.get(i, 0));
        }
        sum
    }
    pub fn inf_norm(&self) -> Scalar {
        let mut max: Scalar = 0.0;
        for i in 0..N {
            let abs_val = fabs(self.get(i, 0));
            if abs_val > max {
                max = abs_val;
            }
        }
        max
    }
    pub fn orthogonal_projection(&self, other: &Self) -> Self
    where
        S: OwnedStorage<[Scalar; NUMEL]> + Copy,
    {
        let scale_factor = other.dot(other);
        if fabs(scale_factor) < 1e-8 {
            return Self::zeroed();
        }
        let ratio = self.dot(other) / scale_factor;
        *other * ratio
    }
    pub fn sum(&self) -> Scalar {
        let mut s: Scalar = 0.0;
        for i in 0..N {
            s += self.get(i, 0);
        }
        s
    }
    pub fn hadamard(&self, other: &Self) -> Self
    where
        S: OwnedStorage<[Scalar; NUMEL]>,
    {
        let mut result = Self::zeroed();
        for i in 0..N {
            result.set(i, 0, self.get(i, 0) * other.get(i, 0));
        }
        result
    }
}

impl<const N: usize, const NUMEL: usize, S: Storage<[Scalar; NUMEL]>> Index<usize>
    for Tensor<N, 1, NUMEL, S>
{
    type Output = Scalar;
    fn index(&self, i: usize) -> &Self::Output {
        &self.data[i * self.row_stride]
    }
}
impl<const N: usize, const NUMEL: usize, S: Storage<[Scalar; NUMEL]>> IndexMut<usize>
    for Tensor<N, 1, NUMEL, S>
{
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.data[i * self.row_stride]
    }
}

impl<const ROWS: usize, const COLS: usize, const NUMEL: usize, S: Storage<[Scalar; NUMEL]>>
    Index<(usize, usize)> for Tensor<ROWS, COLS, NUMEL, S>
{
    type Output = Scalar;
    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self.data[i * self.row_stride + j * self.col_stride]
    }
}
impl<const ROWS: usize, const COLS: usize, const NUMEL: usize, S: Storage<[Scalar; NUMEL]>>
    IndexMut<(usize, usize)> for Tensor<ROWS, COLS, NUMEL, S>
{
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        &mut self.data[i * self.row_stride + j * self.col_stride]
    }
}

impl<
        const ROWS: usize,
        const COLS: usize,
        const NUMEL: usize,
        S: OwnedStorage<[Scalar; NUMEL]>,
    > Add for Tensor<ROWS, COLS, NUMEL, S>
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let mut result = Self::zeroed();
        for k in 0..NUMEL {
            result.data[k] = self.data[k] + rhs.data[k];
        }
        result
    }
}
impl<
        const ROWS: usize,
        const COLS: usize,
        const NUMEL: usize,
        S: OwnedStorage<[Scalar; NUMEL]>,
    > Sub for Tensor<ROWS, COLS, NUMEL, S>
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut result = Self::zeroed();
        for k in 0..NUMEL {
            result.data[k] = self.data[k] - rhs.data[k];
        }
        result
    }
}
impl<
        const ROWS: usize,
        const COLS: usize,
        const NUMEL: usize,
        S: OwnedStorage<[Scalar; NUMEL]>,
    > Neg for Tensor<ROWS, COLS, NUMEL, S>
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        self * -1.0
    }
}
impl<
        const ROWS: usize,
        const COLS: usize,
        const NUMEL: usize,
        S: OwnedStorage<[Scalar; NUMEL]>,
    > Mul<Scalar> for Tensor<ROWS, COLS, NUMEL, S>
{
    type Output = Self;
    fn mul(self, rhs: Scalar) -> Self::Output {
        let mut result = Self::zeroed();
        for k in 0..NUMEL {
            result.data[k] = self.data[k] * rhs;
        }
        result
    }
}
impl<
        const ROWS: usize,
        const COLS: usize,
        const NUMEL: usize,
        S: OwnedStorage<[Scalar; NUMEL]>,
    > Div<Scalar> for Tensor<ROWS, COLS, NUMEL, S>
{
    type Output = Self;
    fn div(self, rhs: Scalar) -> Self::Output {
        self * (1.0 / rhs)
    }
}

impl<const ROWS: usize, const COLS: usize, const NUMEL: usize, S: Storage<[Scalar; NUMEL]>>
    PartialEq for Tensor<ROWS, COLS, NUMEL, S>
{
    fn eq(&self, other: &Self) -> bool {
        let epsilon = 1e-5;
        for k in 0..NUMEL {
            if fabs(self.data[k] - other.data[k]) >= epsilon {
                return false;
            }
        }
        true
    }
}

/// A vector is just a `Tensor` with one column — this alias is the only
/// thing that distinguishes it, `Tensor` is the crate's one elementary
/// structure.
pub type Vector<const N: usize> = Tensor<N, 1, N>;
