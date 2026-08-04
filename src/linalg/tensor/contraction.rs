use super::tensor2d::Tensor;
use super::tensor3d::Tensor3D;
use super::tensor4d::Tensor4D;
use super::tensor6d::Rank6;
use crate::linalg::storage::{OwnedStorage, Storage};
use crate::scalar::Scalar;

/// Contracts the last axis of `a` with the last axis of `b`:
/// (M x K) . (N x K) -> (M x N).
///
/// The shared dimension K is enforced by the signature, and `NUMEL_C == M * N`
/// by `Tensor::new` — both at compile time. `b`'s contracted axis is last
/// (not first) to match `tensordot_2`/`tensordot_3`'s convention: it's what
/// lets `k` be walked contiguously on both operands below.
pub fn tensordot_1<
    const M: usize,
    const K: usize,
    const N: usize,
    const NUMEL_A: usize,
    const NUMEL_B: usize,
    const NUMEL_C: usize,
>(
    a: &Tensor<M, K, NUMEL_A>,
    b: &Tensor<N, K, NUMEL_B>,
) -> Tensor<M, N, NUMEL_C> {
    assert!(a.shape == (M, K) && b.shape == (N, K));
    let mut c = Tensor::<M, N, NUMEL_C>::new();
    for i in 0..M {
        // SAFETY: i < M is guaranteed by the enclosing for loop's bounds.
        let a_base = unsafe { a.row_offset(i) };
        let a_row = a.get_raw_buffer();
        for j in 0..N {
            // SAFETY: j < N is guaranteed by the enclosing for loop's bounds.
            let b_base = unsafe { b.row_offset(j) };
            let b_row = b.get_raw_buffer();
            let mut sum: Scalar = 0.0;
            for k in 0..K {
                // SAFETY: K has stride 1 on both operands (last axis), so
                // a_base + k and b_base + k are the flat indices of (i, k) and
                // (j, k); k < K is guaranteed by the enclosing for loop's bounds.
                let av = unsafe { *a_row.get_unchecked(a_base + k) };
                let bv = unsafe { *b_row.get_unchecked(b_base + k) };
                sum += av * bv;
            }
            // SAFETY: i < M, j < N are guaranteed by the enclosing for loops'
            // bounds; c was just created with shape (M, N).
            unsafe { c.set_unchecked(i, j, sum) };
        }
    }
    c
}

/// Contracts the last two axes of `a` with the last two axes of `b`:
/// (M x K1 x K2) . (N x K1 x K2) -> (M x N).
///
/// The shared dimensions K1 and K2 are enforced by the signature, and
/// `NUMEL_C == M * N` by `Tensor::new` — both at compile time. `b`'s
/// contracted axes are last (not first), matching `tensordot_3`'s
/// convention: `k2` is walked contiguously on both operands below, same
/// pattern as `tensordot_3`'s `ch`/`p`/`q`, one rank down.
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
    b: &Tensor3D<N, K1, K2, NUMEL_B>,
) -> Tensor<M, N, NUMEL_C> {
    assert!(a.shape == [M, K1, K2] && b.shape == [N, K1, K2]);
    let mut c = Tensor::<M, N, NUMEL_C>::new();
    for i in 0..M {
        for j in 0..N {
            let mut sum: Scalar = 0.0;
            for k1 in 0..K1 {
                // SAFETY: i < M, k1 < K1 (resp. j < N, k1 < K1) are guaranteed
                // by the enclosing for loops' bounds.
                let a_base = unsafe { a.row_offset(i, k1) };
                let a_row = a.get_raw_buffer();
                let b_base = unsafe { b.row_offset(j, k1) };
                let b_row = b.get_raw_buffer();
                for k2 in 0..K2 {
                    // SAFETY: K2 has stride 1 on both operands (last axis), so
                    // a_base + k2 and b_base + k2 are the flat indices of
                    // (i, k1, k2) and (j, k1, k2); k2 < K2 is guaranteed by the
                    // enclosing for loop's bounds.
                    let av = unsafe { *a_row.get_unchecked(a_base + k2) };
                    let bv = unsafe { *b_row.get_unchecked(b_base + k2) };
                    sum += av * bv;
                }
            }
            // SAFETY: i < M, j < N are guaranteed by the enclosing for loops'
            // bounds; c was just created with shape (M, N).
            unsafe { c.set_unchecked(i, j, sum) };
        }
    }
    c
}

/// Contracts the last three axes of `a` with the last three axes of `b`:
/// (D0 x D1 x D2 x C x KH x KW) . (K x C x KH x KW) -> (D0 x D1 x D2 x K).
///
/// `a` is any `Rank6` implementor — this is the shared axis-contraction
/// primitive; see [`crate::sp::cross_correlate2d`] for the im2col
/// cross-correlation (conv2d forward pass) built on top of it.
///
/// The shared dimensions C, KH and KW are enforced by the signature, and
/// `NUMEL_C == D0 * D1 * D2 * K` by `Tensor4D::new` — both at compile time.
#[inline(never)]
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
    SB,
    SC,
>(
    a: &A,
    b: &Tensor4D<K, C, KH, KW, NUMEL_B, SB>,
) -> Tensor4D<N, H_OUT, W_OUT, K, NUMEL_C, SC>
where
    A: Rank6<N, H_OUT, W_OUT, C, KH, KW>,
    SB: Storage<[Scalar; NUMEL_B]>,
    // Le résultat peut être aussi gros que l'entrée (1x718x718x2 ≈ 4 Mo) : son
    // stockage doit pouvoir être choisi, sinon l'overflow revient par la sortie.
    SC: OwnedStorage<[Scalar; NUMEL_C]>,
{
    assert!(a.shape() == [N, H_OUT, W_OUT, C, KH, KW] && b.shape == [K, C, KH, KW]);
    let mut c = Tensor4D::<N, H_OUT, W_OUT, K, NUMEL_C, SC>::new();
    for n in 0..N {
        for i in 0..H_OUT {
            for j in 0..W_OUT {
                let mut sums: [Scalar; K] = [0.0; K];
                for ch in 0..C {
                    for p in 0..KH {
                        // SAFETY: n < N, i < H_OUT, j < W_OUT, ch < C, p < KH sont garantis
                        // par les bornes des boucles for englobantes.
                        let a_base = unsafe { a.row_offset(n, i, j, ch, p) };
                        let a_row = a.get_raw_buffer();
                        for q in 0..KW {
                            // SAFETY: KW a un stride de 1 dans tout Rank6 (dernier axe
                            // toujours contigu, par construction de row_offset/get_raw_buffer),
                            // donc a_base + q est l'indice plat de (n, i, j, ch, p, q) ; q < KW
                            // est garanti par la boucle for englobante.
                            let av = unsafe { *a_row.get_unchecked(a_base + q) };
                            for k in 0..K {
                                // SAFETY: k < K, ch < C, p < KH, q < KW sont garantis par les
                                // bornes des boucles for englobantes, et b.shape == [K, C, KH, KW]
                                // est vérifié par l'assert! en tête de fonction.
                                sums[k] += av * unsafe { b.get_unchecked(k, ch, p, q) };
                            }
                        }
                    }
                }
                for k in 0..K {
                    // SAFETY: n < N, i < H_OUT, j < W_OUT, k < K sont garantis par les bornes
                    // des boucles for englobantes ; c vient d'être créé avec shape [N, H_OUT,
                    // W_OUT, K].
                    unsafe { c.set_unchecked(n, i, j, k, sums[k]) };
                }
            }
        }
    }
    c
}
