use super::tensor2d::Tensor;
use super::tensor3d::Tensor3D;
use super::tensor4d::Tensor4D;
use super::tensor6d::Rank6;
use crate::linalg::storage::{OwnedStorage, Storage};
use crate::scalar::Scalar;

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
