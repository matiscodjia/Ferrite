use ferrite::linalg::tensor::tensordot_1;
use ferrite::linalg::tensor::tensordot_2;
use ferrite::linalg::tensor::tensordot_3;
use ferrite::linalg::tensor::Tensor;
use ferrite::linalg::tensor::Tensor3D;
use ferrite::linalg::tensor::Tensor4D;
use ferrite::linalg::tensor::Tensor6D;
use ferrite::linalg::tensor::Vector;
use ferrite::scalar::Scalar;
#[test]
//This is a compilation level test, test NOK = No compilation
fn test_tensor_creation_and_shape() {
    let _m = Tensor::<2, 3, 6>::new([0.0; 6]);
}

#[test]
fn test_indexing() {
    let m = Tensor::<2, 3, 6>::new([0.0; 6]);
    println!("{}", m.get(0, 0))
}

#[test]
#[should_panic]
fn test_indexing_not_valid() {
    let m = Tensor::<2, 3, 6>::new([0.0; 6]);
    println!("{}", m.get(12, 2))
}

#[test]
fn test_setting() {
    let mut m = Tensor::<2, 3, 6>::new([0.0; 6]);
    m.set(0, 0, 2.0);
    assert_eq!(2.0, m.get(0, 0))
}

#[test]
fn test_transpose() {
    let mut m = Tensor::<2, 3, 6>::new([0.0; 6]);
    m.set(1, 2, 4.0);
    m.transpose();
    assert_eq!(4.0, m.get(2, 1));

    m.transpose();
    assert_eq!(4.0, m.get(1, 2));
}

#[test]
fn test_tensor_view() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let m = Tensor::<3, 3, 9>::new(data);
    let m_view = m.view((1, 2), (1, 2));
    assert_eq!(5.0, m_view.get(0, 0));
    assert_eq!(6.0, m_view.get(0, 1));
    assert_eq!(8.0, m_view.get(1, 0));
    assert_eq!(9.0, m_view.get(1, 1));
}

#[test]
#[should_panic]
fn test_view_indexing_not_valid() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let m = Tensor::<3, 3, 9>::new(data);
    let m_view = m.view((1, 2), (1, 2));
    m_view.get(5, 5);
}
#[test]
#[should_panic]
fn test_indexing_axis_overflow() {
    // 2 lignes, 3 colonnes : la colonne 3 sort de l'axe sans sortir du buffer
    let m = Tensor::<2, 3, 6>::new([0.0; 6]);
    m.get(0, 3);
}

#[test]
#[should_panic]
fn test_setting_axis_overflow() {
    let mut m = Tensor::<2, 3, 6>::new([0.0; 6]);
    m.set(0, 3, 1.0);
}

#[test]
#[should_panic]
fn test_indexing_axis_overflow_transposed() {
    // apres transpose la shape est (3, 2) : la colonne 2 sort de l'axe
    let mut m = Tensor::<2, 3, 6>::new([0.0; 6]);
    m.transpose();
    m.get(0, 2);
}

#[test]
fn test_tensordot() {
    // (2 x 3) . (2 x 3) -> (2 x 2) — b's contracted axis (3) is last, per the
    // shared tensordot_1/2/3 convention.
    let a = Tensor::<2, 3, 6>::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Tensor::<2, 3, 6>::new([7.0, 9.0, 11.0, 8.0, 10.0, 12.0]);

    let c: Tensor<2, 2, 4> = tensordot_1(&a, &b);
    assert_eq!(58.0, c.get(0, 0));
    assert_eq!(64.0, c.get(0, 1));
    assert_eq!(139.0, c.get(1, 0));
    assert_eq!(154.0, c.get(1, 1));
}

#[test]
fn test_tensordot_identity() {
    let a = Tensor::<2, 2, 4>::new([1.0, 2.0, 3.0, 4.0]);
    let id = Tensor::<2, 2, 4>::new([1.0, 0.0, 0.0, 1.0]);

    let c: Tensor<2, 2, 4> = tensordot_1(&a, &id);
    assert_eq!(1.0, c.get(0, 0));
    assert_eq!(2.0, c.get(0, 1));
    assert_eq!(3.0, c.get(1, 0));
    assert_eq!(4.0, c.get(1, 1));
}

#[test]
fn test_tensordot_non_square() {
    // (1 x 3) . (4 x 3) -> (1 x 4) — b's contracted axis (3) is last.
    let a = Tensor::<1, 3, 3>::new([1.0, 2.0, 3.0]);
    let b = Tensor::<4, 3, 12>::new([
        1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
    ]);

    let c: Tensor<1, 4, 4> = tensordot_1(&a, &b);
    assert_eq!(38.0, c.get(0, 0));
    assert_eq!(44.0, c.get(0, 1));
    assert_eq!(50.0, c.get(0, 2));
    assert_eq!(56.0, c.get(0, 3));
}

#[test]
#[should_panic]
fn test_tensordot_shape_out_of_sync_with_type() {
    // transpose ne change que la shape runtime : le type annonce toujours (2, 3)
    let mut a = Tensor::<2, 3, 6>::new([0.0; 6]);
    a.transpose();
    let b = Tensor::<2, 3, 6>::new([0.0; 6]);
    let _c: Tensor<2, 2, 4> = tensordot_1(&a, &b);
}

#[test]
fn test_tensordot_2() {
    // (2 x 2 x 2) . (3 x 2 x 2) -> (2 x 3) — b's contracted axes (2, 2) are
    // last, per the shared tensordot_1/2/3 convention.
    let a = Tensor3D::<2, 2, 2, 8>::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = Tensor3D::<3, 2, 2, 12>::new([
        1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0,
    ]);

    let c: Tensor<2, 3, 6> = tensordot_2(&a, &b);
    assert_eq!(70.0, c.get(0, 0));
    assert_eq!(80.0, c.get(0, 1));
    assert_eq!(90.0, c.get(0, 2));
    assert_eq!(158.0, c.get(1, 0));
    assert_eq!(184.0, c.get(1, 1));
    assert_eq!(210.0, c.get(1, 2));
}

#[test]
fn test_tensordot_2_matches_flattened_tensordot_1() {
    // contracter (K1, K2) revient a contracter un seul axe K1 * K2 sur les
    // memes donnees : c'est l'invariant sur lequel repose l'aplatissement.
    let a_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    // b's contracted axes are last, per the shared tensordot_1/2/3
    // convention.
    let b_data = [
        1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0,
    ];

    let a3 = Tensor3D::<2, 2, 2, 8>::new(a_data);
    let b3 = Tensor3D::<3, 2, 2, 12>::new(b_data);
    let c3: Tensor<2, 3, 6> = tensordot_2(&a3, &b3);

    let a2 = Tensor::<2, 4, 8>::new(a_data);
    let b2 = Tensor::<3, 4, 12>::new(b_data);
    let c2: Tensor<2, 3, 6> = tensordot_1(&a2, &b2);

    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(c2.get(i, j), c3.get(i, j));
        }
    }
}

#[test]
fn test_tensordot_2_single_inner_axis() {
    // K2 = 1 : la contraction sur deux axes degenere en produit matriciel.
    // b's contracted axes (3, 1) are last, per the shared convention.
    let a = Tensor3D::<2, 3, 1, 6>::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Tensor3D::<2, 3, 1, 6>::new([7.0, 9.0, 11.0, 8.0, 10.0, 12.0]);

    let c: Tensor<2, 2, 4> = tensordot_2(&a, &b);
    assert_eq!(58.0, c.get(0, 0));
    assert_eq!(64.0, c.get(0, 1));
    assert_eq!(139.0, c.get(1, 0));
    assert_eq!(154.0, c.get(1, 1));
}

#[test]
fn test_tensordot_3() {
    // (1 x 1 x 2 x 2 x 2 x 2) . (2 x 2 x 2 x 2) -> (1 x 1 x 2 x 2)
    // deux patchs de 8 valeurs, deux filtres de 8 valeurs : le resultat est la
    // matrice de Gram entre patchs et filtres, calculee a la main ci-dessous.
    // patch(0,0) = [1..8], patch(0,1) = [9..16]
    // filtre(0)  = [1..8], filtre(1)  = [9..16]
    let mut data = [0.0; 16];
    for k in 0..16 {
        data[k] = (k + 1) as Scalar;
    }
    let a = Tensor6D::<1, 1, 2, 2, 2, 2, 16>::new(data);
    let b = Tensor4D::<2, 2, 2, 2, 16>::new(data);

    let c: Tensor4D<1, 1, 2, 2, 4> = tensordot_3(&a, &b);
    // 1*1 + 2*2 + ... + 8*8 = 204
    assert_eq!(204.0, c.get(0, 0, 0, 0));
    // 1*9 + 2*10 + ... + 8*16 = 492
    assert_eq!(492.0, c.get(0, 0, 0, 1));
    // 9*1 + 10*2 + ... + 16*8 = 492 : le produit scalaire est symetrique
    assert_eq!(492.0, c.get(0, 0, 1, 0));
    // 9*9 + 10*10 + ... + 16*16 = 1292
    assert_eq!(1292.0, c.get(0, 0, 1, 1));
}

#[test]
fn test_tensordot_3_matches_flattened_tensordot_1() {
    // invariant im2col : contracter (C, KH, KW) revient a un produit matriciel
    // (N * H_out * W_out, C * KH * KW) . (K, C * KH * KW) sur les memes
    // donnees — b's contracted axis is last on both sides, per the shared
    // tensordot_1/2/3 convention. Le buffer de `a` est deja dans le bon
    // ordre (row-major), celui de `b` doit etre transpose puisqu'il est
    // stocke en (K, C, KH, KW).
    const N: usize = 2;
    const H_OUT: usize = 2;
    const W_OUT: usize = 1;
    const K: usize = 3;
    const INNER: usize = 4; // C * KH * KW = 2 * 2 * 1

    let mut a_data = [0.0; 16];
    for k in 0..16 {
        a_data[k] = (k + 1) as Scalar;
    }
    let mut b_data = [0.0; 12];
    for k in 0..12 {
        b_data[k] = (k + 1) as Scalar;
    }

    let a6 = Tensor6D::<N, H_OUT, W_OUT, 2, 2, 1, 16>::new(a_data);
    let b4 = Tensor4D::<K, 2, 2, 1, 12>::new(b_data);
    let c4: Tensor4D<N, H_OUT, W_OUT, K, 12> = tensordot_3(&a6, &b4);

    let a2 = Tensor::<4, INNER, 16>::new(a_data);
    let mut b2 = Tensor::<K, INNER, 12>::new([0.0; 12]);
    for k in 0..K {
        for c in 0..2 {
            for p in 0..2 {
                b2.set(k, c * 2 + p, b4.get(k, c, p, 0));
            }
        }
    }
    let c2: Tensor<4, K, 12> = tensordot_1(&a2, &b2);

    for n in 0..N {
        for i in 0..H_OUT {
            for j in 0..W_OUT {
                for k in 0..K {
                    let row = n * H_OUT * W_OUT + i * W_OUT + j;
                    assert_eq!(c2.get(row, k), c4.get(n, i, j, k));
                }
            }
        }
    }
}

#[test]
fn test_tensordot_3_pointwise_filters() {
    // C = KH = KW = 1 : la contraction degenere en produit exterieur, chaque
    // pixel est simplement multiplie par chacun des K scalaires du filtre.
    let a = Tensor6D::<2, 1, 2, 1, 1, 1, 4>::new([1.0, 2.0, 3.0, 4.0]);
    let b = Tensor4D::<3, 1, 1, 1, 3>::new([5.0, 6.0, 7.0]);

    let c: Tensor4D<2, 1, 2, 3, 12> = tensordot_3(&a, &b);
    for n in 0..2 {
        for j in 0..2 {
            for k in 0..3 {
                let expected = a.get(n, 0, j, 0, 0, 0) * b.get(k, 0, 0, 0);
                assert_eq!(expected, c.get(n, 0, j, k));
            }
        }
    }
    assert_eq!(5.0, c.get(0, 0, 0, 0));
    assert_eq!(28.0, c.get(1, 0, 1, 2));
}

#[test]
fn test_tensor3d_creation_and_shape() {
    let _m = Tensor3D::<2, 2, 2, 8>::new([0.0; 8]);
}

#[test]
fn test_indexing_tensore3d() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let m = Tensor3D::<2, 2, 2, 8>::new(data);
    assert_eq!(1.0, m.get(0, 0, 0));
    assert_eq!(2.0, m.get(0, 0, 1));
    assert_eq!(3.0, m.get(0, 1, 0));
    assert_eq!(4.0, m.get(0, 1, 1));
    assert_eq!(5.0, m.get(1, 0, 0));
    assert_eq!(6.0, m.get(1, 0, 1));
    assert_eq!(7.0, m.get(1, 1, 0));
    assert_eq!(8.0, m.get(1, 1, 1));
}

#[test]
#[should_panic]
fn test_indexing3d_not_valid() {
    let m = Tensor3D::<2, 2, 2, 8>::new([0.0; 8]);
    m.get(12, 2, 0);
}

#[test]
#[should_panic]
fn test_indexing3d_axis_overflow() {
    let m = Tensor3D::<2, 2, 2, 8>::new([0.0; 8]);
    m.get(0, 0, 2);
}

#[test]
fn test_tensor4d_creation_and_shape() {
    let _m = Tensor4D::<2, 2, 2, 2, 16>::new([0.0; 16]);
}

#[test]
fn test_indexing_tensore4d() {
    let data = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let m = Tensor4D::<2, 2, 2, 2, 16>::new(data);
    assert_eq!(1.0, m.get(0, 0, 0, 0));
    assert_eq!(16.0, m.get(1, 1, 1, 1));
    assert_eq!(8.0, m.get(0, 1, 1, 1));
}

#[test]
#[should_panic]
fn test_indexing4d_axis_overflow() {
    let m = Tensor4D::<2, 2, 2, 2, 16>::new([0.0; 16]);
    m.get(0, 0, 0, 2);
}

#[test]
fn test_tensor6d_creation_and_shape() {
    let _m = Tensor6D::<2, 2, 2, 2, 2, 2, 64>::new([0.0; 64]);
}

#[test]
fn test_indexing_tensore6d() {
    let mut data = [0.0; 64];
    for k in 0..64 {
        data[k] = (k + 1) as Scalar;
    }
    let m = Tensor6D::<2, 2, 2, 2, 2, 2, 64>::new(data);
    assert_eq!(1.0, m.get(0, 0, 0, 0, 0, 0));
    assert_eq!(2.0, m.get(0, 0, 0, 0, 0, 1));
    assert_eq!(3.0, m.get(0, 0, 0, 0, 1, 0));
    assert_eq!(5.0, m.get(0, 0, 0, 1, 0, 0));
    assert_eq!(9.0, m.get(0, 0, 1, 0, 0, 0));
    assert_eq!(17.0, m.get(0, 1, 0, 0, 0, 0));
    assert_eq!(33.0, m.get(1, 0, 0, 0, 0, 0));
    assert_eq!(64.0, m.get(1, 1, 1, 1, 1, 1));
}

#[test]
fn test_setting_tensor6d() {
    let mut m = Tensor6D::<2, 2, 2, 2, 2, 2, 64>::new([0.0; 64]);
    m.set(1, 0, 1, 0, 1, 0, 42.0);
    assert_eq!(42.0, m.get(1, 0, 1, 0, 1, 0));
    assert_eq!(0.0, m.get(1, 0, 1, 0, 1, 1));
}

#[test]
#[should_panic]
fn test_indexing6d_not_valid() {
    let m = Tensor6D::<2, 2, 2, 2, 2, 2, 64>::new([0.0; 64]);
    m.get(12, 0, 0, 0, 0, 0);
}

#[test]
#[should_panic]
fn test_indexing6d_axis_overflow() {
    let m = Tensor6D::<2, 2, 2, 2, 2, 2, 64>::new([0.0; 64]);
    m.get(0, 0, 0, 0, 0, 2);
}

#[test]
#[should_panic]
fn test_setting6d_axis_overflow() {
    let mut m = Tensor6D::<2, 2, 2, 2, 2, 2, 64>::new([0.0; 64]);
    m.set(0, 0, 0, 0, 2, 0, 1.0);
}

#[test]
fn test_im2col_view_full_window() {
    // KH x KW = H x W : une seule position, la vue redonne le tenseur d'entree
    let m = Tensor4D::<1, 1, 2, 2, 4>::new([1.0, 2.0, 3.0, 4.0]);

    let v = m.im2col_view::<1, 1, 2, 2>(1);
    for p in 0..2 {
        for q in 0..2 {
            assert_eq!(m.get(0, 0, p, q), v.get(0, 0, 0, 0, p, q));
        }
    }
}

#[test]
fn test_im2col_view_sliding_window() {
    // 3x3, fenetre 2x2, stride 1 -> 4 patchs qui se recouvrent
    // 1 2 3
    // 4 5 6
    // 7 8 9
    let m = Tensor4D::<1, 1, 3, 3, 9>::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

    let v = m.im2col_view::<2, 2, 2, 2>(1);
    // patch (0, 0) = [[1, 2], [4, 5]]
    assert_eq!(1.0, v.get(0, 0, 0, 0, 0, 0));
    assert_eq!(2.0, v.get(0, 0, 0, 0, 0, 1));
    assert_eq!(4.0, v.get(0, 0, 0, 0, 1, 0));
    assert_eq!(5.0, v.get(0, 0, 0, 0, 1, 1));
    // patch (0, 1) : decale d'une colonne
    assert_eq!(2.0, v.get(0, 0, 1, 0, 0, 0));
    assert_eq!(6.0, v.get(0, 0, 1, 0, 1, 1));
    // patch (1, 0) : decale d'une ligne
    assert_eq!(4.0, v.get(0, 1, 0, 0, 0, 0));
    assert_eq!(8.0, v.get(0, 1, 0, 0, 1, 1));
    // patch (1, 1)
    assert_eq!(5.0, v.get(0, 1, 1, 0, 0, 0));
    assert_eq!(9.0, v.get(0, 1, 1, 0, 1, 1));
    // le pixel central appartient aux quatre fenetres : la vue alias, elle ne copie pas
    assert_eq!(5.0, v.get(0, 0, 0, 0, 1, 1));
    assert_eq!(5.0, v.get(0, 0, 1, 0, 1, 0));
    assert_eq!(5.0, v.get(0, 1, 0, 0, 0, 1));
    assert_eq!(5.0, v.get(0, 1, 1, 0, 0, 0));
}

#[test]
fn test_im2col_view_stride_2() {
    // 4x4, fenetre 2x2, stride 2 -> 4 patchs disjoints
    //  1  2  3  4
    //  5  6  7  8
    //  9 10 11 12
    // 13 14 15 16
    let mut data = [0.0; 16];
    for k in 0..16 {
        data[k] = (k + 1) as Scalar;
    }
    let m = Tensor4D::<1, 1, 4, 4, 16>::new(data);

    let v = m.im2col_view::<2, 2, 2, 2>(2);
    assert_eq!(1.0, v.get(0, 0, 0, 0, 0, 0));
    assert_eq!(6.0, v.get(0, 0, 0, 0, 1, 1));
    assert_eq!(3.0, v.get(0, 0, 1, 0, 0, 0));
    assert_eq!(8.0, v.get(0, 0, 1, 0, 1, 1));
    assert_eq!(9.0, v.get(0, 1, 0, 0, 0, 0));
    assert_eq!(14.0, v.get(0, 1, 0, 0, 1, 1));
    assert_eq!(11.0, v.get(0, 1, 1, 0, 0, 0));
    assert_eq!(16.0, v.get(0, 1, 1, 0, 1, 1));
}

#[test]
fn test_im2col_view_strides_invariant() {
    // l'invariant complet de la vue, sur tous les axes a la fois :
    // v.get(n, i, j, c, p, q) == m.get(n, c, i * stride + p, j * stride + q)
    const N: usize = 2;
    const C: usize = 2;
    const H: usize = 4;
    const W: usize = 4;
    const KH: usize = 2;
    const KW: usize = 3;

    let mut data = [0.0; 64];
    for k in 0..64 {
        data[k] = (k + 1) as Scalar;
    }
    let m = Tensor4D::<N, C, H, W, 64>::new(data);

    // stride 1 : H_OUT = 3, W_OUT = 2
    let v = m.im2col_view::<3, 2, KH, KW>(1);
    for n in 0..N {
        for i in 0..3 {
            for j in 0..2 {
                for c in 0..C {
                    for p in 0..KH {
                        for q in 0..KW {
                            assert_eq!(m.get(n, c, i + p, j + q), v.get(n, i, j, c, p, q));
                        }
                    }
                }
            }
        }
    }

    // stride 2 : H_OUT = 2, W_OUT = 1
    let v2 = m.im2col_view::<2, 1, KH, KW>(2);
    for n in 0..N {
        for i in 0..2 {
            for c in 0..C {
                for p in 0..KH {
                    for q in 0..KW {
                        assert_eq!(m.get(n, c, i * 2 + p, q), v2.get(n, i, 0, c, p, q));
                    }
                }
            }
        }
    }
}

#[test]
#[should_panic]
fn test_im2col_view_wrong_output_size() {
    // 3x3 avec une fenetre 2x2 et stride 1 donne 2x2, pas 3x3
    let m = Tensor4D::<1, 1, 3, 3, 9>::new([0.0; 9]);
    let _v = m.im2col_view::<3, 3, 2, 2>(1);
}

#[test]
#[should_panic]
fn test_im2col_view_kernel_larger_than_input() {
    let m = Tensor4D::<1, 1, 2, 2, 4>::new([0.0; 4]);
    let _v = m.im2col_view::<1, 1, 3, 3>(1);
}

#[test]
#[should_panic]
fn test_im2col_view_null_stride() {
    let m = Tensor4D::<1, 1, 3, 3, 9>::new([0.0; 9]);
    let _v = m.im2col_view::<3, 3, 1, 1>(0);
}

#[test]
#[should_panic]
fn test_im2col_view_axis_overflow() {
    let m = Tensor4D::<1, 1, 3, 3, 9>::new([0.0; 9]);
    let v = m.im2col_view::<2, 2, 2, 2>(1);
    // W_OUT vaut 2 : la troisieme position de fenetre n'existe pas
    v.get(0, 0, 2, 0, 0, 0);
}

#[test]
fn test_im2col_view_feeds_tensordot_3() {
    // cross-correlation 2D de bout en bout : im2col puis contraction sur (C, KH, KW)
    // 1 2 3
    // 4 5 6
    // 7 8 9
    let m = Tensor4D::<1, 1, 3, 3, 9>::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let v = m.im2col_view::<2, 2, 2, 2>(1);

    // filtre 0 : diagonale (a + d), filtre 1 : somme du patch
    let filters = Tensor4D::<2, 1, 2, 2, 8>::new([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

    // la vue est contractee telle quelle : aucun tenseur de patchs intermediaire
    let out: Tensor4D<1, 2, 2, 2, 8> = tensordot_3(&v, &filters);
    // diagonales : 1+5, 2+6, 4+8, 5+9
    assert_eq!(6.0, out.get(0, 0, 0, 0));
    assert_eq!(8.0, out.get(0, 0, 1, 0));
    assert_eq!(12.0, out.get(0, 1, 0, 0));
    assert_eq!(14.0, out.get(0, 1, 1, 0));
    // sommes : 1+2+4+5, 2+3+5+6, 4+5+7+8, 5+6+8+9
    assert_eq!(12.0, out.get(0, 0, 0, 1));
    assert_eq!(16.0, out.get(0, 0, 1, 1));
    assert_eq!(24.0, out.get(0, 1, 0, 1));
    assert_eq!(28.0, out.get(0, 1, 1, 1));
}

#[test]
fn test_tensordot_3_view_matches_materialised() {
    // contracter la vue doit donner exactement le meme resultat que contracter
    // le tenseur de patchs recopie a la main, batches et canaux compris
    const N: usize = 2;
    const C: usize = 2;
    const H_OUT: usize = 2;
    const W_OUT: usize = 2;
    const KH: usize = 2;
    const KW: usize = 2;
    const K: usize = 3;

    let mut data = [0.0; 36];
    for k in 0..36 {
        data[k] = (k + 1) as Scalar;
    }
    let m = Tensor4D::<N, C, 3, 3, 36>::new(data);
    let v = m.im2col_view::<H_OUT, W_OUT, KH, KW>(1);

    let mut filter_data = [0.0; 24];
    for k in 0..24 {
        filter_data[k] = (k % 5) as Scalar - 2.0;
    }
    let filters = Tensor4D::<K, C, KH, KW, 24>::new(filter_data);

    let mut patches = Tensor6D::<N, H_OUT, W_OUT, C, KH, KW, 64>::new([0.0; 64]);
    for n in 0..N {
        for i in 0..H_OUT {
            for j in 0..W_OUT {
                for c in 0..C {
                    for p in 0..KH {
                        for q in 0..KW {
                            patches.set(n, i, j, c, p, q, v.get(n, i, j, c, p, q));
                        }
                    }
                }
            }
        }
    }

    let from_view: Tensor4D<N, H_OUT, W_OUT, K, 24> = tensordot_3(&v, &filters);
    let from_tensor: Tensor4D<N, H_OUT, W_OUT, K, 24> = tensordot_3(&patches, &filters);

    for n in 0..N {
        for i in 0..H_OUT {
            for j in 0..W_OUT {
                for k in 0..K {
                    assert_eq!(from_tensor.get(n, i, j, k), from_view.get(n, i, j, k));
                }
            }
        }
    }
    // et le resultat n'est pas trivialement nul partout
    assert_ne!(0.0, from_view.get(0, 0, 0, 0));
}

/// Le lieu de stockage ne doit rien changer au résultat : même entrée, même
/// contraction, une fois sur la pile et une fois sur le tas.
#[cfg(feature = "alloc")]
#[test]
fn test_storage_agnostic_cross_correlation() {
    use ferrite::linalg::Tensor4DBoxed;

    const N: usize = 2;
    const C: usize = 3;
    const H: usize = 6;
    const W: usize = 6;
    const K: usize = 2;
    const H_OUT: usize = 4;
    const W_OUT: usize = 4;
    const NUMEL_X: usize = N * C * H * W;
    const NUMEL_F: usize = K * C * 3 * 3;
    const NUMEL_Y: usize = N * H_OUT * W_OUT * K;

    let mut video = [0.0 as Scalar; NUMEL_X];
    for (i, v) in video.iter_mut().enumerate() {
        *v = (i as Scalar) * 0.5 - 3.0;
    }
    let mut filters = [0.0 as Scalar; NUMEL_F];
    for (i, v) in filters.iter_mut().enumerate() {
        *v = ((i % 7) as Scalar) - 2.0;
    }

    let vid_stack = Tensor4D::<N, C, H, W, NUMEL_X>::new(video);
    let fil_stack = Tensor4D::<K, C, 3, 3, NUMEL_F>::new(filters);
    let on_stack: Tensor4D<N, H_OUT, W_OUT, K, NUMEL_Y> =
        tensordot_3(&vid_stack.im2col_view::<H_OUT, W_OUT, 3, 3>(1), &fil_stack);

    let vid_heap = Tensor4DBoxed::<N, C, H, W, NUMEL_X>::from_vec(video.to_vec()).unwrap();
    let fil_heap = Tensor4DBoxed::<K, C, 3, 3, NUMEL_F>::from_vec(filters.to_vec()).unwrap();
    let on_heap: Tensor4DBoxed<N, H_OUT, W_OUT, K, NUMEL_Y> =
        tensordot_3(&vid_heap.im2col_view::<H_OUT, W_OUT, 3, 3>(1), &fil_heap);

    assert_eq!(on_stack.get_shape(), on_heap.get_shape());
    assert_eq!(on_stack.get_data(), on_heap.get_data());
    // et le resultat n'est pas trivialement nul partout
    assert_ne!(0.0, on_heap.get(0, 0, 0, 0));
}

/// Croiser les stockages dans une même contraction : entrée tas, filtres pile,
/// sortie pile. Si ça compile et que ça donne la même chose, `Storage` ne fuit
/// pas dans le noyau de calcul.
#[cfg(feature = "alloc")]
#[test]
fn test_mixed_storage_operands() {
    use ferrite::linalg::Tensor4DBoxed;

    let vid_heap =
        Tensor4DBoxed::<1, 1, 4, 4, 16>::from_vec((0..16).map(|i| i as Scalar).collect()).unwrap();

    let fil_stack = Tensor4D::<1, 1, 2, 2, 4>::new([1.0, 1.0, 1.0, 1.0]);

    let out: Tensor4D<1, 3, 3, 1, 9> =
        tensordot_3(&vid_heap.im2col_view::<3, 3, 2, 2>(1), &fil_stack);

    assert_eq!(10.0, out.get(0, 0, 0, 0)); // 0+1+4+5
    assert_eq!(14.0, out.get(0, 0, 1, 0)); // 1+2+5+6
}

// --- Tensor operator/method parity (Matrix/Vector role) ---

#[test]
fn test_tensor_rows_cols() {
    let m = Tensor::<2, 3, 6>::new([0.0; 6]);
    assert_eq!(m.rows(), 2);
    assert_eq!(m.cols(), 3);
}

#[test]
fn test_tensor_index_get_set() {
    let mut m = Tensor::<2, 2, 4>::new([0.0; 4]);
    m[(0, 1)] = 42.0;
    assert_eq!(m[(0, 1)], 42.0);
    assert_eq!(m[(1, 1)], 0.0);
}

#[test]
#[should_panic]
fn test_tensor_index_out_of_bounds() {
    let m = Tensor::<2, 2, 4>::new([0.0; 4]);
    let _ = m[(2, 0)];
}

#[test]
fn test_tensor_addition() {
    let mut m1 = Tensor::<2, 2, 4>::new([0.0; 4]);
    m1[(0, 0)] = 1.0;
    let mut m2 = Tensor::<2, 2, 4>::new([0.0; 4]);
    m2[(0, 0)] = 2.0;
    assert_eq!(m1 + m2, {
        let mut res = Tensor::<2, 2, 4>::new([0.0; 4]);
        res[(0, 0)] = 3.0;
        res
    });
}

#[test]
fn test_tensor_multiply() {
    let mut m1 = Tensor::<2, 2, 4>::new([0.0; 4]);
    m1[(0, 0)] = 1.0;
    m1[(0, 1)] = 2.0;
    m1[(1, 0)] = 3.0;
    m1[(1, 1)] = 4.0;
    let mut m2 = Tensor::<2, 1, 2>::new([0.0; 2]);
    m2[(0, 0)] = 5.0;
    m2[(1, 0)] = 6.0;
    let res: Tensor<2, 1, 2> = m1.multiply(&m2);
    assert_eq!(res[(0, 0)], 17.0);
    assert_eq!(res[(1, 0)], 39.0);
}

#[test]
fn test_tensor_matmul_accumulate() {
    let mut res = Tensor::<1, 1, 1>::new([0.0; 1]);
    res[(0, 0)] = 10.0;
    let m1 = Tensor::<1, 1, 1>::identity();
    let m2 = Tensor::<1, 1, 1>::identity();
    res.matmul_accumulate(&m1, &m2);
    assert_eq!(res[(0, 0)], 11.0);
}

#[test]
fn test_tensor_transposed() {
    let mut m = Tensor::<1, 2, 2>::new([0.0; 2]);
    m[(0, 0)] = 1.0;
    m[(0, 1)] = 2.0;
    let t = m.transposed();
    assert_eq!(t.rows(), 2);
    assert_eq!(t.cols(), 1);
    assert_eq!(t[(1, 0)], 2.0);
}

#[test]
fn test_tensor_col_extraction() {
    let mut m = Tensor::<2, 2, 4>::new([0.0; 4]);
    m[(0, 1)] = 5.0;
    m[(1, 1)] = 10.0;
    let col = m.get_col(1).unwrap();
    assert_eq!(col, Vector::from_data([5.0, 10.0]));
}

#[test]
fn test_tensor_from_cols() {
    let v1 = Vector::from_data([1.0, 2.0]);
    let v2 = Vector::from_data([3.0, 4.0]);
    let m: Tensor<2, 2, 4> = Tensor::from_cols([v1, v2]);
    assert_eq!(m[(1, 0)], 2.0);
    assert_eq!(m[(1, 1)], 4.0);
}

#[test]
fn test_tensor_identity() {
    let id = Tensor::<3, 3, 9>::identity();
    assert_eq!(id[(0, 0)], 1.0);
    assert_eq!(id[(0, 1)], 0.0);
    assert_eq!(id[(1, 1)], 1.0);
    assert_eq!(id[(2, 2)], 1.0);
}

// --- Vector (Tensor<N, 1, N>) role ---

#[test]
fn test_vector_creation_and_dim() {
    let v: Vector<3> = Vector::from_data([1.0, 2.0, 3.0]);
    assert_eq!(v.dim(), 3);
}

#[test]
fn test_vector_l1_norm() {
    let v = Vector::from_data([1.0, -2.0, 3.0]);
    assert_eq!(v.l1_norm(), 6.0);
}

#[test]
fn test_vector_l2_norm() {
    let v = Vector::from_data([3.0, 4.0]);
    assert_eq!(v.l2_norm(), 5.0);
}

#[test]
fn test_vector_inf_norm() {
    let v = Vector::from_data([-10.0, 2.0, 5.0]);
    assert_eq!(v.inf_norm(), 10.0);
}

#[test]
fn test_vector_dot_product() {
    let v1 = Vector::from_data([1.0, 2.0]);
    let v2 = Vector::from_data([3.0, 4.0]);
    assert_eq!(v1.dot(&v2), 11.0);
}

#[test]
fn test_vector_addition() {
    let v1 = Vector::from_data([1.0, 2.0]);
    let v2 = Vector::from_data([3.0, 4.0]);
    assert_eq!(v1 + v2, Vector::from_data([4.0, 6.0]));
}

#[test]
fn test_vector_subtraction() {
    let v1 = Vector::from_data([5.0, 7.0]);
    let v2 = Vector::from_data([2.0, 3.0]);
    assert_eq!(v1 - v2, Vector::from_data([3.0, 4.0]));
}

#[test]
fn test_vector_scalar_mul() {
    let v = Vector::from_data([1.0, -2.0]);
    assert_eq!(v * 3.0, Vector::from_data([3.0, -6.0]));
}

#[test]
fn test_vector_neg() {
    let v = Vector::from_data([1.0, -2.0]);
    assert_eq!(-v, Vector::from_data([-1.0, 2.0]));
}

#[test]
fn test_vector_div() {
    let v = Vector::from_data([2.0, -4.0]);
    assert_eq!(v / 2.0, Vector::from_data([1.0, -2.0]));
}

#[test]
fn test_vector_hadamard() {
    let v1 = Vector::from_data([2.0, 3.0]);
    let v2 = Vector::from_data([4.0, 5.0]);
    assert_eq!(v1.hadamard(&v2), Vector::from_data([8.0, 15.0]));
}

#[test]
fn test_vector_sum() {
    let v = Vector::from_data([1.0, 2.0, 3.0]);
    assert_eq!(v.sum(), 6.0);
}

#[test]
fn test_vector_index() {
    let mut v = Vector::<2>::from_data([1.0, 2.0]);
    assert_eq!(v[0], 1.0);
    v[1] = 5.0;
    assert_eq!(v[1], 5.0);
}

#[test]
fn test_vector_projection() {
    let v = Vector::from_data([1.0, 1.0]);
    let target = Vector::from_data([1.0, 0.0]);
    assert_eq!(
        v.orthogonal_projection(&target),
        Vector::from_data([1.0, 0.0])
    );
}

#[test]
fn test_vector_null_projection() {
    let v = Vector::from_data([1.0, 2.0]);
    let null_v = Vector::<2>::from_data([0.0, 0.0]);
    assert_eq!(
        v.orthogonal_projection(&null_v),
        Vector::from_data([0.0, 0.0])
    );
}
