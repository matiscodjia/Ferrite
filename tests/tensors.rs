use ferrite::scalar::Scalar;
use ferrite::tensor::tensordot_1;
use ferrite::tensor::tensordot_2;
use ferrite::tensor::tensordot_3;
use ferrite::tensor::Tensor;
use ferrite::tensor::Tensor3D;
use ferrite::tensor::Tensor4D;
use ferrite::tensor::Tensor6D;
#[test]
//This is a compilation level test, test NOK = No compilation
fn test_tensor_creation_and_shape() {
    let _m = Tensor::<2, 3, 6>::new();
}

#[test]
fn test_indexing() {
    let m = Tensor::<2, 3, 6>::new();
    println!("{}", m.get(0, 0))
}

#[test]
#[should_panic]
fn test_indexing_not_valid() {
    let m = Tensor::<2, 3, 6>::new();
    println!("{}", m.get(12, 2))
}

#[test]
fn test_setting() {
    let mut m = Tensor::<2, 3, 6>::new();
    m.set(0, 0, 2.0);
    assert_eq!(2.0, m.get(0, 0))
}

#[test]
fn test_transpose() {
    let mut m = Tensor::<2, 3, 6>::new();
    m.set(1, 2, 4.0);
    m.transpose();
    assert_eq!(4.0, m.get(2, 1));

    m.transpose();
    assert_eq!(4.0, m.get(1, 2));
}

#[test]
fn test_tensor_view() {
    let mut m = Tensor::<3, 3, 9>::new();
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    m.load_data(data);
    let m_view = m.view((1, 2), (1, 2));
    assert_eq!(5.0, m_view.get(0, 0));
    assert_eq!(6.0, m_view.get(0, 1));
    assert_eq!(8.0, m_view.get(1, 0));
    assert_eq!(9.0, m_view.get(1, 1));
}

#[test]
#[should_panic]
fn test_view_indexing_not_valid() {
    let mut m = Tensor::<3, 3, 9>::new();
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    m.load_data(data);
    let m_view = m.view((1, 2), (1, 2));
    m_view.get(5, 5);
}
#[test]
#[should_panic]
fn test_indexing_axis_overflow() {
    // 2 lignes, 3 colonnes : la colonne 3 sort de l'axe sans sortir du buffer
    let m = Tensor::<2, 3, 6>::new();
    m.get(0, 3);
}

#[test]
#[should_panic]
fn test_setting_axis_overflow() {
    let mut m = Tensor::<2, 3, 6>::new();
    m.set(0, 3, 1.0);
}

#[test]
#[should_panic]
fn test_indexing_axis_overflow_transposed() {
    // apres transpose la shape est (3, 2) : la colonne 2 sort de l'axe
    let mut m = Tensor::<2, 3, 6>::new();
    m.transpose();
    m.get(0, 2);
}

#[test]
fn test_tensordot() {
    // (2 x 3) . (3 x 2) -> (2 x 2)
    let mut a = Tensor::<2, 3, 6>::new();
    a.load_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mut b = Tensor::<3, 2, 6>::new();
    b.load_data([7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

    let c: Tensor<2, 2, 4> = tensordot_1(&a, &b);
    assert_eq!(58.0, c.get(0, 0));
    assert_eq!(64.0, c.get(0, 1));
    assert_eq!(139.0, c.get(1, 0));
    assert_eq!(154.0, c.get(1, 1));
}

#[test]
fn test_tensordot_identity() {
    let mut a = Tensor::<2, 2, 4>::new();
    a.load_data([1.0, 2.0, 3.0, 4.0]);
    let mut id = Tensor::<2, 2, 4>::new();
    id.load_data([1.0, 0.0, 0.0, 1.0]);

    let c: Tensor<2, 2, 4> = tensordot_1(&a, &id);
    assert_eq!(1.0, c.get(0, 0));
    assert_eq!(2.0, c.get(0, 1));
    assert_eq!(3.0, c.get(1, 0));
    assert_eq!(4.0, c.get(1, 1));
}

#[test]
fn test_tensordot_non_square() {
    // (1 x 3) . (3 x 4) -> (1 x 4)
    let mut a = Tensor::<1, 3, 3>::new();
    a.load_data([1.0, 2.0, 3.0]);
    let mut b = Tensor::<3, 4, 12>::new();
    b.load_data([
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
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
    let mut a = Tensor::<2, 3, 6>::new();
    a.transpose();
    let b = Tensor::<3, 2, 6>::new();
    let _c: Tensor<2, 2, 4> = tensordot_1(&a, &b);
}

#[test]
fn test_tensordot_2() {
    // (2 x 2 x 2) . (2 x 2 x 3) -> (2 x 3)
    let mut a = Tensor3D::<2, 2, 2, 8>::new();
    a.load_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let mut b = Tensor3D::<2, 2, 3, 12>::new();
    b.load_data([
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
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
    let b_data = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];

    let mut a3 = Tensor3D::<2, 2, 2, 8>::new();
    a3.load_data(a_data);
    let mut b3 = Tensor3D::<2, 2, 3, 12>::new();
    b3.load_data(b_data);
    let c3: Tensor<2, 3, 6> = tensordot_2(&a3, &b3);

    let mut a2 = Tensor::<2, 4, 8>::new();
    a2.load_data(a_data);
    let mut b2 = Tensor::<4, 3, 12>::new();
    b2.load_data(b_data);
    let c2: Tensor<2, 3, 6> = tensordot_1(&a2, &b2);

    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(c2.get(i, j), c3.get(i, j));
        }
    }
}

#[test]
fn test_tensordot_2_single_inner_axis() {
    // K2 = 1 : la contraction sur deux axes degenere en produit matriciel
    let mut a = Tensor3D::<2, 3, 1, 6>::new();
    a.load_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mut b = Tensor3D::<3, 1, 2, 6>::new();
    b.load_data([7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

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
    let mut a = Tensor6D::<1, 1, 2, 2, 2, 2, 16>::new();
    let mut b = Tensor4D::<2, 2, 2, 2, 16>::new();
    let mut data = [0.0; 16];
    for k in 0..16 {
        data[k] = (k + 1) as Scalar;
    }
    a.load_data(data);
    b.load_data(data);

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
    // (N * H_out * W_out, C * KH * KW) . (C * KH * KW, K) sur les memes donnees.
    // Le buffer de `a` est deja dans le bon ordre (row-major), celui de `b` doit
    // etre transpose puisqu'il est stocke en (K, C * KH * KW).
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

    let mut a6 = Tensor6D::<N, H_OUT, W_OUT, 2, 2, 1, 16>::new();
    a6.load_data(a_data);
    let mut b4 = Tensor4D::<K, 2, 2, 1, 12>::new();
    b4.load_data(b_data);
    let c4: Tensor4D<N, H_OUT, W_OUT, K, 12> = tensordot_3(&a6, &b4);

    let mut a2 = Tensor::<4, INNER, 16>::new();
    a2.load_data(a_data);
    let mut b2 = Tensor::<INNER, K, 12>::new();
    for k in 0..K {
        for c in 0..2 {
            for p in 0..2 {
                b2.set(c * 2 + p, k, b4.get(k, c, p, 0));
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
    let mut a = Tensor6D::<2, 1, 2, 1, 1, 1, 4>::new();
    a.load_data([1.0, 2.0, 3.0, 4.0]);
    let mut b = Tensor4D::<3, 1, 1, 1, 3>::new();
    b.load_data([5.0, 6.0, 7.0]);

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
    let _m = Tensor3D::<2, 2, 2, 8>::new();
}

#[test]
fn test_indexing_tensore3d() {
    let mut m = Tensor3D::<2, 2, 2, 8>::new();
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    m.load_data(data);
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
    let m = Tensor3D::<2, 2, 2, 8>::new();
    m.get(12, 2, 0);
}

#[test]
#[should_panic]
fn test_indexing3d_axis_overflow() {
    let m = Tensor3D::<2, 2, 2, 8>::new();
    m.get(0, 0, 2);
}

#[test]
fn test_tensor4d_creation_and_shape() {
    let _m = Tensor4D::<2, 2, 2, 2, 16>::new();
}

#[test]
fn test_indexing_tensore4d() {
    let mut m = Tensor4D::<2, 2, 2, 2, 16>::new();
    let data = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    m.load_data(data);
    assert_eq!(1.0, m.get(0, 0, 0, 0));
    assert_eq!(16.0, m.get(1, 1, 1, 1));
    assert_eq!(8.0, m.get(0, 1, 1, 1));
}

#[test]
#[should_panic]
fn test_indexing4d_axis_overflow() {
    let m = Tensor4D::<2, 2, 2, 2, 16>::new();
    m.get(0, 0, 0, 2);
}

#[test]
fn test_tensor6d_creation_and_shape() {
    let _m = Tensor6D::<2, 2, 2, 2, 2, 2, 64>::new();
}

#[test]
fn test_indexing_tensore6d() {
    let mut m = Tensor6D::<2, 2, 2, 2, 2, 2, 64>::new();
    let mut data = [0.0; 64];
    for k in 0..64 {
        data[k] = (k + 1) as Scalar;
    }
    m.load_data(data);
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
    let mut m = Tensor6D::<2, 2, 2, 2, 2, 2, 64>::new();
    m.set(1, 0, 1, 0, 1, 0, 42.0);
    assert_eq!(42.0, m.get(1, 0, 1, 0, 1, 0));
    assert_eq!(0.0, m.get(1, 0, 1, 0, 1, 1));
}

#[test]
#[should_panic]
fn test_indexing6d_not_valid() {
    let m = Tensor6D::<2, 2, 2, 2, 2, 2, 64>::new();
    m.get(12, 0, 0, 0, 0, 0);
}

#[test]
#[should_panic]
fn test_indexing6d_axis_overflow() {
    let m = Tensor6D::<2, 2, 2, 2, 2, 2, 64>::new();
    m.get(0, 0, 0, 0, 0, 2);
}

#[test]
#[should_panic]
fn test_setting6d_axis_overflow() {
    let mut m = Tensor6D::<2, 2, 2, 2, 2, 2, 64>::new();
    m.set(0, 0, 0, 0, 2, 0, 1.0);
}

#[test]
fn test_im2col_view_full_window() {
    // KH x KW = H x W : une seule position, la vue redonne le tenseur d'entree
    let mut m = Tensor4D::<1, 1, 2, 2, 4>::new();
    m.load_data([1.0, 2.0, 3.0, 4.0]);

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
    let mut m = Tensor4D::<1, 1, 3, 3, 9>::new();
    m.load_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

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
    let mut m = Tensor4D::<1, 1, 4, 4, 16>::new();
    let mut data = [0.0; 16];
    for k in 0..16 {
        data[k] = (k + 1) as Scalar;
    }
    m.load_data(data);

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

    let mut m = Tensor4D::<N, C, H, W, 64>::new();
    let mut data = [0.0; 64];
    for k in 0..64 {
        data[k] = (k + 1) as Scalar;
    }
    m.load_data(data);

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
    let m = Tensor4D::<1, 1, 3, 3, 9>::new();
    let _v = m.im2col_view::<3, 3, 2, 2>(1);
}

#[test]
#[should_panic]
fn test_im2col_view_kernel_larger_than_input() {
    let m = Tensor4D::<1, 1, 2, 2, 4>::new();
    let _v = m.im2col_view::<1, 1, 3, 3>(1);
}

#[test]
#[should_panic]
fn test_im2col_view_null_stride() {
    let m = Tensor4D::<1, 1, 3, 3, 9>::new();
    let _v = m.im2col_view::<3, 3, 1, 1>(0);
}

#[test]
#[should_panic]
fn test_im2col_view_axis_overflow() {
    let m = Tensor4D::<1, 1, 3, 3, 9>::new();
    let v = m.im2col_view::<2, 2, 2, 2>(1);
    // W_OUT vaut 2 : la troisieme position de fenetre n'existe pas
    v.get(0, 0, 2, 0, 0, 0);
}

#[test]
fn test_im2col_view_feeds_tensordot_3() {
    // conv2d de bout en bout : im2col puis contraction sur (C, KH, KW)
    // 1 2 3
    // 4 5 6
    // 7 8 9
    let mut m = Tensor4D::<1, 1, 3, 3, 9>::new();
    m.load_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let v = m.im2col_view::<2, 2, 2, 2>(1);

    // filtre 0 : diagonale (a + d), filtre 1 : somme du patch
    let mut filters = Tensor4D::<2, 1, 2, 2, 8>::new();
    filters.load_data([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

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

    let mut m = Tensor4D::<N, C, 3, 3, 36>::new();
    let mut data = [0.0; 36];
    for k in 0..36 {
        data[k] = (k + 1) as Scalar;
    }
    m.load_data(data);
    let v = m.im2col_view::<H_OUT, W_OUT, KH, KW>(1);

    let mut filters = Tensor4D::<K, C, KH, KW, 24>::new();
    let mut filter_data = [0.0; 24];
    for k in 0..24 {
        filter_data[k] = (k % 5) as Scalar - 2.0;
    }
    filters.load_data(filter_data);

    let mut patches = Tensor6D::<N, H_OUT, W_OUT, C, KH, KW, 64>::new();
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
