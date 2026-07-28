use ferrite::scalar::{fabs, Scalar};
use ferrite::sp::{conv2d, filter_bank, Gaussian3D, Sobel3D};
use ferrite::tensor::{Tensor3D, Tensor4D};

#[test]
fn test_conv2d_single_frame_two_filters() {
    // 1 2 3
    // 4 5 6
    // 7 8 9
    let mut frames = Tensor4D::<1, 1, 3, 3, 9>::new();
    frames.load_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

    // filtre 0 : diagonale (a + d), filtre 1 : somme du patch
    let mut filters = Tensor4D::<2, 1, 2, 2, 8>::new();
    filters.load_data([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

    // chaque filtre devient un canal de sortie
    let out: Tensor4D<1, 2, 2, 2, 8> = conv2d(&frames, &filters, 1);
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
fn test_conv2d_sequence_matches_naive_convolution() {
    // sequence de 2 frames, 2 canaux, 4x4, contre 3 filtres 2x2
    const N: usize = 2;
    const C: usize = 2;
    const H: usize = 4;
    const W: usize = 4;
    const K: usize = 3;
    const KH: usize = 2;
    const KW: usize = 2;
    const H_OUT: usize = 3;
    const W_OUT: usize = 3;

    let mut frames = Tensor4D::<N, C, H, W, 64>::new();
    let mut data = [0.0; 64];
    for i in 0..64 {
        data[i] = (i % 7) as Scalar - 3.0;
    }
    frames.load_data(data);

    let mut filters = Tensor4D::<K, C, KH, KW, 24>::new();
    let mut filter_data = [0.0; 24];
    for i in 0..24 {
        filter_data[i] = (i % 5) as Scalar - 2.0;
    }
    filters.load_data(filter_data);

    let out: Tensor4D<N, H_OUT, W_OUT, K, 54> = conv2d(&frames, &filters, 1);

    // reference : la convolution ecrite a la main, boucle par boucle
    for n in 0..N {
        for i in 0..H_OUT {
            for j in 0..W_OUT {
                for k in 0..K {
                    let mut expected: Scalar = 0.0;
                    for c in 0..C {
                        for p in 0..KH {
                            for q in 0..KW {
                                expected +=
                                    frames.get(n, c, i + p, j + q) * filters.get(k, c, p, q);
                            }
                        }
                    }
                    assert_eq!(expected, out.get(n, i, j, k));
                }
            }
        }
    }
    // et le resultat n'est pas trivialement nul partout
    assert_ne!(0.0, out.get(0, 0, 0, 0));
}

#[test]
fn test_conv2d_stride_2() {
    // stride 2 sur une 4x4 : fenetres disjointes, sortie 2x2
    let mut frames = Tensor4D::<1, 1, 4, 4, 16>::new();
    frames.load_data([
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ]);

    let mut filters = Tensor4D::<1, 1, 2, 2, 4>::new();
    filters.load_data([1.0, 1.0, 1.0, 1.0]);

    let out: Tensor4D<1, 2, 2, 1, 4> = conv2d(&frames, &filters, 2);
    assert_eq!(14.0, out.get(0, 0, 0, 0)); // 1+2+5+6
    assert_eq!(22.0, out.get(0, 0, 1, 0)); // 3+4+7+8
    assert_eq!(46.0, out.get(0, 1, 0, 0)); // 9+10+13+14
    assert_eq!(54.0, out.get(0, 1, 1, 0)); // 11+12+15+16
}

#[test]
fn test_conv2d_gaussian_and_sobel_bank() {
    // 2 frames monocanal 4x4, remplies d'une rampe : f(i, j) = 1 + 4i + j pour
    // la premiere frame, +16 pour la seconde
    let mut frames = Tensor4D::<2, 1, 4, 4, 32>::new();
    let mut data = [0.0; 32];
    for i in 0..32 {
        data[i] = (i + 1) as Scalar;
    }
    frames.load_data(data);

    // un seul banc contenant les deux kernels : canal 0 = flou, canal 1 = bord
    let gaussian: Tensor3D<1, 3, 3, 9> = Gaussian3D::kernel();
    let sobel_x: Tensor3D<1, 3, 3, 9> = Sobel3D::x();
    let bank: Tensor4D<2, 1, 3, 3, 18> = filter_bank([&gaussian, &sobel_x]);

    let out: Tensor4D<2, 2, 2, 2, 16> = conv2d(&frames, &bank, 1);

    // gaussienne a gain unite : sur une rampe (lineaire), elle rend le pixel
    // central de la fenetre, donc f(i + 1, j + 1)
    assert_eq!(6.0, out.get(0, 0, 0, 0));
    assert_eq!(7.0, out.get(0, 0, 1, 0));
    assert_eq!(10.0, out.get(0, 1, 0, 0));
    assert_eq!(11.0, out.get(0, 1, 1, 0));
    // seconde frame : meme rampe decalee de 16
    assert_eq!(22.0, out.get(1, 0, 0, 0));
    assert_eq!(27.0, out.get(1, 1, 1, 0));

    // sobel x : gradient constant de 1 par colonne, gain 8 du kernel -> 8 partout
    for n in 0..2 {
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(8.0, out.get(n, i, j, 1));
            }
        }
    }
}

#[test]
fn test_conv2d_bank_over_two_channels() {
    // la profondeur des kernels sert ici : canal 0 = rampe, canal 1 = zeros.
    // La contraction somme les canaux, et les kernels normalisent par C, donc
    // la reponse est la moyenne des reponses par canal.
    let mut frames = Tensor4D::<1, 2, 4, 4, 32>::new();
    let mut data = [0.0; 32];
    for i in 0..16 {
        data[i] = (i + 1) as Scalar;
    }
    frames.load_data(data);

    let gaussian: Tensor3D<2, 3, 3, 18> = Gaussian3D::kernel();
    let sobel_y: Tensor3D<2, 3, 3, 18> = Sobel3D::y();
    let bank: Tensor4D<2, 2, 3, 3, 36> = filter_bank([&gaussian, &sobel_y]);

    let out: Tensor4D<1, 2, 2, 2, 8> = conv2d(&frames, &bank, 1);

    // gaussienne : (pixel central + 0) / 2
    assert_eq!(3.0, out.get(0, 0, 0, 0));
    assert_eq!(5.5, out.get(0, 1, 1, 0));
    // sobel y : gradient de 4 par ligne, gain 8 -> 32 sur le canal 0, 0 sur le
    // canal 1, moyenne 16
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(16.0, out.get(0, i, j, 1));
        }
    }
}

#[test]
fn test_gaussian_bank_preserves_constant_sequence() {
    // gain DC unite, canaux compris : une sequence constante ressort identique
    let mut frames = Tensor4D::<1, 3, 3, 3, 27>::new();
    frames.load_data([7.0; 27]);

    let gaussian: Tensor3D<3, 3, 3, 27> = Gaussian3D::kernel();
    let sobel_x: Tensor3D<3, 3, 3, 27> = Sobel3D::x();
    let bank: Tensor4D<2, 3, 3, 3, 54> = filter_bank([&gaussian, &sobel_x]);

    let out: Tensor4D<1, 1, 1, 2, 2> = conv2d(&frames, &bank, 1);
    // 1/(16 * 3) n'est pas exact en binaire : on compare a l'epsilon pres
    assert!(fabs(7.0 - out.get(0, 0, 0, 0)) < 1e-5);
    // et un gradient nul sur une image plate
    assert_eq!(0.0, out.get(0, 0, 0, 1));
}
