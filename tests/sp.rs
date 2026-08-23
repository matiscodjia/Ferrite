use frugal_ml::linalg::tensor::{Tensor3D, Tensor4D};
use frugal_ml::scalar::{fabs, Scalar};
use frugal_ml::sp::{cross_correlate2d, filter_bank, Gaussian3D, Sobel3D};

#[test]
fn test_cross_correlate2d_single_frame_two_filters() {
    // 1 2 3
    // 4 5 6
    // 7 8 9
    let frames = Tensor4D::<1, 1, 3, 3, 9>::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

    // filter 0: diagonal (a + d), filter 1: sum of the patch
    let filters = Tensor4D::<2, 1, 2, 2, 8>::new([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

    // each filter becomes an output channel
    let out: Tensor4D<1, 2, 2, 2, 8> = cross_correlate2d(&frames, &filters, 1);
    // diagonals: 1+5, 2+6, 4+8, 5+9
    assert_eq!(6.0, out.get(0, 0, 0, 0));
    assert_eq!(8.0, out.get(0, 0, 1, 0));
    assert_eq!(12.0, out.get(0, 1, 0, 0));
    assert_eq!(14.0, out.get(0, 1, 1, 0));
    // sums: 1+2+4+5, 2+3+5+6, 4+5+7+8, 5+6+8+9
    assert_eq!(12.0, out.get(0, 0, 0, 1));
    assert_eq!(16.0, out.get(0, 0, 1, 1));
    assert_eq!(24.0, out.get(0, 1, 0, 1));
    assert_eq!(28.0, out.get(0, 1, 1, 1));
}

#[test]
fn test_cross_correlate2d_sequence_matches_naive_correlation() {
    // sequence of 2 frames, 2 channels, 4x4, against 3 2x2 filters
    const N: usize = 2;
    const C: usize = 2;
    const H: usize = 4;
    const W: usize = 4;
    const K: usize = 3;
    const KH: usize = 2;
    const KW: usize = 2;
    const H_OUT: usize = 3;
    const W_OUT: usize = 3;

    let mut data = [0.0; 64];
    for i in 0..64 {
        data[i] = (i % 7) as Scalar - 3.0;
    }
    let frames = Tensor4D::<N, C, H, W, 64>::new(data);

    let mut filter_data = [0.0; 24];
    for i in 0..24 {
        filter_data[i] = (i % 5) as Scalar - 2.0;
    }
    let filters = Tensor4D::<K, C, KH, KW, 24>::new(filter_data);

    let out: Tensor4D<N, H_OUT, W_OUT, K, 54> = cross_correlate2d(&frames, &filters, 1);

    // reference: the convolution written out by hand, loop by loop
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
    // and the result isn't trivially zero everywhere
    assert_ne!(0.0, out.get(0, 0, 0, 0));
}

#[test]
fn test_cross_correlate2d_stride_2() {
    // stride 2 on a 4x4: disjoint windows, 2x2 output
    let frames = Tensor4D::<1, 1, 4, 4, 16>::new([
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ]);

    let filters = Tensor4D::<1, 1, 2, 2, 4>::new([1.0, 1.0, 1.0, 1.0]);

    let out: Tensor4D<1, 2, 2, 1, 4> = cross_correlate2d(&frames, &filters, 2);
    assert_eq!(14.0, out.get(0, 0, 0, 0)); // 1+2+5+6
    assert_eq!(22.0, out.get(0, 0, 1, 0)); // 3+4+7+8
    assert_eq!(46.0, out.get(0, 1, 0, 0)); // 9+10+13+14
    assert_eq!(54.0, out.get(0, 1, 1, 0)); // 11+12+15+16
}

#[test]
fn test_cross_correlate2d_gaussian_and_sobel_bank() {
    // 2 single-channel 4x4 frames, filled with a ramp: f(i, j) = 1 + 4i + j for
    // the first frame, +16 for the second
    let mut data = [0.0; 32];
    for i in 0..32 {
        data[i] = (i + 1) as Scalar;
    }
    let frames = Tensor4D::<2, 1, 4, 4, 32>::new(data);

    // a single bank holding both kernels: channel 0 = blur, channel 1 = edge
    let gaussian: Tensor3D<1, 3, 3, 9> = Gaussian3D::kernel();
    let sobel_x: Tensor3D<1, 3, 3, 9> = Sobel3D::x();
    let bank: Tensor4D<2, 1, 3, 3, 18> = filter_bank([&gaussian, &sobel_x]);

    let out: Tensor4D<2, 2, 2, 2, 16> = cross_correlate2d(&frames, &bank, 1);

    // unit-gain gaussian: on a (linear) ramp, it returns the window's
    // center pixel, i.e. f(i + 1, j + 1)
    assert_eq!(6.0, out.get(0, 0, 0, 0));
    assert_eq!(7.0, out.get(0, 0, 1, 0));
    assert_eq!(10.0, out.get(0, 1, 0, 0));
    assert_eq!(11.0, out.get(0, 1, 1, 0));
    // second frame: same ramp shifted by 16
    assert_eq!(22.0, out.get(1, 0, 0, 0));
    assert_eq!(27.0, out.get(1, 1, 1, 0));

    // sobel x: constant gradient of 1 per column, kernel gain 8 -> 8 everywhere
    for n in 0..2 {
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(8.0, out.get(n, i, j, 1));
            }
        }
    }
}

#[test]
fn test_cross_correlate2d_bank_over_two_channels() {
    // the kernels' depth matters here: channel 0 = ramp, channel 1 = zeros.
    // The contraction sums the channels, and the kernels normalize by C, so
    // the response is the average of the per-channel responses.
    let mut data = [0.0; 32];
    for i in 0..16 {
        data[i] = (i + 1) as Scalar;
    }
    let frames = Tensor4D::<1, 2, 4, 4, 32>::new(data);

    let gaussian: Tensor3D<2, 3, 3, 18> = Gaussian3D::kernel();
    let sobel_y: Tensor3D<2, 3, 3, 18> = Sobel3D::y();
    let bank: Tensor4D<2, 2, 3, 3, 36> = filter_bank([&gaussian, &sobel_y]);

    let out: Tensor4D<1, 2, 2, 2, 8> = cross_correlate2d(&frames, &bank, 1);

    // gaussian: (center pixel + 0) / 2
    assert_eq!(3.0, out.get(0, 0, 0, 0));
    assert_eq!(5.5, out.get(0, 1, 1, 0));
    // sobel y: gradient of 4 per row, gain 8 -> 32 on channel 0, 0 on
    // channel 1, average 16
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(16.0, out.get(0, i, j, 1));
        }
    }
}

#[test]
fn test_gaussian_bank_preserves_constant_sequence() {
    // unit DC gain, channels included: a constant sequence comes out identical
    let frames = Tensor4D::<1, 3, 3, 3, 27>::new([7.0; 27]);

    let gaussian: Tensor3D<3, 3, 3, 27> = Gaussian3D::kernel();
    let sobel_x: Tensor3D<3, 3, 3, 27> = Sobel3D::x();
    let bank: Tensor4D<2, 3, 3, 3, 54> = filter_bank([&gaussian, &sobel_x]);

    let out: Tensor4D<1, 1, 1, 2, 2> = cross_correlate2d(&frames, &bank, 1);
    // 1/(16 * 3) isn't exact in binary: compare within epsilon
    assert!(fabs(7.0 - out.get(0, 0, 0, 0)) < 1e-5);
    // and a zero gradient on a flat image
    assert_eq!(0.0, out.get(0, 0, 0, 1));
}
