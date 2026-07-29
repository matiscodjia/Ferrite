use ferrite::linalg::decomposition::{
    gram_schmidt, jacobi_rotation, qr_decomposition, solve_linear_system, solve_upper_triangular,
    svd, svd_2x2,
};
use ferrite::{Matrix, Scalar, Vector};

#[test]
fn test_gram_schmidt_2d() {
    let v1 = Vector::new([1.0, 1.0]);
    let v2 = Vector::new([0.0, 1.0]);
    let ortho = gram_schmidt::<2, 2>(&[v1, v2]);
    assert!(ortho[0].dot(&ortho[1]).abs() < 1e-6);
    assert!((ortho[0].l2_norm() - 1.0).abs() < 1e-6);
    assert!((ortho[1].l2_norm() - 1.0).abs() < 1e-6);
}

#[test]
fn test_qr_decomposition_simple() {
    let mut a = Matrix::<2, 2>::new();
    a[(0, 0)] = 1.0;
    a[(0, 1)] = 1.0;
    a[(1, 0)] = 0.0;
    a[(1, 1)] = 1.0;
    let (q, r) = qr_decomposition(&a);
    assert_eq!(q * r, a);
    assert_eq!(q.transpose() * q, Matrix::<2, 2>::identity());
}

#[test]
fn test_solve_upper_triangular() {
    let mut r = Matrix::<2, 2>::new();
    r[(0, 0)] = 2.0;
    r[(0, 1)] = 1.0;
    r[(1, 1)] = 1.0;
    let b = Vector::new([5.0, 1.0]);
    let x = solve_upper_triangular(&r, &b).unwrap();
    assert_eq!(x, Vector::new([2.0, 1.0]));
}

#[test]
fn test_solve_linear_system_2d() {
    let mut a = Matrix::<2, 2>::new();
    a[(0, 0)] = 1.0;
    a[(0, 1)] = 1.0;
    a[(1, 0)] = 1.0;
    a[(1, 1)] = -1.0;
    let b = Vector::new([3.0, 1.0]);
    let x = solve_linear_system(&a, &b).unwrap();
    assert_eq!(x, Vector::new([2.0, 1.0]));
}

#[test]
fn test_singular_system() {
    let a = Matrix::<2, 2>::new();
    let b = Vector::new([1.0, 1.0]);
    assert!(solve_linear_system(&a, &b).is_none());
}

#[test]
fn test_gram_schmidt_dependent() {
    let v1 = Vector::new([1.0, 0.0]);
    let v2 = Vector::new([2.0, 0.0]);
    let ortho = gram_schmidt::<2, 2>(&[v1, v2]);
    assert_eq!(ortho[1], Vector::new([0.0, 0.0]));
}

#[test]
fn test_qr_3x2_matrix() {
    let mut a = Matrix::<3, 2>::new();
    a[(0, 0)] = 12.0;
    a[(0, 1)] = -51.0;
    a[(1, 0)] = 6.0;
    a[(1, 1)] = 167.0;
    a[(2, 0)] = -4.0;
    a[(2, 1)] = 24.0;
    let (q, r) = qr_decomposition(&a);
    assert_eq!(q * r, a);
}

#[test]
fn test_back_substitution_3d() {
    let mut r = Matrix::<3, 3>::new();
    r[(0, 0)] = 1.0;
    r[(0, 1)] = 2.0;
    r[(0, 2)] = 3.0;
    r[(1, 1)] = 1.0;
    r[(1, 2)] = 2.0;
    r[(2, 2)] = 1.0;
    let b = Vector::new([6.0, 3.0, 1.0]);
    let x = solve_upper_triangular(&r, &b).unwrap();
    assert_eq!(x, Vector::new([1.0, 1.0, 1.0]));
}

#[test]
fn test_identity_solver() {
    let a = Matrix::<3, 3>::identity();
    let b = Vector::new([1.0, 2.0, 3.0]);
    let x = solve_linear_system(&a, &b).unwrap();
    assert_eq!(x, b);
}

#[test]
fn test_orthogonal_projection_consistency() {
    let v = Vector::new([1.0, 2.0, 3.0]);
    let u = Vector::new([1.0, 0.0, 0.0]);
    let proj = v.orthogonal_projection(&u);
    assert_eq!(proj, Vector::new([1.0, 0.0, 0.0]));
}

#[test]
fn test_svd_2x2() {
    let mut a = Matrix::<2, 2>::new();
    a[(0, 0)] = 2.0;
    a[(0, 1)] = 1.0;
    a[(1, 0)] = 1.0;
    a[(1, 1)] = 2.0;
    let (u, _, _) = svd_2x2(&a);
    assert!(u.get_col(0).unwrap().dot(&u.get_col(1).unwrap()).abs() < 1e-5);
}

#[test]
fn test_svd_reconstruction_3x3() {
    let mut a = Matrix::<3, 3>::new();
    a[(0, 0)] = 4.0;
    a[(0, 1)] = 2.0;
    a[(0, 2)] = 1.0;
    a[(1, 0)] = 2.0;
    a[(1, 1)] = 3.0;
    a[(1, 2)] = 1.0;
    a[(2, 0)] = 1.0;
    a[(2, 1)] = 1.0;
    a[(2, 2)] = 2.0;
    let (u, sigma, v) = svd(&a);
    let mut sigma_mat = Matrix::<3, 3>::new();
    sigma_mat[(0, 0)] = sigma[0];
    sigma_mat[(1, 1)] = sigma[1];
    sigma_mat[(2, 2)] = sigma[2];
    assert_eq!(u * sigma_mat * v.transpose(), a);
    assert_eq!(u.transpose() * u, Matrix::<3, 3>::identity());
    assert_eq!(v.transpose() * v, Matrix::<3, 3>::identity());
}

#[test]
fn test_svd_identity_3x3() {
    let a = Matrix::<3, 3>::identity();
    let (_, sigma, _) = svd(&a);
    for i in 0..3 {
        assert!((sigma[i] - 1.0).abs() < 1e-5, "sigma[{i}] = {}", sigma[i]);
    }
}

#[test]
fn test_svd_reconstruction_4x4() {
    let mut a = Matrix::<4, 4>::new();
    a[(0, 0)] = 5.0;
    a[(0, 1)] = 1.0;
    a[(0, 2)] = 2.0;
    a[(0, 3)] = 0.0;
    a[(1, 0)] = 1.0;
    a[(1, 1)] = 4.0;
    a[(1, 2)] = 1.0;
    a[(1, 3)] = 1.0;
    a[(2, 0)] = 2.0;
    a[(2, 1)] = 1.0;
    a[(2, 2)] = 3.0;
    a[(2, 3)] = 0.0;
    a[(3, 0)] = 0.0;
    a[(3, 1)] = 1.0;
    a[(3, 2)] = 0.0;
    a[(3, 3)] = 2.0;
    let (u, sigma, v) = svd(&a);
    let mut sigma_mat = Matrix::<4, 4>::new();
    for i in 0..4 {
        sigma_mat[(i, i)] = sigma[i];
    }
    assert_eq!(u * sigma_mat * v.transpose(), a);
    assert_eq!(u.transpose() * u, Matrix::<4, 4>::identity());
    assert_eq!(v.transpose() * v, Matrix::<4, 4>::identity());
}

#[test]
fn bench_matmul() {
    let mut a2 = Matrix::<2, 2>::new();
    a2[(0, 0)] = 1.0;
    a2[(0, 1)] = 2.0;
    a2[(1, 0)] = 3.0;
    a2[(1, 1)] = 4.0;
    let mut a3 = Matrix::<3, 3>::new();
    for i in 0..3 {
        for j in 0..3 {
            a3[(i, j)] = (i * 3 + j + 1) as Scalar;
        }
    }
    let mut a4 = Matrix::<4, 4>::new();
    for i in 0..4 {
        for j in 0..4 {
            a4[(i, j)] = (i * 4 + j + 1) as Scalar;
        }
    }
    let n = 100_000u32;
    let t = std::time::Instant::now();
    for _ in 0..n {
        let _ = a2 * a2;
    }
    println!(
        "matmul 2x2 x{n}: {:?} ({:.1}ns/iter)",
        t.elapsed(),
        t.elapsed().as_nanos() as f64 / n as f64
    );
    let t = std::time::Instant::now();
    for _ in 0..n {
        let _ = a3 * a3;
    }
    println!(
        "matmul 3x3 x{n}: {:?} ({:.1}ns/iter)",
        t.elapsed(),
        t.elapsed().as_nanos() as f64 / n as f64
    );
    let t = std::time::Instant::now();
    for _ in 0..n {
        let _ = a4 * a4;
    }
    println!(
        "matmul 4x4 x{n}: {:?} ({:.1}ns/iter)",
        t.elapsed(),
        t.elapsed().as_nanos() as f64 / n as f64
    );
}

#[test]
fn bench_svd() {
    let mut a2 = Matrix::<2, 2>::new();
    a2[(0, 0)] = 2.0;
    a2[(0, 1)] = 1.0;
    a2[(1, 0)] = 1.0;
    a2[(1, 1)] = 2.0;
    let mut a3 = Matrix::<3, 3>::new();
    a3[(0, 0)] = 4.0;
    a3[(0, 1)] = 2.0;
    a3[(0, 2)] = 1.0;
    a3[(1, 0)] = 2.0;
    a3[(1, 1)] = 3.0;
    a3[(1, 2)] = 1.0;
    a3[(2, 0)] = 1.0;
    a3[(2, 1)] = 1.0;
    a3[(2, 2)] = 2.0;
    let mut a4 = Matrix::<4, 4>::new();
    a4[(0, 0)] = 5.0;
    a4[(0, 1)] = 1.0;
    a4[(0, 2)] = 2.0;
    a4[(0, 3)] = 0.0;
    a4[(1, 0)] = 1.0;
    a4[(1, 1)] = 4.0;
    a4[(1, 2)] = 1.0;
    a4[(1, 3)] = 1.0;
    a4[(2, 0)] = 2.0;
    a4[(2, 1)] = 1.0;
    a4[(2, 2)] = 3.0;
    a4[(2, 3)] = 0.0;
    a4[(3, 0)] = 0.0;
    a4[(3, 1)] = 1.0;
    a4[(3, 2)] = 0.0;
    a4[(3, 3)] = 2.0;
    let n = 10_000u32;
    let t = std::time::Instant::now();
    for _ in 0..n {
        let _ = svd_2x2(&a2);
    }
    println!(
        "svd 2x2 x{n}: {:?} ({:.1}ns/iter)",
        t.elapsed(),
        t.elapsed().as_nanos() as f64 / n as f64
    );
    let t = std::time::Instant::now();
    for _ in 0..n {
        let _ = svd(&a3);
    }
    println!(
        "svd 3x3 x{n}: {:?} ({:.1}ns/iter)",
        t.elapsed(),
        t.elapsed().as_nanos() as f64 / n as f64
    );
    let t = std::time::Instant::now();
    for _ in 0..n {
        let _ = svd(&a4);
    }
    println!(
        "svd 4x4 x{n}: {:?} ({:.1}ns/iter)",
        t.elapsed(),
        t.elapsed().as_nanos() as f64 / n as f64
    );
}

#[test]
fn test_svd_recomposition() {
    let mut a = Matrix::<2, 2>::new();
    a[(0, 0)] = 3.0;
    a[(0, 1)] = 1.0;
    a[(1, 0)] = 1.0;
    a[(1, 1)] = 3.0;
    let (u, sigma, v) = svd_2x2(&a);
    let mut s = Matrix::<2, 2>::new();
    s[(0, 0)] = sigma[0];
    s[(1, 1)] = sigma[1];
    assert_eq!(u * s * v.transpose(), a);
}

#[test]
fn test_jacobi_rotation_zero() {
    let (cos, sin) = jacobi_rotation(1.0, 1.0, 0.0);
    assert_eq!(cos, 1.0);
    assert_eq!(sin, 0.0);
}
