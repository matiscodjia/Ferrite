use core::f32::EPSILON;

use crate::matrix::Matrix;
use crate::vector::Vector;
use libm::{fabsf, sqrtf};

/// Transforms a set of N vectors of size M into an orthonormal basis.
///
/// In this static version, we return an array of exactly N vectors.
/// If a vector is linearly dependent, it will result in a null vector.
pub fn gram_schmidt<const M: usize, const N: usize>(base: &[Vector<M>; N]) -> [Vector<M>; N] {
    let mut orthogonal_basis = [Vector::<M>::new([0.0; M]); N];

    for i in 0..N {
        let v = &base[i];
        let mut q = *v; // Copy the original vector

        // Subtract projections on all previously computed basis vectors
        for j in 0..i {
            let u = &orthogonal_basis[j];
            let proj = v.orthogonal_projection(u);
            q = q - proj;
        }

        let norm = q.l2_norm();
        if norm > 1e-6 {
            q = q * (1.0 / norm);
            orthogonal_basis[i] = q;
        } else {
            // Keep it as a null vector if it's not linearly independent
            orthogonal_basis[i] = Vector::<M>::new([0.0; M]);
        }
    }
    orthogonal_basis
}

/// QR decomposition of a M x N matrix.
///
/// Returns (Q, R) where:
/// - Q is an M x N orthogonal matrix.
/// - R is an N x N upper triangular matrix.
pub fn qr_decomposition<const M: usize, const N: usize>(
    mat: &Matrix<M, N>,
) -> (Matrix<M, N>, Matrix<N, N>) {
    // 1. Extract columns into a fixed-size array
    let mut cols = [Vector::<M>::new([0.0; M]); N];
    for j in 0..N {
        cols[j] = mat.get_col(j).unwrap();
    }

    // 2. Perform Gram-Schmidt
    let ortho_cols = gram_schmidt::<M, N>(&cols);

    // 3. Create Q from the orthonormal columns
    let q = Matrix::<M, N>::from_cols(ortho_cols);

    // 4. Calculate R = Q^T * mat
    // Resulting R is N x N
    let mut r = Matrix::<N, N>::new();
    r.matmul_accumulate(&q.transpose(), mat);

    (q, r)
}

/// Solves an upper triangular system Rx = b using back-substitution.
///
/// R is an N x N matrix, b is a Vector of size N.
pub fn solve_upper_triangular<const N: usize>(
    r: &Matrix<N, N>,
    b: &Vector<N>,
) -> Option<Vector<N>> {
    let mut x_data = [0.0; N];
    let b_data = b.get_data();

    for i in (0..N).rev() {
        let diag = r[(i, i)];

        if fabsf(diag) < 1e-10 {
            return None; // Singular matrix
        }

        let mut sum = 0.0;
        for j in (i + 1)..N {
            sum += r[(i, j)] * x_data[j];
        }

        x_data[i] = (b_data[i] - sum) / diag;
    }

    Some(Vector::new(x_data))
}

/// Solves a linear system Ax = b using QR decomposition.
///
/// A is M x N, b is size M, result x is size N.
pub fn solve_linear_system<const M: usize, const N: usize>(
    a: &Matrix<M, N>,
    b: &Vector<M>,
) -> Option<Vector<N>> {
    let (q, r) = qr_decomposition(a);

    // Compute c = Q^T * b (Vector of size N)
    let mut c_data = [0.0; N];
    for i in 0..N {
        let q_col = q.get_col(i).unwrap();
        c_data[i] = q_col.dot(b);
    }
    let c = Vector::new(c_data);

    // Solve Rx = c
    solve_upper_triangular(&r, &c)
}

fn sort_svd<const M: usize, const N: usize>(
    sigma: &mut Vector<N>,
    u: &mut Matrix<M, N>,
    v: &mut Matrix<N, N>,
) {
    for i in 0..N {
        let mut max_idx = i;
        for j in (i + 1)..N {
            if sigma[j] > sigma[max_idx] {
                max_idx = j;
            }
        }
        if max_idx != i {
            let tmp = sigma[i];
            sigma[i] = sigma[max_idx];
            sigma[max_idx] = tmp;

            let col_i = u.get_col(i).unwrap();
            let col_max = u.get_col(max_idx).unwrap();
            u.set_col(i, &col_max);
            u.set_col(max_idx, &col_i);

            let col_i = v.get_col(i).unwrap();
            let col_max = v.get_col(max_idx).unwrap();
            v.set_col(i, &col_max);
            v.set_col(max_idx, &col_i);
        }
    }
}

pub fn jacobi_rotation(p: f32, q: f32, d: f32) -> (f32, f32) {
    if fabsf(d) > EPSILON {
        let tau: f32 = (q - p) / (2.0 * d);
        let t = tau.signum() / (fabsf(tau) + sqrtf(1.0 + (tau * tau)));
        let cos = 1.0 / sqrtf(1.0 + (t * t));
        let sin = t * cos;
        (cos, sin)
    } else {
        (1.0, 0.0)
    }
}

pub fn svd_2x2(mat: &Matrix<2, 2>) -> (Matrix<2, 2>, Vector<2>, Matrix<2, 2>) {
    let (a, b) = (mat.get_col(0).unwrap(), mat.get_col(1).unwrap());
    let p = a.dot(&a);
    let q = b.dot(&b);
    let d = a.dot(&b);
    let (cos, sin) = jacobi_rotation(p, q, d);
    let a_prime = a * cos - b * sin;
    let b_prime = a * sin + b * cos;

    let sigma_1 = a_prime.l2_norm();
    let sigma_2 = b_prime.l2_norm();
    let u1 = if sigma_1 > EPSILON { a_prime * (1.0 / sigma_1) } else { a_prime };
    let u2 = if sigma_2 > EPSILON { b_prime * (1.0 / sigma_2) } else { b_prime };

    let u = Matrix::from_cols([u1, u2]);
    let mut v = Matrix::<2, 2>::new();
    v[(0, 0)] = cos;
    v[(0, 1)] = sin;
    v[(1, 0)] = -sin;
    v[(1, 1)] = cos;
    let sigma = Vector::new([sigma_1, sigma_2]);

    (u, sigma, v)
}

pub fn svd<const M: usize, const N: usize>(
    mat: &Matrix<M, N>,
) -> (Matrix<M, N>, Vector<N>, Matrix<N, N>) {
    let mut b = *mat;
    let mut v = Matrix::<N, N>::identity();
    let max_iter = 100 * N * N;
    let mut iter = 0;

    loop {
        let mut converged = true;
        for p in 0..N {
            for q in (p + 1)..N {
                let col_p = b.get_col(p).unwrap();
                let col_q = b.get_col(q).unwrap();

                let dot_pp = col_p.dot(&col_p);
                let dot_qq = col_q.dot(&col_q);
                let dot_pq = col_p.dot(&col_q);

                if fabsf(dot_pq) < EPSILON * sqrtf(dot_pp * dot_qq) {
                    continue;
                }

                converged = false;
                let (cos, sin) = jacobi_rotation(dot_pp, dot_qq, dot_pq);

                let new_b_p = col_p * cos - col_q * sin;
                let new_b_q = col_p * sin + col_q * cos;
                b.set_col(p, &new_b_p);
                b.set_col(q, &new_b_q);

                let v_col_p = v.get_col(p).unwrap();
                let v_col_q = v.get_col(q).unwrap();
                let new_v_p = v_col_p * cos - v_col_q * sin;
                let new_v_q = v_col_p * sin + v_col_q * cos;
                v.set_col(p, &new_v_p);
                v.set_col(q, &new_v_q);
            }
        }
        iter += 1;
        if converged || iter >= max_iter {
            break;
        }
    }

    let mut sigma = Vector::<N>::new([0.0; N]);
    let mut u = Matrix::<M, N>::new();

    for i in 0..N {
        let col = b.get_col(i).unwrap();
        let norm = col.l2_norm();
        sigma[i] = norm;
        if norm > EPSILON {
            u.set_col(i, &(col * (1.0 / norm)));
        } else {
            u.set_col(i, &col);
        }
    }

    sort_svd(&mut sigma, &mut u, &mut v);
    (u, sigma, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gram_schmidt_2d() {
        let v1 = Vector::new([1.0, 1.0]);
        let v2 = Vector::new([0.0, 1.0]);
        let ortho = gram_schmidt::<2, 2>(&[v1, v2]);

        // Check orthogonality
        assert!(ortho[0].dot(&ortho[1]).abs() < 1e-6);
        // Check normality
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

        // A = QR
        assert_eq!(q * r, a);
        // Q is orthogonal
        let i_check = q.transpose() * q;
        assert_eq!(i_check, Matrix::<2, 2>::identity());
    }

    #[test]
    fn test_solve_upper_triangular() {
        let mut r = Matrix::<2, 2>::new();
        r[(0, 0)] = 2.0;
        r[(0, 1)] = 1.0;
        r[(1, 1)] = 1.0;
        let b = Vector::new([5.0, 1.0]);

        let x = solve_upper_triangular(&r, &b).unwrap();
        // 2x + 1y = 5, 1y = 1 => y=1, 2x=4 => x=2
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
        let a = Matrix::<2, 2>::new(); // All zeros
        let b = Vector::new([1.0, 1.0]);
        assert!(solve_linear_system(&a, &b).is_none());
    }

    #[test]
    fn test_gram_schmidt_dependent() {
        let v1 = Vector::new([1.0, 0.0]);
        let v2 = Vector::new([2.0, 0.0]); // Linearly dependent
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
        a[(0, 0)] = 4.0; a[(0, 1)] = 2.0; a[(0, 2)] = 1.0;
        a[(1, 0)] = 2.0; a[(1, 1)] = 3.0; a[(1, 2)] = 1.0;
        a[(2, 0)] = 1.0; a[(2, 1)] = 1.0; a[(2, 2)] = 2.0;

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
        a[(0, 0)] = 5.0; a[(0, 1)] = 1.0; a[(0, 2)] = 2.0; a[(0, 3)] = 0.0;
        a[(1, 0)] = 1.0; a[(1, 1)] = 4.0; a[(1, 2)] = 1.0; a[(1, 3)] = 1.0;
        a[(2, 0)] = 2.0; a[(2, 1)] = 1.0; a[(2, 2)] = 3.0; a[(2, 3)] = 0.0;
        a[(3, 0)] = 0.0; a[(3, 1)] = 1.0; a[(3, 2)] = 0.0; a[(3, 3)] = 2.0;

        let (u, sigma, v) = svd(&a);

        let mut sigma_mat = Matrix::<4, 4>::new();
        for i in 0..4 { sigma_mat[(i, i)] = sigma[i]; }

        assert_eq!(u * sigma_mat * v.transpose(), a);
        assert_eq!(u.transpose() * u, Matrix::<4, 4>::identity());
        assert_eq!(v.transpose() * v, Matrix::<4, 4>::identity());
    }

    // --- Timing (cargo test -- --nocapture pour voir les résultats) ---

    #[test]
    fn bench_matmul() {
        let mut a2 = Matrix::<2, 2>::new();
        a2[(0, 0)] = 1.0; a2[(0, 1)] = 2.0; a2[(1, 0)] = 3.0; a2[(1, 1)] = 4.0;

        let mut a3 = Matrix::<3, 3>::new();
        for i in 0..3 { for j in 0..3 { a3[(i, j)] = (i * 3 + j + 1) as f32; } }

        let mut a4 = Matrix::<4, 4>::new();
        for i in 0..4 { for j in 0..4 { a4[(i, j)] = (i * 4 + j + 1) as f32; } }

        let n = 100_000u32;

        let t = std::time::Instant::now();
        for _ in 0..n { let _ = a2 * a2; }
        println!("matmul 2x2 x{n}: {:?} ({:.1}ns/iter)", t.elapsed(), t.elapsed().as_nanos() as f64 / n as f64);

        let t = std::time::Instant::now();
        for _ in 0..n { let _ = a3 * a3; }
        println!("matmul 3x3 x{n}: {:?} ({:.1}ns/iter)", t.elapsed(), t.elapsed().as_nanos() as f64 / n as f64);

        let t = std::time::Instant::now();
        for _ in 0..n { let _ = a4 * a4; }
        println!("matmul 4x4 x{n}: {:?} ({:.1}ns/iter)", t.elapsed(), t.elapsed().as_nanos() as f64 / n as f64);
    }

    #[test]
    fn bench_svd() {
        let mut a2 = Matrix::<2, 2>::new();
        a2[(0, 0)] = 2.0; a2[(0, 1)] = 1.0; a2[(1, 0)] = 1.0; a2[(1, 1)] = 2.0;

        let mut a3 = Matrix::<3, 3>::new();
        a3[(0, 0)] = 4.0; a3[(0, 1)] = 2.0; a3[(0, 2)] = 1.0;
        a3[(1, 0)] = 2.0; a3[(1, 1)] = 3.0; a3[(1, 2)] = 1.0;
        a3[(2, 0)] = 1.0; a3[(2, 1)] = 1.0; a3[(2, 2)] = 2.0;

        let mut a4 = Matrix::<4, 4>::new();
        a4[(0, 0)] = 5.0; a4[(0, 1)] = 1.0; a4[(0, 2)] = 2.0; a4[(0, 3)] = 0.0;
        a4[(1, 0)] = 1.0; a4[(1, 1)] = 4.0; a4[(1, 2)] = 1.0; a4[(1, 3)] = 1.0;
        a4[(2, 0)] = 2.0; a4[(2, 1)] = 1.0; a4[(2, 2)] = 3.0; a4[(2, 3)] = 0.0;
        a4[(3, 0)] = 0.0; a4[(3, 1)] = 1.0; a4[(3, 2)] = 0.0; a4[(3, 3)] = 2.0;

        let n = 10_000u32;

        let t = std::time::Instant::now();
        for _ in 0..n { let _ = svd_2x2(&a2); }
        println!("svd 2x2 x{n}: {:?} ({:.1}ns/iter)", t.elapsed(), t.elapsed().as_nanos() as f64 / n as f64);

        let t = std::time::Instant::now();
        for _ in 0..n { let _ = svd(&a3); }
        println!("svd 3x3 x{n}: {:?} ({:.1}ns/iter)", t.elapsed(), t.elapsed().as_nanos() as f64 / n as f64);

        let t = std::time::Instant::now();
        for _ in 0..n { let _ = svd(&a4); }
        println!("svd 4x4 x{n}: {:?} ({:.1}ns/iter)", t.elapsed(), t.elapsed().as_nanos() as f64 / n as f64);
    }

    #[test]
    fn test_svd_recomposition() {
        let mut a = Matrix::<2, 2>::new();
        a[(0, 0)] = 2.0;
        a[(0, 1)] = 1.0;
        a[(1, 0)] = 1.0;
        a[(1, 1)] = 2.0;
        let (u, sigma, v) = svd_2x2(&a);
        let mut sigma_mat = Matrix::<2, 2>::new();
        sigma_mat[(0, 0)] = sigma[0];
        sigma_mat[(1, 1)] = sigma[1];
        let tmp = u * sigma_mat;
        let reconstructed = tmp * v.transpose();
        assert_eq!(a, reconstructed);
    }
}
