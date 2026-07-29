use crate::linalg::matrix::Matrix;
use crate::linalg::vector::Vector;
use crate::scalar::{fabs, sqrt, Scalar};

const EPSILON: Scalar = Scalar::EPSILON;

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

        if fabs(diag) < 1e-10 {
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

pub fn jacobi_rotation(p: Scalar, q: Scalar, d: Scalar) -> (Scalar, Scalar) {
    if fabs(d) > EPSILON {
        let tau: Scalar = (q - p) / (2.0 * d);
        let t = tau.signum() / (fabs(tau) + sqrt(1.0 + (tau * tau)));
        let cos = 1.0 / sqrt(1.0 + (t * t));
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
    let u1 = if sigma_1 > EPSILON {
        a_prime * (1.0 / sigma_1)
    } else {
        a_prime
    };
    let u2 = if sigma_2 > EPSILON {
        b_prime * (1.0 / sigma_2)
    } else {
        b_prime
    };

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

                if fabs(dot_pq) < EPSILON * sqrt(dot_pp * dot_qq) {
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
