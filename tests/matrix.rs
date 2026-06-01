use ferrite::{Matrix, Vector};

#[test]
fn test_matrix_creation_and_size() {
    let m = Matrix::<2, 3>::new();
    assert_eq!(m.rows(), 2);
    assert_eq!(m.cols(), 3);
}

#[test]
fn test_matrix_get_set() {
    let mut m = Matrix::<2, 2>::new();
    m[(0, 1)] = 42.0;
    assert_eq!(m[(0, 1)], 42.0);
    assert_eq!(m[(1, 1)], 0.0);
}

#[test]
#[should_panic]
fn test_matrix_out_of_bounds() {
    let m = Matrix::<2, 2>::new();
    let _ = m[(2, 0)];
}

#[test]
fn test_matrix_addition() {
    let mut m1 = Matrix::<2, 2>::new();
    m1[(0, 0)] = 1.0;
    let mut m2 = Matrix::<2, 2>::new();
    m2[(0, 0)] = 2.0;
    assert_eq!(m1 + m2, {
        let mut res = Matrix::<2, 2>::new();
        res[(0, 0)] = 3.0;
        res
    });
}

#[test]
fn test_matrix_multiplication() {
    let mut m1 = Matrix::<2, 2>::new();
    m1[(0, 0)] = 1.0; m1[(0, 1)] = 2.0;
    m1[(1, 0)] = 3.0; m1[(1, 1)] = 4.0;
    let mut m2 = Matrix::<2, 1>::new();
    m2[(0, 0)] = 5.0; m2[(1, 0)] = 6.0;
    let res = m1 * m2;
    assert_eq!(res[(0, 0)], 17.0);
    assert_eq!(res[(1, 0)], 39.0);
}

#[test]
fn test_matmul_accumulate() {
    let mut res = Matrix::<1, 1>::new();
    res[(0, 0)] = 10.0;
    let m1 = Matrix::<1, 1>::identity();
    let m2 = Matrix::<1, 1>::identity();
    res.matmul_accumulate(&m1, &m2);
    assert_eq!(res[(0, 0)], 11.0);
}

#[test]
fn test_matrix_transpose() {
    let mut m = Matrix::<1, 2>::new();
    m[(0, 0)] = 1.0; m[(0, 1)] = 2.0;
    let t = m.transpose();
    assert_eq!(t.rows(), 2);
    assert_eq!(t.cols(), 1);
    assert_eq!(t[(1, 0)], 2.0);
}

#[test]
fn test_matrix_col_extraction() {
    let mut m = Matrix::<2, 2>::new();
    m[(0, 1)] = 5.0; m[(1, 1)] = 10.0;
    let col = m.get_col(1).unwrap();
    assert_eq!(col, Vector::new([5.0, 10.0]));
}

#[test]
fn test_matrix_from_cols() {
    let v1 = Vector::new([1.0, 2.0]);
    let v2 = Vector::new([3.0, 4.0]);
    let m = Matrix::from_cols([v1, v2]);
    assert_eq!(m[(1, 0)], 2.0);
    assert_eq!(m[(1, 1)], 4.0);
}

#[test]
fn test_matrix_identity() {
    let id = Matrix::<3, 3>::identity();
    assert_eq!(id[(0, 0)], 1.0);
    assert_eq!(id[(0, 1)], 0.0);
    assert_eq!(id[(1, 1)], 1.0);
    assert_eq!(id[(2, 2)], 1.0);
}
