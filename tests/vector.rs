use ferrite::Vector;

#[test]
fn test_vector_creation_and_dim() {
    let v = Vector::new([1.0, 2.0, 3.0]);
    assert_eq!(v.dim(), 3);
}

#[test]
fn test_vector_l1_norm() {
    let v = Vector::new([1.0, -2.0, 3.0]);
    assert_eq!(v.l1_norm(), 6.0);
}

#[test]
fn test_vector_l2_norm() {
    let v = Vector::new([3.0, 4.0]);
    assert_eq!(v.l2_norm(), 5.0);
}

#[test]
fn test_vector_inf_norm() {
    let v = Vector::new([-10.0, 2.0, 5.0]);
    assert_eq!(v.inf_norm(), 10.0);
}

#[test]
fn test_vector_dot_product() {
    let v1 = Vector::new([1.0, 2.0]);
    let v2 = Vector::new([3.0, 4.0]);
    assert_eq!(v1.dot(&v2), 11.0);
}

#[test]
fn test_vector_addition() {
    let v1 = Vector::new([1.0, 2.0]);
    let v2 = Vector::new([3.0, 4.0]);
    assert_eq!(&v1 + &v2, Vector::new([4.0, 6.0]));
}

#[test]
fn test_vector_subtraction() {
    let v1 = Vector::new([5.0, 7.0]);
    let v2 = Vector::new([2.0, 3.0]);
    assert_eq!(&v1 - &v2, Vector::new([3.0, 4.0]));
}

#[test]
fn test_vector_scalar_mul() {
    let v = Vector::new([1.0, -2.0]);
    assert_eq!(&v * 3.0, Vector::new([3.0, -6.0]));
}

#[test]
fn test_vector_projection() {
    let v = Vector::new([1.0, 1.0]);
    let target = Vector::new([1.0, 0.0]);
    assert_eq!(v.orthogonal_projection(&target), Vector::new([1.0, 0.0]));
}

#[test]
fn test_vector_null_projection() {
    let v = Vector::new([1.0, 2.0]);
    let null_v = Vector::<2>::new([0.0, 0.0]);
    assert_eq!(v.orthogonal_projection(&null_v), Vector::new([0.0, 0.0]));
}
