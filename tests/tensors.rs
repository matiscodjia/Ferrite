use ferrite::tensor::Tensor;

#[test]
fn test_tensor_creation_and_shape() {
    let _m = Tensor::<2, 3, 6>::new();
}

#[test]
fn test_indexing() {
    let m = Tensor::<2, 3, 6>::new();
    assert_eq!(0.0, m.get(0, 0));
}

#[test]
#[should_panic]
fn test_indexing_not_valid() {
    let m = Tensor::<2, 3, 6>::new();
    m.get(12, 2);
}

#[test]
fn test_setting() {
    let mut m = Tensor::<2, 3, 6>::new();
    m.set(0, 0, 2.0);
    assert_eq!(2.0, m.get(0, 0));
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
