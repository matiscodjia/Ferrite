use ferrite::tensor::Tensor;

#[test]
//This is a compilation level test, test NOK = No compilation
fn test_tensor_creation_and_shape() {
    let m = Tensor::<2, 3, 6>::new();
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
