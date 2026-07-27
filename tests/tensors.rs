use ferrite::scalar::Scalar;
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
