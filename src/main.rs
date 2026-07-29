use ferrite::io::npy::{read_npy, write_npy};
fn main() {
    let path = std::path::Path::new("tests/fixtures/smoke.npy");
    let shape = vec![2, 3];
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    // Écrire
    //write_npy(path, &shape, &data).expect("write failed");
    //println!("✓ Written");

    // Lire
    let result = read_npy(path).expect("read failed");
    println!("Shape: {:?}", result.shape);
    println!("Data: {:?}", result.data);

    // Vérifier
    assert_eq!(result.shape, shape);
    assert_eq!(result.data, data);
    println!("✓ Test passed");
}
