use criterion::{criterion_group, criterion_main, Criterion};
use ferrite::linalg::{tensordot_3, Tensor3D, Tensor4D};
use ferrite::sp::{filter_bank, Gaussian3D};
use std::hint::black_box;

fn bench_tensordot(c: &mut Criterion) {
    let mut tensor = Tensor4D::<1, 3, 128, 128, 49152>::new();
    let data = (0..49152).map(|x| x as f32).collect();
    tensor.load_vec(data).unwrap();
    // canaux du filtre = canaux du tenseur (3) : filter_bank exige un kernel
    // qui couvre toute la profondeur d'entrée, pas seulement 1 canal.
    let filter: Tensor3D<3, 3, 3, 27> = Gaussian3D::kernel();
    let filters: Tensor4D<1, 3, 3, 3, 27> = filter_bank([&filter; 1]);

    c.bench_function("contraction tensorielle 3D", |b| {
        b.iter(|| {
            let out: Tensor4D<1, 126, 126, 1, 15876> = tensordot_3(
                black_box(&tensor.im2col_view::<126, 126, 3, 3>(1)),
                black_box(&filters),
            );
            black_box(out)
        })
    });
}

criterion_group!(benches, bench_tensordot);
criterion_main!(benches);
