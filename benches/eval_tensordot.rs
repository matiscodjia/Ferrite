use criterion::{criterion_group, criterion_main, Criterion};
use frugal_ml::linalg::{tensordot_3, Tensor3D, Tensor4D};
use frugal_ml::sp::{filter_bank, Gaussian3D};
use std::hint::black_box;

fn bench_for_k<const K: usize, const NUMEL_BANK: usize, const NUMEL_OUT: usize>(
    c: &mut Criterion,
    tensor: &Tensor4D<1, 3, 128, 128, 49152>,
    filter: &Tensor3D<3, 3, 3, 27>,
    name: &str,
) {
    let filters: Tensor4D<K, 3, 3, 3, NUMEL_BANK> = filter_bank([filter; K]);
    c.bench_function(name, |b| {
        b.iter(|| {
            let out: Tensor4D<1, 126, 126, K, NUMEL_OUT> = tensordot_3(
                black_box(&tensor.im2col_view::<126, 126, 3, 3>(1)),
                black_box(&filters),
            );
            black_box(out)
        })
    });
}

fn bench_tensordot(c: &mut Criterion) {
    let data = (0..49152).map(|x| x as f32).collect();
    let tensor = Tensor4D::<1, 3, 128, 128, 49152>::from_vec(data).unwrap();
    // filter channels = tensor channels (3): filter_bank requires a kernel
    // that covers the whole input depth, not just 1 channel.
    let filter: Tensor3D<3, 3, 3, 27> = Gaussian3D::kernel();

    bench_for_k::<1, 27, 15876>(c, &tensor, &filter, "3D tensor contraction, K=1");
    bench_for_k::<2, 54, 31752>(c, &tensor, &filter, "3D tensor contraction, K=2");
    bench_for_k::<4, 108, 63504>(c, &tensor, &filter, "3D tensor contraction, K=4");
    bench_for_k::<8, 216, 127008>(c, &tensor, &filter, "3D tensor contraction, K=8");
    bench_for_k::<16, 432, 254016>(c, &tensor, &filter, "3D tensor contraction, K=16");
}

criterion_group!(benches, bench_tensordot);
criterion_main!(benches);
