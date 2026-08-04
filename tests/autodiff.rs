mod iris_data;

use ferrite::autodiff::core::module::Module;
use ferrite::autodiff::layers::activations::{ReLU, Softmax, Tanh};
use ferrite::autodiff::layers::linear::Linear;
use ferrite::autodiff::loss::{cross_entropy, mse};
use ferrite::autodiff::optim::sgd::Sgd;
use ferrite::autodiff::optim::train::train_step;
use ferrite::linalg::tensor::{Tensor, Vector};
use ferrite::seq;
use ferrite::Scalar;

#[test]
fn test_integration_vector_tensor() {
    let v = Vector::from_data([1.0, 2.0]);
    assert_eq!(v.dim(), 2);
    let m = Tensor::<2, 2, 4>::new();
    assert_eq!(m.rows(), 2);
}

#[test]
fn test_mlp_training_step() {
    let mut network = seq!(
        Linear::<2, 4, 8>::from_seed(42),
        ReLU::<4> {},
        Linear::<4, 1, 4>::from_seed(137)
    );
    let input = Vector::from_data([1.0, 0.5]);
    let target = Vector::from_data([1.0]);

    let mut opt = Sgd::new(0.01);
    let loss = train_step(&mut network, &mut opt, input, target, mse);

    let (output2, _) = network.forward(input);
    let (loss2, _) = mse(output2, target);
    assert!(
        loss2 < loss,
        "loss should decrease after one SGD step: {loss2} >= {loss}"
    );
}

// cargo test test_convergence -- --nocapture
#[test]
fn test_convergence() {
    let dataset: [(Vector<2>, Vector<1>); 4] = [
        (Vector::from_data([1.0, 1.0]), Vector::from_data([1.0])),
        (Vector::from_data([1.0, -1.0]), Vector::from_data([-1.0])),
        (Vector::from_data([-1.0, 1.0]), Vector::from_data([-1.0])),
        (Vector::from_data([-1.0, -1.0]), Vector::from_data([1.0])),
    ];

    let mut network = seq!(
        Linear::<2, 8, 16>::from_seed(42),
        Tanh::<8> {},
        Linear::<8, 1, 8>::from_seed(137)
    );

    let mut opt = Sgd::new(0.05);
    for epoch in 0..=2000 {
        let mut total_loss = 0.0;
        for &(input, target) in &dataset {
            total_loss += train_step(&mut network, &mut opt, input, target, mse);
        }
        if epoch % 200 == 0 {
            println!(
                "epoch {:4} | loss {:.6}",
                epoch,
                total_loss / dataset.len() as Scalar
            );
        }
    }

    let mut final_loss = 0.0;
    for &(input, target) in &dataset {
        let (output, _) = network.forward(input);
        final_loss += mse(output, target).0;
    }
    final_loss /= dataset.len() as Scalar;

    println!("loss finale : {final_loss:.6}");
    assert!(
        final_loss < 0.05,
        "réseau n'a pas convergé : loss finale = {final_loss:.6}"
    );
}

// cargo test test_classifier -- --nocapture
#[test]
fn test_classifier() {
    let dataset: [(Vector<4>, Vector<3>); 6] = [
        (
            Vector::from_data([1.0, 0.0, 0.0, 0.0]),
            Vector::from_data([1.0, 0.0, 0.0]),
        ),
        (
            Vector::from_data([0.9, 0.1, 0.0, 0.0]),
            Vector::from_data([1.0, 0.0, 0.0]),
        ),
        (
            Vector::from_data([0.0, 1.0, 0.0, 0.0]),
            Vector::from_data([0.0, 1.0, 0.0]),
        ),
        (
            Vector::from_data([0.1, 0.8, 0.1, 0.0]),
            Vector::from_data([0.0, 1.0, 0.0]),
        ),
        (
            Vector::from_data([0.0, 0.0, 1.0, 1.0]),
            Vector::from_data([0.0, 0.0, 1.0]),
        ),
        (
            Vector::from_data([0.0, 0.1, 0.8, 0.9]),
            Vector::from_data([0.0, 0.0, 1.0]),
        ),
    ];

    let mut network = seq!(
        Linear::<4, 8, 32>::from_seed(42),
        Tanh::<8> {},
        Linear::<8, 3, 24>::from_seed(137),
        Softmax::<3> {}
    );

    let mut opt = Sgd::new(0.05);
    for epoch in 0..=1000 {
        let mut total_loss = 0.0;
        for &(input, target) in &dataset {
            total_loss += train_step(&mut network, &mut opt, input, target, cross_entropy);
        }
        if epoch % 100 == 0 {
            println!(
                "epoch {:4} | loss {:.6}",
                epoch,
                total_loss / dataset.len() as Scalar
            );
        }
    }

    let mut correct = 0;
    for &(input, target) in &dataset {
        let (probs, _) = network.forward(input);
        let pred = (0..3)
            .max_by(|&a, &b| probs[a].partial_cmp(&probs[b]).unwrap())
            .unwrap();
        let true_class = (0..3)
            .max_by(|&a, &b| target[a].partial_cmp(&target[b]).unwrap())
            .unwrap();
        if pred == true_class {
            correct += 1;
        }
    }
    println!("accuracy : {}/{}", correct, dataset.len());
    assert_eq!(correct, dataset.len(), "le classifieur n'a pas convergé");
}

// cargo test test_iris -- --nocapture
#[test]
fn test_iris() {
    use iris_data::{FEATURE_MAX, FEATURE_MIN, IRIS};

    let normalize = |raw: [f32; 4]| -> Vector<4> {
        let mut data = [0.0; 4];
        for i in 0..4 {
            data[i] = (raw[i] as Scalar - FEATURE_MIN[i] as Scalar)
                / (FEATURE_MAX[i] as Scalar - FEATURE_MIN[i] as Scalar);
        }
        Vector::from_data(data)
    };

    let to_onehot = |class: usize| -> Vector<3> {
        let mut data = [0.0; 3];
        data[class] = 1.0;
        Vector::from_data(data)
    };

    let mut train: [(Vector<4>, Vector<3>); 120] =
        [(Vector::from_data([0.0; 4]), Vector::from_data([0.0; 3])); 120];
    let mut test: [(Vector<4>, Vector<3>); 30] =
        [(Vector::from_data([0.0; 4]), Vector::from_data([0.0; 3])); 30];

    for class in 0..3 {
        for i in 0..40 {
            let idx = class * 50 + i;
            train[class * 40 + i] = (normalize(IRIS[idx].0), to_onehot(IRIS[idx].1));
        }
        for i in 0..10 {
            let idx = class * 50 + 40 + i;
            test[class * 10 + i] = (normalize(IRIS[idx].0), to_onehot(IRIS[idx].1));
        }
    }

    let mut network = seq!(
        Linear::<4, 16, 64>::from_seed(42),
        Tanh::<16> {},
        Linear::<16, 3, 48>::from_seed(137),
        Softmax::<3> {}
    );

    let mut opt = Sgd::new(0.05);
    for epoch in 0..=2000 {
        let mut total_loss = 0.0;
        for &(input, target) in &train {
            total_loss += train_step(&mut network, &mut opt, input, target, cross_entropy);
        }
        if epoch % 200 == 0 {
            println!(
                "epoch {:4} | train loss {:.4}",
                epoch,
                total_loss / train.len() as Scalar
            );
        }
    }

    let accuracy = |dataset: &[(Vector<4>, Vector<3>)]| -> usize {
        let mut correct = 0;
        for &(input, target) in dataset {
            let (probs, _) = network.forward(input);
            let pred = (0..3)
                .max_by(|&a, &b| probs[a].partial_cmp(&probs[b]).unwrap())
                .unwrap();
            let true_class = (0..3)
                .max_by(|&a, &b| target[a].partial_cmp(&target[b]).unwrap())
                .unwrap();
            if pred == true_class {
                correct += 1;
            }
        }
        correct
    };

    let train_acc = accuracy(&train);
    let test_acc = accuracy(&test);
    println!(
        "train accuracy : {}/{} ({:.1}%)",
        train_acc,
        train.len(),
        100.0 * train_acc as Scalar / train.len() as Scalar
    );
    println!(
        "test  accuracy : {}/{} ({:.1}%)",
        test_acc,
        test.len(),
        100.0 * test_acc as Scalar / test.len() as Scalar
    );
    assert!(test_acc >= 27, "accuracy trop basse : {}/30", test_acc);
}

// cargo test bench_autodiff -- --nocapture
#[test]
fn bench_autodiff() {
    let input = Vector::from_data([1.0, 0.5]);
    let target = Vector::from_data([1.0]);

    let mut net_s = seq!(
        Linear::<2, 4, 8>::from_seed(42),
        Tanh::<4> {},
        Linear::<4, 1, 4>::from_seed(99)
    );
    let mut net_m = seq!(
        Linear::<2, 8, 16>::from_seed(42),
        Tanh::<8> {},
        Linear::<8, 4, 32>::from_seed(99),
        Tanh::<4> {},
        Linear::<4, 1, 4>::from_seed(7)
    );

    let n = 10_000u32;

    let t = std::time::Instant::now();
    for _ in 0..n {
        let _ = net_s.forward(input);
    }
    println!(
        "forward  2→4→1   x{n}: {:?} ({:.1}ns/iter)",
        t.elapsed(),
        t.elapsed().as_nanos() as f64 / n as f64
    );

    let t = std::time::Instant::now();
    for _ in 0..n {
        let _ = net_m.forward(input);
    }
    println!(
        "forward  2→8→4→1 x{n}: {:?} ({:.1}ns/iter)",
        t.elapsed(),
        t.elapsed().as_nanos() as f64 / n as f64
    );

    let t = std::time::Instant::now();
    for _ in 0..n {
        let (output, ctx) = net_s.forward(input);
        let (_, grad) = mse(output, target);
        let _ = net_s.backward(grad, &ctx);
    }
    println!(
        "fwd+bwd  2→4→1   x{n}: {:?} ({:.1}ns/iter)",
        t.elapsed(),
        t.elapsed().as_nanos() as f64 / n as f64
    );

    let t = std::time::Instant::now();
    for _ in 0..n {
        let (output, ctx) = net_m.forward(input);
        let (_, grad) = mse(output, target);
        let _ = net_m.backward(grad, &ctx);
    }
    println!(
        "fwd+bwd  2→8→4→1 x{n}: {:?} ({:.1}ns/iter)",
        t.elapsed(),
        t.elapsed().as_nanos() as f64 / n as f64
    );

    let mut opt = Sgd::new(0.01);
    let t = std::time::Instant::now();
    for _ in 0..n {
        train_step(&mut net_s, &mut opt, input, target, mse);
    }
    println!(
        "step     2→4→1   x{n}: {:?} ({:.1}ns/iter)",
        t.elapsed(),
        t.elapsed().as_nanos() as f64 / n as f64
    );

    let t = std::time::Instant::now();
    for _ in 0..n {
        train_step(&mut net_m, &mut opt, input, target, mse);
    }
    println!(
        "step     2→8→4→1 x{n}: {:?} ({:.1}ns/iter)",
        t.elapsed(),
        t.elapsed().as_nanos() as f64 / n as f64
    );
}
