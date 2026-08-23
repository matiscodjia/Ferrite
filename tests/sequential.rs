use frugal_ml::autodiff::core::module::Module;
use frugal_ml::autodiff::grad_check::GradChecker;
use frugal_ml::autodiff::layers::activations::{LeakyReLU, ReLU, Tanh};
use frugal_ml::autodiff::layers::linear::Linear;
use frugal_ml::autodiff::loss::mse;
use frugal_ml::autodiff::optim::sgd::Sgd;
use frugal_ml::autodiff::optim::train::train_step;
use frugal_ml::linalg::tensor::Vector;
use frugal_ml::seq;
use frugal_ml::Scalar;

#[test]
fn seq_2_layers_forward_backward() {
    let net = seq!(Linear::<2, 4, 8>::from_seed(42), ReLU::<4> {});
    let input = Vector::from_data([1.0, 0.5]);
    let (output, ctx) = net.forward(input);
    let _ = net.backward(output, &ctx);
}

#[test]
fn seq_3_layers_loss() {
    let net = seq!(
        Linear::<2, 4, 8>::from_seed(42),
        Tanh::<4> {},
        Linear::<4, 1, 4>::from_seed(99)
    );
    let input = Vector::from_data([1.0, 0.5]);
    let target = Vector::from_data([1.0]);
    let (output, ctx) = net.forward(input);
    let (_, loss_grad) = mse(output, target);
    let _ = net.backward(loss_grad, &ctx);
}

#[test]
fn seq_single_layer() {
    let net = seq!(Linear::<3, 2, 6>::from_seed(7));
    let input = Vector::from_data([1.0, 0.0, -1.0]);
    let (output, ctx) = net.forward(input);
    let _ = net.backward(output, &ctx);
}

#[test]
fn seq_training_step_decreases_loss() {
    let mut net = seq!(
        Linear::<2, 8, 16>::from_seed(42),
        Tanh::<8> {},
        Linear::<8, 1, 8>::from_seed(137)
    );
    let input = Vector::from_data([1.0, 0.5]);
    let target = Vector::from_data([1.0]);

    let mut opt = Sgd::new(0.01);
    let loss = train_step(&mut net, &mut opt, input, target, mse);

    let (output2, _) = net.forward(input);
    let (loss2, _) = mse(output2, target);
    assert!(
        loss2 < loss,
        "loss should decrease after one SGD step: {loss2} >= {loss}"
    );
}

#[test]
fn seq_training_step_decreases_loss_leaky_relu() {
    let mut net = seq!(
        Linear::<2, 8, 16>::from_seed(42),
        LeakyReLU::<8>::new(0.01),
        Linear::<8, 1, 8>::from_seed(137)
    );
    let dataset = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ];

    let loss_before: Scalar = dataset
        .iter()
        .map(|(x, y)| {
            let (out, _) = net.forward(Vector::from_data(*x));
            mse(out, Vector::from_data(*y)).0
        })
        .sum();

    let mut opt = Sgd::new(0.01);
    for _ in 0..50 {
        for (x, y) in dataset {
            train_step(
                &mut net,
                &mut opt,
                Vector::from_data(x),
                Vector::from_data(y),
                mse,
            );
        }
    }

    let loss_after: Scalar = dataset
        .iter()
        .map(|(x, y)| {
            let (out, _) = net.forward(Vector::from_data(*x));
            mse(out, Vector::from_data(*y)).0
        })
        .sum();

    assert!(
        loss_after < loss_before * 0.5,
        "la loss devrait avoir significativement baissé : {loss_before:.4} → {loss_after:.4}"
    );
}

#[test]
fn grad_check_linear_tanh_linear() {
    let net = seq!(
        Linear::<2, 4, 8>::from_seed(42),
        Tanh::<4> {},
        Linear::<4, 1, 4>::from_seed(99)
    );
    let input = Vector::from_data([1.0, 0.5]);
    let target = Vector::from_data([1.0]);

    let result = GradChecker::check::<17, _, _>(net, input, target, mse, 1e-4);

    println!();
    println!("=== Gradient Check ===");
    println!("  mean rel. error : {:.2e}", result.mean_relative_error);
    println!("  max  rel. error : {:.2e}", result.max_relative_error);
    println!("  min  rel. error : {:.2e}", result.min_relative_error);
    println!("  std  rel. error : {:.2e}", result.std_relative_error);
    println!("======================");

    assert!(
        result.max_relative_error < 1e-2,
        "gradient check échoué : max erreur relative = {:.2e}",
        result.max_relative_error
    );
}
