use crate::autodiff::flat_grads::FlatGrads;
use crate::autodiff::module::Module;
use crate::autodiff::perturb::Perturb;
use crate::scalar::{fabs, sqrt, Scalar};

pub struct GradChecker {
    pub mean_relative_error: Scalar,
    pub max_relative_error: Scalar,
    pub min_relative_error: Scalar,
    pub std_relative_error: Scalar,
}

impl GradChecker {
    pub fn check<const N: usize, Net, Input>(
        net: Net,
        input: Input,
        target: <Net as Module<Input>>::Output,
        loss_fn: fn(
            <Net as Module<Input>>::Output,
            <Net as Module<Input>>::Output,
        ) -> (Scalar, <Net as Module<Input>>::Output),
        eps: Scalar,
    ) -> Self
    where
        Net: Module<Input> + Perturb + FlatGrads + Clone,
        <Net as Module<Input>>::Output: Copy,
        Input: Copy,
    {
        debug_assert_eq!(net.num_params(), N, "N doit correspondre à net.num_params()");

        // Gradients analytiques
        let (output, ctx) = net.forward(input);
        let (_, loss_grad) = loss_fn(output, target);
        let (_, grads) = net.backward(loss_grad, &ctx);
        let mut buf_ana: [Scalar; N] = [0.0; N];
        let mut offset = 0;
        Net::write_grads(&grads, &mut buf_ana, &mut offset);

        // Gradients numériques (différences finies centrées)
        let mut buf_num: [Scalar; N] = [0.0; N];
        for i in 0..N {
            let mut net_plus = net.clone();
            net_plus.perturb(i, eps);
            let (out_plus, _) = net_plus.forward(input);
            let (loss_plus, _) = loss_fn(out_plus, target);

            let mut net_minus = net.clone();
            net_minus.perturb(i, -eps);
            let (out_minus, _) = net_minus.forward(input);
            let (loss_minus, _) = loss_fn(out_minus, target);

            buf_num[i] = (loss_plus - loss_minus) / (2.0 * eps);
        }

        // Erreur relative : |num - ana| / max(|num|, |ana|, ε)
        let mut errors: [Scalar; N] = [0.0; N];
        for i in 0..N {
            let abs_num = fabs(buf_num[i]);
            let abs_ana = fabs(buf_ana[i]);
            let denom = if abs_num > abs_ana { abs_num } else { abs_ana };
            let denom = if denom > 1e-8 { denom } else { 1e-8 };
            errors[i] = fabs(buf_num[i] - buf_ana[i]) / denom;
        }

        let mut sum: Scalar = 0.0;
        let mut max: Scalar = 0.0;
        let mut min: Scalar = Scalar::MAX;
        for i in 0..N {
            let e = errors[i];
            sum += e;
            if e > max { max = e; }
            if e < min { min = e; }
        }
        let mean = sum / N as Scalar;

        let mut var_sum: Scalar = 0.0;
        for i in 0..N {
            let d = errors[i] - mean;
            var_sum += d * d;
        }

        GradChecker {
            mean_relative_error: mean,
            max_relative_error: max,
            min_relative_error: min,
            std_relative_error: sqrt(var_sum / N as Scalar),
        }
    }
}
