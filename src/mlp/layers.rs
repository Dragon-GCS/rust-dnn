use crate::functions as F;
use crate::{init_matrix, Matrix};
use rayon::prelude::*;

pub struct Linear {
    pub weights: Matrix<f64>,
    pub biases: Matrix<f64>,
    t: i32,
    grad_m: (Matrix<f64>, Matrix<f64>),
    grad_v: (Matrix<f64>, Matrix<f64>),
    active: bool,
    pub input: Option<Matrix<f64>>,
    pub output: Option<Matrix<f64>>,
}

impl Linear {
    pub fn new(in_channel: usize, out_channel: usize, active: bool) -> Self {
        let weights = init_matrix(in_channel, out_channel);
        let biases = Matrix::new(vec![0f64; out_channel], 1, out_channel);
        let grad_m = (
            Matrix::new(
                vec![0f64; in_channel * out_channel],
                in_channel,
                out_channel,
            ),
            biases.clone(),
        );
        let grad_v = grad_m.clone();
        Linear {
            weights,
            biases,
            t: 1,
            grad_m,
            grad_v,
            active,
            input: None,
            output: None,
        }
    }

    pub fn forward(&mut self, input: &Matrix<f64>) -> &Matrix<f64> {
        // input.shape = (batch, in_channel)
        let mut output = &(input & &self.weights) + &self.biases;
        if self.active {
            output = F::relu(&output)
        }
        self.input = Some(input.clone());
        self.output = Some(output);
        // output.shape = (batch, out_channel)
        self.output.as_ref().unwrap()
    }

    pub fn backward(&mut self, grad: &Matrix<f64>, lr: f64) -> Matrix<f64> {
        // d_output.shape: (batch, out_channel)
        let d_output = if self.active {
            &(grad * &F::relu_prime(self.output.as_ref().unwrap()))
        } else {
            grad
        };
        // d_weights: (in_channel, batch) & (batch, out_channel)
        let d_weights = &self.input.as_ref().unwrap().transpose() & d_output;
        // d_bias: (out_channel, 1)
        let d_bias = d_output.dim_sum(0);
        let batch = grad.rows as f64;

        d_bias
            .v
            .par_iter()
            .zip(self.biases.v.par_iter_mut())
            .zip(self.grad_m.1.v.par_iter_mut())
            .zip(self.grad_v.1.v.par_iter_mut())
            .enumerate()
            .for_each(|(t, (((&d_bias, bias), grad_m), grad_v))| {
                let t = self.t + t as i32;
                let d_bias_batch = d_bias / batch;
                *grad_m = (0.9 * *grad_m + 0.1 * d_bias_batch) / (1.0 - 0.9f64.powi(t));
                *grad_v =
                    (0.999 * *grad_v + 0.001 * d_bias * d_bias_batch) / (1.0 - 0.999f64.powi(t));
                *bias -= lr * *grad_m / (grad_v.sqrt() + 1e-8);
            });

        d_weights
            .v
            .par_iter()
            .zip(self.weights.v.par_iter_mut())
            .zip(self.grad_m.0.v.par_iter_mut())
            .zip(self.grad_v.0.v.par_iter_mut())
            .enumerate()
            .for_each(|(t, (((&d_weight, weight), grad_m), grad_v))| {
                let t = self.t + t as i32;
                let d_weight_batch = d_weight / batch;
                *grad_m = (0.9 * *grad_m + 0.1 * d_weight_batch) / (1.0 - 0.9f64.powi(t));
                *grad_v = (0.999 * *grad_v + 0.001 * d_weight * d_weight_batch)
                    / (1.0 - 0.999f64.powi(t));
                *weight -= lr * *grad_m / (grad_v.sqrt() + 1e-8);
            });
        self.t += d_weights.v.len() as i32;
        // d_input: (in_channel, out_channel) & (out_channel, num)
        d_output & &self.weights.transpose()
    }
}

#[derive(Default)]
pub struct CEOutput {
    pub input: Option<Matrix<f64>>,
    pub output: Option<Matrix<f64>>,
}

impl CEOutput {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn forward(&mut self, x: &Matrix<f64>, y: &Matrix<f64>) -> (&Matrix<f64>, f64) {
        self.input = Some(x.clone());
        let softmax = F::softmax(x, 1);
        self.output = Some(softmax);
        (
            self.output.as_ref().unwrap(),
            F::cross_entropy(self.output.as_ref().unwrap(), y),
        )
    }

    pub fn backward(&mut self, y: &Matrix<f64>) -> Matrix<f64> {
        F::cross_entropy_prime(self.output.as_ref().unwrap(), y)
    }
}
