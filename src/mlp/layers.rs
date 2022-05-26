use crate::functions as F;
use crate::{init_matrix, Matrix};

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
            Matrix::new(vec![0f64; out_channel], 1, out_channel),
        );
        let grad_v = (
            Matrix::new(
                vec![0f64; in_channel * out_channel],
                in_channel,
                out_channel,
            ),
            Matrix::new(vec![0f64; out_channel], 1, out_channel),
        );
        Linear {
            weights,
            biases,
            t: 1,
            grad_m,
            grad_v,
            active: active,
            input: None,
            output: None,
        }
    }

    pub fn forward(&mut self, input: &Matrix<f64>) -> Matrix<f64> {
        // input.shape = (batch, in_channel)
        self.input = Some(input.clone());
        let mut output = (input.clone() & self.weights.clone()) + self.biases.clone();
        if self.active {
            output = F::relu(&output)
        }
        self.output = Some(output.clone());
        // output.shape = (batch, out_channel)
        output
    }

    pub fn backward(&mut self, grad: &Matrix<f64>, lr: f64) -> Matrix<f64> {
        let mut d_output = grad.clone();
        // d_output.shape: (batch, out_channel)
        if self.active {
            d_output = d_output * F::relu_prime(&self.output.clone().unwrap())
        }
        // d_weights: (in_channel, batch) & (batch, out_channel)
        let d_weights = self.input.clone().unwrap().transpose() & d_output.clone();
        // d_bias: (out_channel, 1)
        let d_bias = d_output.clone().dim_sum(0);
        let batch = grad.rows as f64;
        for i in 0..d_weights.v.len() {
            let m = (0.9 * self.grad_m.0.v[i] + 0.1 * d_weights.v[i] / batch)
                / (1.0 - 0.9f64.powi(self.t));
            let v = (0.999 * self.grad_v.0.v[i] + 0.001 * d_weights.v[i] * d_weights.v[i] / batch)
                / (1.0 - 0.999f64.powi(self.t));
            self.grad_m.0.v[i] = m;
            self.grad_v.0.v[i] = v;
            self.weights.v[i] -= lr * m / (v.sqrt() + 1e-8);
            if i / self.weights.cols == 0 {
                let m = (0.9 * self.grad_m.1.v[i] + 0.1 * d_bias.v[i] / batch)
                    / (1.0 - 0.9f64.powi(self.t));
                let v = (0.999 * self.grad_v.1.v[i] + 0.001 * d_bias.v[i] * d_bias.v[i] / batch)
                    / (1.0 - 0.999f64.powi(self.t));
                self.grad_m.1.v[i] = m;
                self.grad_v.1.v[i] = v;
                self.biases.v[i] -= lr * m / (v.sqrt() + 1e-8);
            }
            self.t += 1;
        }
        // d_input: (in_channel, out_channel) & (out_channel, num)
        d_output & self.weights.clone().transpose()
    }
}

pub struct CEOutput {
    pub input: Option<Matrix<f64>>,
    pub output: Option<Matrix<f64>>,
}

impl CEOutput {
    pub fn new() -> Self {
        CEOutput {
            input: None,
            output: None,
        }
    }

    pub fn forward(&mut self, x: &Matrix<f64>, y: &Matrix<f64>) -> (Matrix<f64>, f64) {
        self.input = Some(x.clone());
        let softmax = F::softmax(x, 1);
        self.output = Some(softmax.clone());
        (softmax, F::cross_entropy(&self.output.clone().unwrap(), y))
    }

    pub fn backward(&mut self, y: &Matrix<f64>) -> Matrix<f64> {
        F::cross_entropy_prime(&self.output.clone().unwrap(), y)
    }
}
