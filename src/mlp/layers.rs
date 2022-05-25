use crate::{init_matrix, Matrix};
use crate::functions as F;

pub struct Linear {
    pub weights: Matrix<f64>,
    pub biases: Matrix<f64>,
    active: bool,
    pub input: Option<Matrix<f64>>,
    pub output: Option<Matrix<f64>>,
}

impl Linear {
    pub fn new(in_channel: usize, out_channel: usize, active: bool) -> Self {
        let weights = init_matrix(in_channel, out_channel);
        let biases = init_matrix(1, out_channel);
        Linear {
            weights,
            biases,
            active: active,
            input: None,
            output: None,
        }
    }

    pub fn forward(&mut self, input: &Matrix<f64>) -> Matrix<f64> {
        // input.shape = (batch, in_channel)
        self.input = Some(input.clone());
        let mut output = (input.clone() & self.weights.clone()) + self.biases.clone();
        if self.active { output = F::relu(&output) }
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
            self.weights.v[i] -= lr * d_weights.v[i] / batch;
            if i / self.weights.cols == 0 {
                self.biases.v[i] -= lr * d_bias.v[i] / batch;
            }
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
