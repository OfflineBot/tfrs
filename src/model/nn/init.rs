use ndarray::Array1;
use crate::{model::nn::nn::LayerParams, utils::xavier_init};

impl LayerParams {

    #[allow(dead_code)]
    pub fn init(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        Self {
            weights_1: xavier_init(input_size, hidden_size),
            biases_1: Array1::zeros(hidden_size),

            weights_2: xavier_init(hidden_size, output_size),
            biases_2: Array1::zeros(output_size),

            input: None,
            a1: None,
            z1: None,
            z2: None,

            weight_grad_1: None,
            weight_grad_2: None,

            bias_grad_1: None,
            bias_grad_2: None,
        }
    }
}
