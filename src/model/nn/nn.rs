#![allow(dead_code)]

use ndarray::{Array1, Array2};
use crate::utils::{Activation, Loss, Optimizer};


/// per definition *1 hidden layer* with *ReLU* activation
pub struct NeuralNetworkConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub init_weights_range: (f32, f32),
}

impl NeuralNetworkConfig {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, init_weights_range: (f32, f32)) -> Self {
        Self {
            input_size,
            hidden_size,
            output_size,
            init_weights_range,
        }
    }
}


pub struct LayerParams {

    pub weights_1: Array2<f32>,
    pub biases_1: Array1<f32>,

    pub weights_2: Array2<f32>,
    pub biases_2: Array1<f32>,

    // ==== CACHE ====
    // ---- Foward ---
    pub(super) input: Option<Array2<f32>>,
    pub(super) z1: Option<Array2<f32>>,
    pub(super) a1: Option<Array2<f32>>,
    pub(super) z2: Option<Array2<f32>>,

    // --- Backward ---
    pub weight_grad_1: Option<Array2<f32>>,
    pub weight_grad_2: Option<Array2<f32>>,

    pub bias_grad_1: Option<Array1<f32>>,
    pub bias_grad_2: Option<Array1<f32>>,
}


/// per definition *1 hidden layer* with *ReLU* activation
pub struct NeuralNetwork {
    pub config: NeuralNetworkConfig,
    pub activation: Activation,
    pub loss: Loss,
    pub optim: Optimizer,
    pub layer: LayerParams,
}


impl NeuralNetwork {
    pub fn new(config: NeuralNetworkConfig, activation: Activation, loss: Loss, optimizer: Optimizer) -> Self {

        let layer = LayerParams::init(config.input_size, config.hidden_size, config.output_size, config.init_weights_range);

        Self {
            config,
            activation,
            loss,
            layer,
            optim: optimizer,
        }
    }

    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.layer.forward(input, self.activation)
    }

    pub fn backward(&mut self, truth: &Array2<f32>) {
        let Some(output) = self.layer.z2.clone() else {
            panic!("there is not z2 output in the fully connected");
        };
        let error = self.loss.deriv_loss(truth, &output);
        self.layer.backward(error, self.activation);
    }

    pub fn step(&mut self) {
        self.layer.update(self.optim);
        self.layer.clear_cache();

    }

}

