#![allow(dead_code)]

use ndarray::{Array1, Array2};
use crate::utils::Activation;



/// per definition *1 hidden layer* with *ReLU* activation
pub struct NeuralNetworkConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
}

impl NeuralNetworkConfig {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        Self {
            input_size,
            hidden_size,
            output_size,
        }
    }
}


pub struct LayerParams {

    pub weights_1: Array2<f32>,
    pub biases_1: Array1<f32>,

    pub weights_2: Array2<f32>,
    pub biases_2: Array1<f32>,

    // ==== CACHE ====
    pub input: Option<Array2<f32>>,
    pub z1: Option<Array2<f32>>,
    pub a1: Option<Array2<f32>>,
    pub z2: Option<Array2<f32>>,
}


/// per definition *1 hidden layer* with *ReLU* activation
pub struct NeuralNetwork {
    pub config: NeuralNetworkConfig,
    pub activation: Activation,
    pub layer: LayerParams,
}

