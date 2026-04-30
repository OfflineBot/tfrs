use ndarray::{Array1, Array2};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

use crate::{model::nn::{NeuralNetwork, NeuralNetworkConfig}, utils::{Activation, Loss, Optimizer}};


#[derive(Clone, Copy)]
pub struct EncoderConfig {
    /// dimension of tokens (array length/size for each token)
    d_model: usize,

    random_inits: (f32, f32),

    // ====== neural network =======
    /// dimension/size of hidden layer
    d_ff: usize,
    activation_ff: Activation,
    loss_ff: Loss,
    optim_ff: Optimizer,
}


impl EncoderConfig {
    pub fn new(
        d_model: usize,
        random_inits: (f32, f32),
        d_ff: usize,
        activation_ff: Activation,
        loss_ff: Loss,
        optim_ff: Optimizer,
    ) -> Self {
        Self {
            d_model, 
            random_inits, 
            d_ff, 
            activation_ff,
            loss_ff, 
            optim_ff
        }
    }
}


#[derive(Clone)]
pub struct Encoder {

    config: EncoderConfig,

    w_q: Array2<f32>,
    w_k: Array2<f32>,
    w_v: Array2<f32>,

    w_o: Option<Array2<f32>>,

    ff: NeuralNetwork,

    norm1_gamma: Array1<f32>,
    norm1_beta: Array1<f32>,
    norm2_gamma: Array1<f32>,
    norm2_beta: Array1<f32>,
}

impl Encoder {

    pub fn new(config: EncoderConfig) -> Self {

        let dist = Uniform::new(config.random_inits.0, config.random_inits.1).unwrap();

        let random_weight = || -> Array2<f32> { 
            Array2::random((config.d_model, config.d_model.clone()), dist) 
        };

        let d_model_ones = Array1::ones(config.d_model);
        let d_model_zeros = Array1::zeros(config.d_model);

        let nn_config = NeuralNetworkConfig::new(config.d_model, config.d_ff, config.d_model, config.random_inits);

        Self {
            config,
            w_q: random_weight(),
            w_k: random_weight(),
            w_v: random_weight(),
            w_o: None,
            ff: NeuralNetwork::new(nn_config, config.activation_ff, config.loss_ff, config.optim_ff),

            norm1_gamma: d_model_ones.clone(),
            norm1_beta: d_model_zeros.clone(),

            norm2_gamma: d_model_ones,
            norm2_beta: d_model_zeros,
        }
    }
}


pub struct EncoderBlock {
    layers: Vec<Encoder>,
}

impl EncoderBlock {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn insert(&mut self, encoder: Encoder) {
        self.layers.push(encoder);
    }
}

