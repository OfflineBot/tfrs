
use ndarray::Array2;

use crate::{model::{nn::{NeuralNetwork, NeuralNetworkConfig}, norm::AddNorm}, utils::{Activation, Loss, Optimizer, xavier_init}};


#[derive(Clone, Copy)]
pub struct DecoderConfig {
    /// dimension of tokens (array length/size for each token)
    d_model: usize,

    random_inits: (f32, f32),

    eps: f32,

    // ====== neural network =======
    /// dimension/size of hidden layer
    d_ff: usize,
    activation_ff: Activation,
    loss_ff: Loss,
    optim_ff: Optimizer,
}


impl DecoderConfig {
    #[allow(unused)]
    pub fn new(
        d_model: usize,
        random_inits: (f32, f32),
        eps: f32,
        d_ff: usize,
        activation_ff: Activation,
        loss_ff: Loss,
        optim_ff: Optimizer,
    ) -> Self {
        Self {
            d_model, 
            random_inits, 
            eps,
            d_ff, 
            activation_ff,
            loss_ff, 
            optim_ff
        }
    }
}


#[derive(Clone)]
#[allow(dead_code)]
pub struct Decoder {

    config: DecoderConfig,

    self_w_q: Array2<f32>,
    self_w_k: Array2<f32>,
    self_w_v: Array2<f32>,
    self_w_o: Option<Array2<f32>>,

    cross_w_q: Array2<f32>,
    cross_w_k: Array2<f32>,
    cross_w_v: Array2<f32>,
    cross_w_o: Option<Array2<f32>>,

    ff: NeuralNetwork,

    add_norm1: AddNorm,
    add_norm2: AddNorm,
    add_norm3: AddNorm,
}

impl Decoder {

    pub fn new(config: DecoderConfig) -> Self {

        let nn_config = NeuralNetworkConfig::new(config.d_model, config.d_ff, config.d_model, config.random_inits);

        Self {
            config,

            self_w_q: xavier_init(config.d_model, config.d_model),
            self_w_k: xavier_init(config.d_model, config.d_model),
            self_w_v: xavier_init(config.d_model, config.d_model),
            self_w_o: None,

            cross_w_q: xavier_init(config.d_model, config.d_model),
            cross_w_k: xavier_init(config.d_model, config.d_model),
            cross_w_v: xavier_init(config.d_model, config.d_model),
            cross_w_o: None,

            ff: NeuralNetwork::new(nn_config, config.activation_ff, config.loss_ff, config.optim_ff),

            add_norm1: AddNorm::new(config.d_model, config.eps),
            add_norm2: AddNorm::new(config.d_model, config.eps),
            add_norm3: AddNorm::new(config.d_model, config.eps),
        }
    }
}


pub struct DecoderBlock {
    layers: Vec<Decoder>,
}

impl DecoderBlock {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn insert(&mut self, encoder: Decoder) {
        self.layers.push(encoder);
    }
}

