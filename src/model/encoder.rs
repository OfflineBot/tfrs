
use crate::{
    model::{
        attention::Attention,
        nn::{NeuralNetwork, NeuralNetworkConfig},
        norm::AddNorm
    },
    utils::{Activation, Loss, Optimizer}};


#[derive(Clone, Copy)]
pub struct EncoderConfig {
    n_heads: usize,
    eps: f32,

    // ====== neural network =======
    /// dimension/size of hidden layer
    d_ff: usize,
    activation_ff: Activation,
    loss_ff: Loss,
    optim_ff: Optimizer,
}


impl EncoderConfig {
    pub fn new(
        n_heads: usize,
        eps: f32,
        d_ff: usize,
        activation_ff: Activation,
        loss_ff: Loss,
        optim_ff: Optimizer,
    ) -> Self {
        Self {
            n_heads,
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
pub struct Encoder {

    config: EncoderConfig,

    self_attention: Attention,
    add_norm1: AddNorm,

    ff: NeuralNetwork,
    add_norm2: AddNorm,
}

impl Encoder {

    pub fn new(config: EncoderConfig, d_model: usize) -> Self {

        let nn_config = NeuralNetworkConfig::new(d_model, config.d_ff, d_model);

        Self {
            config,
            self_attention: Attention::new(d_model, config.n_heads),

            ff: NeuralNetwork::new(nn_config, config.activation_ff, config.loss_ff, config.optim_ff),

            add_norm1: AddNorm::new(d_model, config.eps),
            add_norm2: AddNorm::new(d_model, config.eps),
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

