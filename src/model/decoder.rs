
use ndarray::Array2;

use crate::{model::{attention::Attention, nn::{NeuralNetwork, NeuralNetworkConfig}, norm::AddNorm}, utils::{Activation, Loss, Optimizer}};


#[derive(Clone, Copy)]
pub struct DecoderConfig {
    n_heads: usize,
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
pub struct Decoder {

    config: DecoderConfig,

    cross_attention: Attention,
    add_norm1: AddNorm,

    self_attention: Attention,
    add_norm2: AddNorm,

    ff: NeuralNetwork,
    add_norm3: AddNorm,

}

impl Decoder {

    pub fn new(config: DecoderConfig, d_model: usize) -> Self {

        let nn_config = NeuralNetworkConfig::new(d_model, config.d_ff, d_model);

        Self {
            config,

            self_attention: Attention::new(d_model, config.n_heads),
            cross_attention: Attention::new(d_model, config.n_heads),

            ff: NeuralNetwork::new(nn_config, config.activation_ff, config.loss_ff, config.optim_ff),

            add_norm1: AddNorm::new(d_model, config.eps),
            add_norm2: AddNorm::new(d_model, config.eps),
            add_norm3: AddNorm::new(d_model, config.eps),
        }
    }

    pub fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> {
        let mut input = self.cross_attention.forward(x, None);
        input = self.add_norm1.forward(&input);

        input = self.self_attention.forward(&input, None);
        input = self.add_norm2.forward(&input);

        input = self.ff.forward(&input);
        input = self.add_norm3.forward(&input);

        input
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

    pub fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> {
        let mut input = x.clone();

        for d in self.layers.iter_mut() {
            input = d.forward(&input);
        }

        input
    }
}

