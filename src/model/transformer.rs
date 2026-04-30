#![allow(dead_code)]

use ndarray::{Array1, Array2};

use crate::{model::{decoder::{Decoder, DecoderBlock, DecoderConfig}, encoder::{Encoder, EncoderBlock, EncoderConfig}}, utils::{TransformerOptimizer, xavier_init}};

pub struct Transformer {

    d_model: usize,

    encoder: EncoderBlock,
    decoder: DecoderBlock,

    w_out: Array2<f32>,
    b_out: Array1<f32>,
}

impl Transformer {
    // ============== PROPERTY FUNCTIONS ================
    pub fn new_empty(d_model: usize, num_classes: usize) -> Self {
        Self { 
            encoder: EncoderBlock::new(),
            decoder: DecoderBlock::new(),
            d_model,
            w_out: xavier_init(d_model, num_classes),
            b_out: Array1::zeros(num_classes)
        }
    }

    pub fn set_encoder(&mut self, encoder: EncoderBlock) { self.encoder = encoder; }
    pub fn set_decoder(&mut self, decoder: DecoderBlock) { self.decoder = decoder; }

    pub fn unset_encoder(&mut self) { self.encoder = EncoderBlock::new(); }
    pub fn unset_decoder(&mut self) { self.decoder = DecoderBlock::new(); }

    pub fn set_encoder_configs(&mut self, encoder: Vec<EncoderConfig>) {
        for e in encoder {
            self.encoder.insert(Encoder::new(e, self.d_model));
        }
    }

    pub fn set_decoder_configs(&mut self, decoder: Vec<DecoderConfig>) {
        for d in decoder {
            self.decoder.insert(Decoder::new(d, self.d_model));
        }
    }

    // ============== TRAINING =========================
    pub fn forward(src: Array2<f32>, target: Array2<f32>) -> Array2<f32> {
        Array2::zeros((0, 0))
    }

    pub fn backward(delta: Array2<f32>) {

    }

    pub fn step(optimizer: TransformerOptimizer) {
        optimizer.apply();
    }
}

