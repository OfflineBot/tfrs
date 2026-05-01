#![allow(dead_code)]

use ndarray::{Array1, Array2};

use crate::{
    model::{
        decoder::{Decoder, DecoderBlock, DecoderConfig}, embedding::Embeddings, encoder::{Encoder, EncoderBlock, EncoderConfig}
    }, 
    utils::{TransformerOptimizer, xavier_init}
};

pub struct Transformer {
    d_model: usize,

    embeddings: Embeddings,

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
            embeddings: Embeddings,
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
    pub fn forward(&mut self, src: &[usize], tgt: Option<&[usize]>) -> Array2<f32> {

        let src = self.embeddings.forward();
        let memory = self.encoder.forward(&src);

        let memory = self.encoder.forward(&src);
        let mask = causal_mask(tgt.shape()[0]);
        let decoder_output = self.decoder.forward(&tgt, &memory, Some(&mask));
        decoder_output.dot(&self.w_out) + &self.b_out
    }

    pub fn backward(&mut self, _delta: Array2<f32>) {

    }

    pub fn step(&mut self, optimizer: TransformerOptimizer) {
        optimizer.apply();
    }
}

  pub fn causal_mask(seq: usize) -> Array2<f32> {
      let mut m = Array2::<f32>::zeros((seq, seq));
      for i in 0..seq {
          for j in (i + 1)..seq {
              m[[i, j]] = f32::NEG_INFINITY;
          }
      }
      m
  }
