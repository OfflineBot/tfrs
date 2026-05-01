#![allow(dead_code)]

use ndarray::{Array1, Array2};

use crate::{
    model::{
        decoder::{Decoder, DecoderBlock, DecoderConfig}, embedding::{Embeddings, positional_encoding}, encoder::{Encoder, EncoderBlock, EncoderConfig}
    }, 
    utils::{Trainable, xavier_init}
};

pub struct Transformer {
    d_model: usize,

    src_embed: Embeddings,
    tgt_embed: Embeddings,

    encoder: EncoderBlock,
    decoder: DecoderBlock,

    w_out: Array2<f32>,
    b_out: Array1<f32>,

    w_out_grad: Option<Array2<f32>>,
    b_out_grad: Option<Array1<f32>>,
}

impl Transformer {
    // ============== PROPERTY FUNCTIONS ================
    pub fn new_empty(d_model: usize, src_vocab: usize, tgt_vocab: usize, num_classes: usize) -> Self {
        Self { 
            encoder: EncoderBlock::new(),
            decoder: DecoderBlock::new(),
            src_embed: Embeddings::new(src_vocab, d_model),
            tgt_embed: Embeddings::new(tgt_vocab, d_model),
            d_model,
            w_out: xavier_init(d_model, num_classes),
            b_out: Array1::zeros(num_classes),
            w_out_grad: None,
            b_out_grad: None,
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
    pub fn forward(&mut self, src_ids: Option<&[usize]>, tgt_ids: Option<&[usize]>) -> Array2<f32> {


        let memory = src_ids.map(|ids| {
            let s = self.src_embed.forward(ids) + positional_encoding(ids.len(), self.d_model);
            self.encoder.forward(&s)
        });

        let h = match (memory, tgt_ids) {
            (Some(m), None) => m,
            (mem, Some(ids)) => {
                let t = self.tgt_embed.forward(ids) + positional_encoding(ids.len(), self.d_model);
                let mask = causal_mask(ids.len());
                self.decoder.forward(&t, mem.as_ref(), Some(&mask))
            },
            (None, None) => panic!("need atleast src or target. Got nothing")
        };

        h.dot(&self.w_out) + &self.b_out
    }

    pub fn backward(&mut self, _delta: Array2<f32>) {

    }
}


impl Trainable for Transformer {
    fn update(&mut self, opt: &crate::utils::Optimizer) {
        self.encoder.update(opt);
        self.decoder.update(opt);
        opt.step_w(&mut self.w_out, self.w_out_grad.as_ref().unwrap());
        opt.step_b(&mut self.b_out, self.b_out_grad.as_ref().unwrap());
    }

    fn clear_grads(&mut self) {
        self.encoder.clear_grads();
        self.decoder.clear_grads();
        self.w_out_grad = None;
        self.b_out_grad = None;
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
