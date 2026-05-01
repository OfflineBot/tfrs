#![allow(dead_code)]

use ndarray::{Array1, Array2};

use crate::{
    model::{
        decoder::{Decoder, DecoderBlock, DecoderConfig}, embedding::{Embeddings, positional_encoding}, encoder::{Encoder, EncoderBlock, EncoderConfig}
    }, 
    utils::{AdamState1, AdamState2, Trainable, xavier_init}
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

    w_out_state: AdamState2,
    b_out_state: AdamState1,

    // ===== forward cache (for backward) =====
    h_cache:        Option<Array2<f32>>,
    used_encoder:   bool,
    used_decoder:   bool,
    decoder_had_memory: bool,
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
            w_out_state: AdamState2::default(),
            b_out_state: AdamState1::default(),
            h_cache: None,
            used_encoder: false,
            used_decoder: false,
            decoder_had_memory: false,
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

        let (h, used_encoder, used_decoder, decoder_had_memory) = match (memory, tgt_ids) {
            (Some(m), None)  => (m, true, false, false),
            (mem, Some(ids)) => {
                let had_mem = mem.is_some();
                let t       = self.tgt_embed.forward(ids) + positional_encoding(ids.len(), self.d_model);
                let mask    = causal_mask(ids.len());
                let out     = self.decoder.forward(&t, mem.as_ref(), Some(&mask));
                (out, had_mem, true, had_mem)
            },
            (None, None) => panic!("need atleast src or target. Got nothing")
        };

        let logits = h.dot(&self.w_out) + &self.b_out;

        self.h_cache            = Some(h);
        self.used_encoder       = used_encoder;
        self.used_decoder       = used_decoder;
        self.decoder_had_memory = decoder_had_memory;

        logits
    }

    pub fn backward(&mut self, delta: Array2<f32>) {
        // logits = h @ w_out + b_out
        let h = self.h_cache.as_ref().expect("forward must run before backward").clone();

        self.w_out_grad = Some(h.t().dot(&delta));
        self.b_out_grad = Some(delta.sum_axis(ndarray::Axis(0)));
        let d_h = delta.dot(&self.w_out.t());

        // dispatch mirrors forward's match
        let d_memory: Option<Array2<f32>> = if self.used_decoder {
            let (d_t, d_mem) = self.decoder.backward(d_h, self.decoder_had_memory);
            // PE has no params -> delta passes through unchanged
            self.tgt_embed.backward(&d_t);
            d_mem
        } else {
            // encoder-only path: d_h *is* d_memory
            Some(d_h)
        };

        if self.used_encoder {
            let d_mem = d_memory.expect("encoder ran but no memory delta");
            let d_s   = self.encoder.backward(d_mem);
            self.src_embed.backward(&d_s);
        }
    }
}


impl Trainable for Transformer {
    fn update(&mut self, opt: &crate::utils::Optimizer) {
        self.encoder.update(opt);
        self.decoder.update(opt);

        self.src_embed.update(opt);
        self.tgt_embed.update(opt);

        opt.step_w(&mut self.w_out, self.w_out_grad.as_ref().unwrap(), &mut self.w_out_state);
        opt.step_b(&mut self.b_out, self.b_out_grad.as_ref().unwrap(), &mut self.b_out_state);
    }

    fn clear_grads(&mut self) {
        self.encoder.clear_grads();
        self.decoder.clear_grads();

        self.src_embed.clear_grads();
        self.tgt_embed.clear_grads();

        self.w_out_grad = None;
        self.b_out_grad = None;

        self.h_cache            = None;
        self.used_encoder       = false;
        self.used_decoder       = false;
        self.decoder_had_memory = false;
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
