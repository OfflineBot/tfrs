#![allow(dead_code)]

use crate::model::{decoder::{Decoder, DecoderBlock, DecoderConfig}, encoder::{Encoder, EncoderBlock, EncoderConfig}};

pub struct Transformer {
    encoder: EncoderBlock,
    decoder: DecoderBlock,
}

impl Transformer {
    pub fn new_empty() -> Self {
        Self { encoder: EncoderBlock::new(), decoder: DecoderBlock::new() }
    }

    pub fn new_decoder_only(decoder: DecoderBlock) -> Self {
        Self { encoder: EncoderBlock::new(), decoder }
    }

    pub fn new_encoder_only(encoder: EncoderBlock) -> Self {
        Self { encoder, decoder: DecoderBlock::new() }
    }

    pub fn new(encoder: EncoderBlock, decoder: DecoderBlock) -> Self {
        Self { encoder, decoder }
    }

    pub fn set_encoder(&mut self, encoder: EncoderBlock) { self.encoder = encoder; }
    pub fn set_decoder(&mut self, decoder: DecoderBlock) { self.decoder = decoder; }

    pub fn unset_encoder(&mut self) { self.encoder = EncoderBlock::new(); }
    pub fn unset_decoder(&mut self) { self.decoder = DecoderBlock::new(); }

    pub fn set_encoder_configs(&mut self, encoder: Vec<EncoderConfig>) {
        for e in encoder {
            self.encoder.insert(Encoder::new(e));
        }
    }

    pub fn set_decoder_configs(&mut self, decoder: Vec<DecoderConfig>) {
        for d in decoder {
            self.decoder.insert(Decoder::new(d));
        }
    }
}

