use crate::{
    model::{
        encoder::EncoderConfig, transformer::Transformer
    },
    utils::{
        Activation, Loss, Optimizer, Trainable
    }
};

mod model;
mod train;
mod utils;

fn main() {

    let mut transformer = Transformer::new_empty(512, 30_000, 30_000, 10);

    let config = EncoderConfig::new(
        1,
        0.001,
        10,
        Activation::ReLU,
        Loss::MSE,
        Optimizer::SGD(0.01)
    );

    let config_array = vec![config, config, config, config, config, config];

    transformer.set_encoder_configs(config_array);
}

