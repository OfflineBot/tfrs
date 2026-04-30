use crate::{
    model::{
        encoder::EncoderConfig, transformer::Transformer
    },
    utils::{
        Activation, Loss, Optimizer
    }
};

mod model;
mod train;
mod utils;

fn main() {

    let mut transformer = Transformer::new_empty();

    let config = EncoderConfig::new(
        512, 
        (-1., 1.), 
        10, 
        Activation::ReLU, 
        Loss::MSE, 
        Optimizer::SGD(0.01)
    );

    let config_array = vec![config, config, config, config, config, config];

    transformer.set_encoder_configs(config_array);
}

