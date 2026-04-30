use ndarray::{array};
use crate::utils::{Activation, Loss, Optimizer};


mod model;
mod train;
mod utils;

fn main() {
    println!("Hello, world!");


    let nn_config = model::nn::NeuralNetworkConfig::new(2, 10, 1, (-1., 1.));

    let mut nn = model::nn::NeuralNetwork::new(
        nn_config, 
        Activation::LeakyReLU(0.001),
        Loss::MSE,
        Optimizer::SGD(0.001)
    );

    let input = array![
        [ 1., 1. ],
        [ 0., 0. ],
        [ 1., 0. ],
        [ 0., 1. ],
    ];

    let truth = array![
        [0.],
        [0.],
        [1.],
        [1.],
    ];

    let epochs = 1_000;

    for i in 0..epochs {
        nn.forward(&input);
        if i % (epochs/10) == 0 {
            let loss = nn.item_loss(&truth);
            println!("{} | Loss: {}", i+1, loss);
        }
        nn.backward(&truth);
        nn.step();
    }
}

