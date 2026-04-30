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

    let out = nn.forward(&input);
    nn.backward(&truth);
    nn.step();
}

