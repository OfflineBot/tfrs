
use ndarray::Array2;
use crate::{model::nn::nn::LayerParams, utils::Activation};


impl LayerParams {

    #[allow(dead_code)]
    pub fn backward(&mut self, truth: &Array2<f32>, activation: Activation) {

    }
}

