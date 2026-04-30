
use crate::model::nn::LayerParams;

#[derive(Clone, Copy)]
pub enum Optimizer {
    /// `SGD` with *learning rate* as *f32*
    SGD(f32),
}


impl Optimizer {
    pub fn apply(
        &self, 
        layer_params: &mut LayerParams,
    ) {
        match self {
            Self::SGD(lr) => {
                let wg1 = layer_params.weight_grad_1.clone().unwrap();
                let wg2 = layer_params.weight_grad_2.clone().unwrap();

                let bg1 = layer_params.bias_grad_1.clone().unwrap();
                let bg2 = layer_params.bias_grad_2.clone().unwrap();

                layer_params.weights_1 -= &(&wg1 * lr.clone());
                layer_params.weights_2 -= &(&wg2 * lr.clone());

                layer_params.biases_1 -= &(bg1 * lr.clone());
                layer_params.biases_2 -= &(bg2 * lr.clone());
            },
        }
    }
}

