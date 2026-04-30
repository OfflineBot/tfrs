use crate::{model::nn::nn::LayerParams, utils::Optimizer};


impl LayerParams {

    #[allow(dead_code)]
    pub fn update(&mut self, optimizer: Optimizer) {
        optimizer.apply(self);
    }


    #[allow(dead_code)]
    pub fn clear_cache(&mut self) {
        self.input = None;
        self.z1 = None;
        self.a1 = None;
        self.z2 = None;

        self.weight_grad_1 = None;
        self.weight_grad_2 = None;

        self.bias_grad_1 = None;
        self.bias_grad_2 = None;
    }
}

