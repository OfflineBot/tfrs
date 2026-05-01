use ndarray::{Array1, Array2, Axis};

use crate::utils::Trainable;

#[derive(Clone)]
#[allow(dead_code)]
pub struct AddNorm {
    gamma: Array1<f32>,
    beta: Array1<f32>,
    eps: f32,


    gamma_grad: Option<Array1<f32>>,
    beta_grad: Option<Array1<f32>>,
    input: Option<Array2<f32>>,
    x_norm: Option<Array2<f32>>,
    std: Option<Array2<f32>>,
}

impl AddNorm {

    pub fn new(d_model: usize, eps: f32) -> Self {
        Self {
            gamma: Array1::ones(d_model),
            beta: Array1::zeros(d_model),
            eps,
            gamma_grad: None,
            beta_grad: None,
            input: None,
            x_norm: None,
            std: None,
        }
    }

    pub fn forward(&mut self, x: &Array2<f32>, sublayer_output: &Array2<f32>) -> Array2<f32> {
        let summed = x + sublayer_output;
        let mean = summed.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1));
        let var = summed.var_axis(Axis(1), 0.).insert_axis(Axis(1));
        let std = (var + self.eps).mapv(f32::sqrt);
        let x_norm = (x - &mean) / &std;
        let out = &x_norm * &self.gamma + &self.beta;

        self.input = Some(summed);
        self.x_norm = Some(x_norm);
        self.std = Some(std);

        out
    }
}


impl Trainable for AddNorm {
    fn update(&mut self, opt: &crate::utils::Optimizer) {
        opt.step_b(&mut self.gamma, self.gamma_grad.as_ref().unwrap());
        opt.step_b(&mut self.beta, self.beta_grad.as_ref().unwrap());
    }

    fn clear_grads(&mut self) {
        self.gamma_grad = None;
        self.beta_grad = None;
        self.input = None;
        self.x_norm = None;
        self.std = None;
    }
}

