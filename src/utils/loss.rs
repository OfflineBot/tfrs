
use ndarray::Array2;

#[derive(Clone, Copy)]
pub enum Loss {
    MSE,
    #[allow(unused)]
    Custom {
        loss: fn(&Array2<f32>, &Array2<f32>) -> f32,
        derivative: fn(&Array2<f32>, &Array2<f32>) -> Array2<f32>,
    }
}


impl Loss {
    pub fn loss_item(&self, truth: &Array2<f32>, output: &Array2<f32>) -> f32 {
        match self {
            Self::MSE => mse_item(truth, output),
            Self::Custom { loss, .. } => loss(truth, output),
        }
    }

    pub fn deriv_loss(&self, truth: &Array2<f32>, output: &Array2<f32>) -> Array2<f32> {
        match self {
            Self::MSE => deriv_mse_item(truth, output),
            Self::Custom { derivative, .. } => derivative(truth, output),
        }
    }
}


fn mse_item(truth: &Array2<f32>, output: &Array2<f32>) -> f32 { 
    (truth - output).powf(2.).mean().unwrap()
}
fn deriv_mse_item(truth: &Array2<f32>, output: &Array2<f32>) -> Array2<f32> { output - truth }

