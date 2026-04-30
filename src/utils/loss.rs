
use ndarray::Array2;

#[derive(Clone, Copy)]
pub enum Loss {
    MSE,
    Custom {
        loss: fn(&Array2<f32>) -> Array2<f32>,
        derivative: fn(&Array2<f32>) -> Array2<f32>,
    }
}

