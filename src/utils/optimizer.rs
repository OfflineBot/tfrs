
use ndarray::{Array1, Array2};

pub trait Trainable {
    fn update(&mut self, opt: &Optimizer);
    fn clear_grads(&mut self);
}


#[derive(Clone, Copy)]
pub enum Optimizer {
    /// `SGD` with *learning rate* as *f32*
    SGD(f32),
    ADAM,
}


impl Optimizer {
    pub fn step_w(&self, w: &mut Array2<f32>, g: &Array2<f32>) {
        match self {
            Self::SGD(lr) => *w -= &(g * *lr),
            Self::ADAM => {},
        }
    }

    pub fn step_b(&self, w: &mut Array1<f32>, g: &Array1<f32>) {
        match self {
            Self::SGD(lr) => *w -= &(g * *lr),
            Self::ADAM => {},
        }
    }
}

