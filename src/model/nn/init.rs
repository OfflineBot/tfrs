use ndarray::{Array1, Array2};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use crate::model::nn::nn::LayerParams;

impl LayerParams {

    #[allow(dead_code)]
    pub fn init(input_size: usize, hidden_size: usize, output_size: usize, random_range: (f32, f32)) -> Self {
        let dist = Uniform::new(random_range.0, random_range.1).unwrap();
        Self {
            weights_1: Array2::random((input_size, hidden_size), dist),
            biases_1: Array1::random(hidden_size, dist),

            weights_2: Array2::random((hidden_size, output_size), dist),
            biases_2: Array1::random(output_size, dist),

            input: None,
            a1: None,
            z1: None,
            z2: None
        }
    }
}
