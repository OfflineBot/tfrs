use ndarray::Array2;

use crate::utils::xavier_init;


pub struct Attention {
    pub w_q: Array2<f32>,
    pub w_k: Array2<f32>,
    pub w_v: Array2<f32>,
    pub w_o: Array2<f32>,

    d_model: usize,

    // ===== cache =====
    q: Option<Array2<f32>>,
    k: Option<Array2<f32>>,
    v: Option<Array2<f32>>,

    scores: Option<Array2<f32>>,
    weights: Option<Array2<f32>>,
    input: Option<Array2<f32>>,
}

impl Attention {
    pub fn new(d_model: usize) -> Self {
        Self {
            d_model,
            w_q: xavier_init(d_model, d_model),
            w_k: xavier_init(d_model, d_model),
            w_v: xavier_init(d_model, d_model),
            w_o: xavier_init(d_model, d_model),

            q: None,
            k: None,
            v: None,

            scores: None,
            weights: None,
            input: None,
        }
    }
}

