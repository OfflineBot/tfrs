use core::f32;

use ndarray::{Array2, Axis};

use crate::utils::xavier_init;


#[derive(Clone)]
pub struct Attention {
    pub w_q: Array2<f32>,
    pub w_k: Array2<f32>,
    pub w_v: Array2<f32>,
    pub w_o: Array2<f32>,

    n_heads: usize,
    d_head: usize, // d_model / n_heads

    // ===== cache =====
    q: Option<Array2<f32>>,
    k: Option<Array2<f32>>,
    v: Option<Array2<f32>>,

    scores: Option<Array2<f32>>,
    weights: Option<Array2<f32>>,
    input: Option<Array2<f32>>,
}

impl Attention {
    pub fn new(d_model: usize, n_heads: usize) -> Self {
        assert!(d_model % n_heads == 0, "d_model must be divisible by n_heads");
        Self {
            d_head: d_model / n_heads,
            n_heads,
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

    pub fn forward(&mut self, x: &Array2<f32>, mask: Option<&Array2<f32>>) -> Array2<f32> {
        let scale = (self.d_head as f32).sqrt();
        let seq = x.shape()[0];

        let q = x.dot(&self.w_q);
        let k = x.dot(&self.w_k);
        let v = x.dot(&self.w_v);

        let q = q.to_shape((seq, self.n_heads, self.d_head))
            .unwrap()
            .permuted_axes([1, 0, 2]);
        let k = k.to_shape((seq, self.n_heads, self.d_head))
            .unwrap()
            .permuted_axes([1, 0, 2]);
        let v = v.to_shape((seq, self.n_heads, self.d_head))
            .unwrap()
            .permuted_axes([1, 0, 2]);

        let mut head_outs = Vec::new();
        for h in 0..self.n_heads {
            let qh = q.index_axis(Axis(0), h).to_owned();
            let kh = k.index_axis(Axis(0), h).to_owned();
            let vh = v.index_axis(Axis(0), h).to_owned();
            let mut scores = qh.dot(&kh.t()) / scale;
            if let Some(m) = mask { scores = scores + m; }
            let weights = Self::softmax(&scores);
            head_outs.push(weights.dot(&vh));
        }

        let concat = ndarray::concatenate(
            Axis(1),
            &head_outs.iter().map(|h| h.view()).collect::<Vec<_>>()
        ).unwrap();

        concat.dot(&self.w_o)
    }


    fn softmax(x: &Array2<f32>) -> Array2<f32> {
        let max = x.map_axis(Axis(1), |row| row.fold(f32::NEG_INFINITY, |a, &b| a.max(b)))
                   .insert_axis(Axis(1));
        let exp = (x - &max).mapv(f32::exp);
        let sum = exp.sum_axis(Axis(1)).insert_axis(Axis(1));
        exp / sum
    }
}

