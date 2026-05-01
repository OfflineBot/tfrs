use core::f32;

use ndarray::{Array2, Array3, Axis};

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
    input_q: Option<Array2<f32>>,
    input_kv: Option<Array2<f32>>,
    q: Option<Array3<f32>>,
    k: Option<Array3<f32>>,
    v: Option<Array3<f32>>,
    weights: Option<Array3<f32>>,
    concat: Option<Array2<f32>>,
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

            input_q: None,
            input_kv: None,
            q: None,
            k: None,
            v: None,
            weights: None,
            concat: None,
        }
    }

    pub fn forward(&mut self, x_q: &Array2<f32>, x_kv: &Array2<f32>, mask: Option<&Array2<f32>>) -> Array2<f32> {
        let scale = (self.d_head as f32).sqrt();
        let seq_q = x_q.shape()[0];
        let seq_kv = x_kv.shape()[0];

        let q = x_q.to_shape((seq_q, self.n_heads, self.d_head))
            .unwrap()
            .permuted_axes([1, 0, 2]);
        let k = x_kv.to_shape((seq_kv, self.n_heads, self.d_head))
            .unwrap()
            .permuted_axes([1, 0, 2]);
        let v = x_kv.to_shape((seq_kv, self.n_heads, self.d_head))
            .unwrap()
            .permuted_axes([1, 0, 2]);

        let mut weights = Array3::<f32>::zeros((self.n_heads, seq_q, seq_kv));
        let mut head_outs = Array3::<f32>::zeros((self.n_heads, seq_q, self.d_head));

        for h in 0..self.n_heads {
            let qh = q.index_axis(Axis(0), h).to_owned();
            let kh = k.index_axis(Axis(0), h).to_owned();
            let vh = v.index_axis(Axis(0), h).to_owned();

            let mut s = qh.dot(&kh.t()) / scale;
            if let Some(m) = mask { s = s + m; }
            let w = Self::softmax(&s);
            let o = w.dot(&vh);

            weights.index_axis_mut(Axis(0), h).assign(&w);
            head_outs.index_axis_mut(Axis(0), h).assign(&o);
        }

        let concat = head_outs
            .permuted_axes([1, 0, 2])
            .to_shape((seq_q, self.n_heads * self.d_head)).unwrap()
            .to_owned();

        let out = concat.dot(&self.w_o);

        self.input_q = Some(x_q.clone());
        self.input_kv = Some(x_kv.clone());
        self.q = Some(q.to_owned());
        self.k = Some(k.to_owned());
        self.v = Some(v.to_owned());
        self.weights = Some(weights);
        self.concat = Some(concat);

        out
    }


    fn softmax(x: &Array2<f32>) -> Array2<f32> {
        let max = x.map_axis(Axis(1), |row| row.fold(f32::NEG_INFINITY, |a, &b| a.max(b)))
                   .insert_axis(Axis(1));
        let exp = (x - &max).mapv(f32::exp);
        let sum = exp.sum_axis(Axis(1)).insert_axis(Axis(1));
        exp / sum
    }
}

