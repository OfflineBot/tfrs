//! Multi-label email classification on the `dataset_*.ron` files.
//!
//! Architecture: encoder-only transformer (no decoder layers), mean-pool the
//! per-token logits over the sequence to one vector of `num_classes` logits,
//! then sigmoid + per-class binary cross-entropy.

use std::path::PathBuf;

use ndarray::{Array2, Axis};

use crate::{
    model::{
        encoder::EncoderConfig,
        tokenizer::{HfTokenizer, Tokenizer},
        transformer::{Transformer, TransformerHeader},
    },
    train::mail::{MailDataset, Sample, build_dataset, find_dataset_files},
    utils::{Activation, Loss, Optimizer, Trainable},
};

/// What the trainer should do.
#[derive(Clone, Copy, Debug)]
pub enum MailMode {
    /// Initialise weights from scratch, train, save, then evaluate.
    Fresh,
    /// Load weights from `save_path`, continue training, save again, evaluate.
    Continue,
    /// Load weights from `save_path` and only evaluate (no training, no save).
    EvalOnly,
}

/// How many gradient updates to do.
/// Use `Steps` if you think in update-count, `Epochs` if you think in passes.
#[derive(Clone, Copy, Debug)]
pub enum TrainBudget {
    Steps(usize),
    Epochs(usize),
}

/// Stopping criteria for training. Any combination may be set; training
/// stops as soon as **any** active criterion fires (logical OR).
///
/// Eval-based criteria (`target_loss`, `target_label_acc`, `target_exact_acc`)
/// are checked at every epoch boundary and additionally every
/// `eval_every_steps` steps if that field is set.
#[derive(Clone, Copy, Debug, Default)]
pub struct StopCriteria {
    pub max_steps: Option<usize>,
    pub max_epochs: Option<usize>,
    pub target_loss: Option<f32>,
    pub target_label_acc: Option<f32>,
    pub target_exact_acc: Option<f32>,
    pub eval_every_steps: Option<usize>,
}

impl StopCriteria {
    pub fn from_budget(b: TrainBudget) -> Self {
        let mut s = Self::default();
        match b {
            TrainBudget::Steps(n)  => s.max_steps  = Some(n),
            TrainBudget::Epochs(n) => s.max_epochs = Some(n),
        }
        s
    }
    pub fn any_set(&self) -> bool {
        self.max_steps.is_some()
            || self.max_epochs.is_some()
            || self.target_loss.is_some()
            || self.target_label_acc.is_some()
            || self.target_exact_acc.is_some()
    }
}

pub struct MailTrainConfig {
    pub mode: MailMode,
    pub stop: StopCriteria,

    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub n_enc_layers: usize,
    pub seq_len: usize,
    pub lr: f32,
    pub seed: u64,

    pub save_path: PathBuf,
    /// If empty, scans `~/Downloads` for `dataset*.ron`.
    pub files: Vec<PathBuf>,
    pub log_every: usize,
}

impl MailTrainConfig {
    pub fn default_for_dataset() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        Self {
            mode: MailMode::Fresh,
            stop: StopCriteria { max_epochs: Some(3), ..Default::default() },
            d_model: 64,
            n_heads: 4,
            d_ff: 128,
            n_enc_layers: 2,
            seq_len: 96,
            lr: 1e-3,
            seed: 0xC0FFEE,
            save_path: PathBuf::from(format!("{home}/Coding/rust/transformer/mail_model.bin")),
            files: Vec::new(),
            log_every: 25,
        }
    }
}

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn shuffle_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);
    for i in (1..n).rev() {
        let j = (xorshift64(&mut state) as usize) % (i + 1);
        idx.swap(i, j);
    }
    idx
}

fn build_model(cfg: &MailTrainConfig, vocab: usize, num_classes: usize, opt: Optimizer) -> Transformer {
    // tgt_vocab = 2 — placeholder, decoder-side embedding never runs in encoder-only mode.
    let mut model = Transformer::new_empty(cfg.d_model, vocab, 2, num_classes);
    let enc_cfg = EncoderConfig::new(
        cfg.n_heads, 1e-5, cfg.d_ff, Activation::ReLU, Loss::MSE, opt,
    );
    model.set_encoder_configs((0..cfg.n_enc_layers).map(|_| enc_cfg).collect());
    model
}

fn forward_predict(model: &mut Transformer, ids: &[usize]) -> Vec<f32> {
    let logits = model.forward(Some(ids), None);
    let mean = logits.mean_axis(Axis(0)).unwrap();
    mean.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
}

fn train_step(
    model: &mut Transformer,
    opt: &Optimizer,
    s: &Sample,
    num_classes: usize,
) -> f32 {
    model.clear_grads();
    let logits = model.forward(Some(&s.ids), None);
    let seq    = s.ids.len();

    let mean = logits.mean_axis(Axis(0)).unwrap();
    let mut probs = vec![0f32; num_classes];
    let mut loss  = 0f32;
    for c in 0..num_classes {
        let p = (1.0 / (1.0 + (-mean[c]).exp())).clamp(1e-7, 1.0 - 1e-7);
        probs[c] = p;
        let y = s.labels[c];
        loss += -(y * p.ln() + (1.0 - y) * (1.0 - p).ln());
    }

    let mut delta = Array2::<f32>::zeros((seq, num_classes));
    for r in 0..seq {
        for c in 0..num_classes {
            delta[[r, c]] = (probs[c] - s.labels[c]) / seq as f32;
        }
    }
    model.backward(delta);
    model.update(opt);
    loss
}

fn evaluate(model: &mut Transformer, samples: &[Sample], num_classes: usize) -> (f32, f32, f32) {
    if samples.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut total_loss = 0f32;
    let mut correct_labels = 0usize;
    let mut total_labels = 0usize;
    let mut exact_matches = 0usize;

    for s in samples {
        let probs = forward_predict(model, &s.ids);
        let mut all_match = true;
        for c in 0..num_classes {
            let p = probs[c].clamp(1e-7, 1.0 - 1e-7);
            let y = s.labels[c];
            total_loss += -(y * p.ln() + (1.0 - y) * (1.0 - p).ln());
            let pred = if probs[c] > 0.5 { 1.0 } else { 0.0 };
            if pred == y { correct_labels += 1; } else { all_match = false; }
            total_labels += 1;
        }
        if all_match { exact_matches += 1; }
    }
    let avg_loss  = total_loss / samples.len() as f32;
    let label_acc = correct_labels as f32 / total_labels as f32;
    let exact_acc = exact_matches as f32 / samples.len() as f32;
    (avg_loss, label_acc, exact_acc)
}

pub fn train_and_eval_mail(cfg: MailTrainConfig) {
    println!("=== mail classification ===");
    println!("mode: {:?}   stop: {:?}", cfg.mode, cfg.stop);
    println!("checkpoint: {}", cfg.save_path.display());

    let tok = HfTokenizer::default_pretrained().expect("load default tokenizer");

    let files = if cfg.files.is_empty() {
        let home = std::env::var("HOME").expect("HOME");
        let downloads = PathBuf::from(home).join("Downloads");
        let found = find_dataset_files(&downloads, "dataset").expect("scan ~/Downloads");
        if found.is_empty() {
            panic!("no ~/Downloads/dataset*.ron files found");
        }
        found
    } else {
        cfg.files.clone()
    };
    println!("dataset files: {} found", files.len());
    for f in &files { println!("  - {}", f.display()); }

    let ds: MailDataset = build_dataset(&files, &tok, cfg.seq_len).expect("build dataset");
    println!(
        "loaded {} samples, {} classes: {:?}",
        ds.samples.len(),
        ds.num_classes(),
        ds.categories
    );

    let order = shuffle_indices(ds.samples.len(), cfg.seed);
    let n_test = (ds.samples.len() / 10).max(1);
    let n_train = ds.samples.len() - n_test;
    let train: Vec<&Sample> = order[..n_train].iter().map(|&i| &ds.samples[i]).collect();
    let test:  Vec<Sample>  = order[n_train..].iter().map(|&i| Sample {
        ids: ds.samples[i].ids.clone(),
        labels: ds.samples[i].labels.clone(),
    }).collect();
    println!("train: {}  test: {}", train.len(), test.len());

    let opt = Optimizer::adam_default(cfg.lr);
    let num_classes = ds.num_classes();
    let mut model = build_model(&cfg, tok.vocab_size(), num_classes, opt);

    // ===== load checkpoint if requested =====
    let do_train: bool;
    match cfg.mode {
        MailMode::Fresh => {
            println!("starting from random initialisation");
            do_train = true;
        }
        MailMode::Continue => {
            let _ = model.load_from_file(&cfg.save_path)
                .unwrap_or_else(|e| panic!("could not load {}: {e}", cfg.save_path.display()));
            println!("loaded checkpoint, continuing training");
            do_train = true;
        }
        MailMode::EvalOnly => {
            let _ = model.load_from_file(&cfg.save_path)
                .unwrap_or_else(|e| panic!("could not load {}: {e}", cfg.save_path.display()));
            println!("loaded checkpoint, evaluation only (no training)");
            do_train = false;
        }
    }

    // ===== training loop =====
    if do_train {
        // Always need *some* stop criterion. If none set, default to 1 epoch.
        let stop = if cfg.stop.any_set() {
            cfg.stop
        } else {
            StopCriteria { max_epochs: Some(1), ..Default::default() }
        };

        let mut step_state = cfg.seed;
        let mut order: Vec<usize> = (0..train.len()).collect();
        // initial shuffle
        for i in (1..order.len()).rev() {
            let j = (xorshift64(&mut step_state) as usize) % (i + 1);
            order.swap(i, j);
        }

        let mut running = 0f32;
        let mut running_n = 0usize;
        let mut cursor = 0usize;
        let mut epoch  = 0usize;
        let mut step   = 0usize;
        let mut last_eval: Option<(f32, f32, f32)> = None;
        let mut stopped_for: Option<&'static str> = None;

        let check_eval_targets = |tl: f32, la: f32, ea: f32| -> Option<&'static str> {
            if let Some(t) = stop.target_loss      { if tl <= t { return Some("target_loss"); } }
            if let Some(t) = stop.target_label_acc { if la >= t { return Some("target_label_acc"); } }
            if let Some(t) = stop.target_exact_acc { if ea >= t { return Some("target_exact_acc"); } }
            None
        };

        loop {
            // hard step cap
            if let Some(m) = stop.max_steps {
                if step >= m { stopped_for = Some("max_steps"); break; }
            }

            // epoch boundary?
            if cursor >= order.len() {
                let (tl, la, ea) = evaluate(&mut model, &test, num_classes);
                last_eval = Some((tl, la, ea));
                println!(
                    "epoch {} done — test loss = {:.4}  label_acc = {:.2}%  exact_acc = {:.2}%",
                    epoch, tl, la * 100.0, ea * 100.0
                );
                if let Some(reason) = check_eval_targets(tl, la, ea) {
                    stopped_for = Some(reason); break;
                }
                epoch += 1;
                if let Some(m) = stop.max_epochs {
                    if epoch >= m { stopped_for = Some("max_epochs"); break; }
                }
                cursor = 0;
                for i in (1..order.len()).rev() {
                    let j = (xorshift64(&mut step_state) as usize) % (i + 1);
                    order.swap(i, j);
                }
                running = 0.0;
                running_n = 0;
            }

            let i = order[cursor];
            cursor += 1;
            let l = train_step(&mut model, &opt, train[i], num_classes);
            running   += l;
            running_n += 1;
            step += 1;

            if cfg.log_every > 0 && step % cfg.log_every == 0 {
                let cap = stop.max_steps.map(|m| format!("/{m}")).unwrap_or_default();
                println!(
                    "step {:>5}{}  epoch {} ({:>4}/{})  avg_loss = {:.4}",
                    step, cap,
                    epoch, cursor, order.len(),
                    running / running_n as f32
                );
            }

            // optional mid-epoch eval to check loss/accuracy targets
            if let Some(every) = stop.eval_every_steps {
                if every > 0 && step % every == 0 {
                    let (tl, la, ea) = evaluate(&mut model, &test, num_classes);
                    last_eval = Some((tl, la, ea));
                    println!(
                        "[eval @ step {}] test loss = {:.4}  label_acc = {:.2}%  exact_acc = {:.2}%",
                        step, tl, la * 100.0, ea * 100.0
                    );
                    if let Some(reason) = check_eval_targets(tl, la, ea) {
                        stopped_for = Some(reason); break;
                    }
                }
            }
        }

        let (tl, la, ea) = match last_eval {
            Some(v) => v,
            None => evaluate(&mut model, &test, num_classes),
        };
        println!(
            "training done ({}) — steps = {}  test loss = {:.4}  label_acc = {:.2}%  exact_acc = {:.2}%",
            stopped_for.unwrap_or("manual"), step,
            tl, la * 100.0, ea * 100.0
        );

        // ===== save =====
        if let Some(parent) = cfg.save_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        model.save_to_file(&cfg.save_path, &ds.categories).expect("save model");
        let bytes = std::fs::metadata(&cfg.save_path).map(|m| m.len()).unwrap_or(0);
        println!("saved model to {} ({} bytes)", cfg.save_path.display(), bytes);

        // round-trip check
        let mut loaded = build_model(&cfg, tok.vocab_size(), num_classes, opt);
        let _ = loaded.load_from_file(&cfg.save_path).expect("load model");
        let (tl, la, ea) = evaluate(&mut loaded, &test, num_classes);
        println!(
            "reloaded model — test loss = {:.4}  label_acc = {:.2}%  exact_acc = {:.2}%",
            tl, la * 100.0, ea * 100.0
        );
    } else {
        let (tl, la, ea) = evaluate(&mut model, &test, num_classes);
        println!(
            "eval-only — test loss = {:.4}  label_acc = {:.2}%  exact_acc = {:.2}%",
            tl, la * 100.0, ea * 100.0
        );
    }

    // ===== qualitative samples =====
    println!("--- sample predictions ---");
    for (k, s) in test.iter().take(3).enumerate() {
        let probs = forward_predict(&mut model, &s.ids);
        let pred:  Vec<u8> = probs.iter().map(|&p| if p > 0.5 { 1 } else { 0 }).collect();
        let truth: Vec<u8> = s.labels.iter().map(|&t| t as u8).collect();
        println!("  [{}] truth={:?}", k, truth);
        println!("       pred ={:?}", pred);
        let ps: Vec<String> = probs.iter().map(|p| format!("{:.2}", p)).collect();
        println!("       prob ={:?}", ps);
    }
}

/// Inputs for a single email prediction. All fields optional except `body`.
/// The trainer feeds the model `"{subject} | {sender_name} <{sender_email}> | {body}"`,
/// so for best results pass the same parts here. If you only have raw text,
/// put it in `body` and leave the others empty.
#[derive(Default, Debug, Clone)]
pub struct MailInput {
    pub subject: String,
    pub sender_name: String,
    pub sender_email: String,
    pub body: String,
}

impl MailInput {
    pub fn from_body(body: impl Into<String>) -> Self {
        Self { body: body.into(), ..Default::default() }
    }
    /// Format identical to training (`mail_to_text`): keeps the model in-distribution.
    pub fn to_text(&self) -> String {
        format!(
            "{} | {} <{}> | {}",
            self.subject.trim(),
            self.sender_name.trim(),
            self.sender_email.trim(),
            self.body
        )
    }
}

pub struct PredictConfig {
    pub model_path: PathBuf,
    pub seq_len: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub n_enc_layers: usize,
    pub threshold: f32,
    pub lr: f32,
}

impl PredictConfig {
    pub fn defaults_for(model_path: PathBuf) -> Self {
        Self {
            model_path,
            seq_len: 512,
            d_model: 64,
            n_heads: 4,
            d_ff: 128,
            n_enc_layers: 2,
            threshold: 0.5,
            lr: 1e-3,
        }
    }
}

pub struct PredictedLabel {
    pub name: String,
    pub probability: f32,
    pub predicted: bool,
}

/// Load the model from disk and run a single email through it. Returns one
/// `PredictedLabel` per category, in the order the categories were saved.
pub fn predict_one(cfg: &PredictConfig, input: &MailInput) -> Vec<PredictedLabel> {
    let tok = HfTokenizer::default_pretrained().expect("load default tokenizer");

    // peek at the header so we can build a matching empty model
    let header: TransformerHeader = Transformer::read_header(&cfg.model_path)
        .unwrap_or_else(|e| panic!("read header {}: {e}", cfg.model_path.display()));

    if header.d_model      != cfg.d_model      { panic!("d_model mismatch: file={} cfg={}",      header.d_model,      cfg.d_model); }
    if header.n_enc_layers != cfg.n_enc_layers { panic!("n_enc_layers mismatch: file={} cfg={}", header.n_enc_layers, cfg.n_enc_layers); }
    if header.src_vocab    != tok.vocab_size() { panic!("vocab mismatch: file={} tok={}",        header.src_vocab,    tok.vocab_size()); }

    let opt = Optimizer::adam_default(cfg.lr);
    let mcfg = MailTrainConfig {
        mode: MailMode::EvalOnly,
        stop: StopCriteria::default(),
        d_model: cfg.d_model,
        n_heads: cfg.n_heads,
        d_ff: cfg.d_ff,
        n_enc_layers: cfg.n_enc_layers,
        seq_len: cfg.seq_len,
        lr: cfg.lr,
        seed: 0,
        save_path: cfg.model_path.clone(),
        files: Vec::new(),
        log_every: 0,
    };
    let mut model = build_model(&mcfg, tok.vocab_size(), header.num_classes, opt);
    let categories = model.load_from_file(&cfg.model_path)
        .unwrap_or_else(|e| panic!("load {}: {e}", cfg.model_path.display()));

    // tokenize same way the trainer does
    let text = input.to_text();
    let mut ids = tok.encode(&text);
    ids.insert(0, tok.bos_id());
    let ids = tok.pad_to(ids, cfg.seq_len);

    let probs = forward_predict(&mut model, &ids);
    categories.into_iter().enumerate().map(|(i, name)| PredictedLabel {
        name,
        probability: probs[i],
        predicted: probs[i] >= cfg.threshold,
    }).collect()
}
