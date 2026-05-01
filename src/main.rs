mod model;
mod train;
mod utils;

use crate::train::{TrainConfig, train_copy_task};

fn main() {
    let mut cfg = TrainConfig::small_copy();
    cfg.steps     = 2000;
    cfg.lr        = 1e-3;
    cfg.log_every = 50;
    cfg.use_adam  = true;
    train_copy_task(cfg);
}
